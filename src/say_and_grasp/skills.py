"""Robot skills the assistant calls: look, pick, place, go home.

    bench = Bench(port="/dev/ttyACM0")
    bench.connect()
    bench.pick("side cutters")          # finds it, grasps, checks it's in the jaws, retries
    bench.place_xy(0.20, -0.15)         # puts it down at a table position (metres, arm frame)
    bench.go_home()
    bench.close()

Every pick checks the gripper afterwards: if the jaws closed all the way, nothing is in them,
so it opens, goes home, takes a fresh picture and tries again.
"""
from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

from . import events
from .arm import Arm, load_home
from .calibration import (
    REACH_MAX,
    REACH_MIN,
    load_calib,
    pixel_dir_to_base,
    pixel_to_base,
    shade_reach,
    table_z_at,
)
from .camera import CameraStream, capture
from .kinematics import Kinematics
from .paths import CALIB_FILE, CONFIG_DIR, URDF_FILE, WRIST_FILE, output

BIAS_FILE = CONFIG_DIR / "bias.json"  # learned overhead-vs-wrist offset (see Bench._learn_bias)
from .sam import OVERHEAD_SIDE, Detection, GraspTarget, draw_overlay, grasp_from_mask, segment, segment_robust


def _wrap90(a: float) -> float:
    """Angle difference for directions that repeat every 180 deg, in (-90, 90]."""
    a = (a + 90.0) % 180.0 - 90.0
    return a


def _elongated(mask: np.ndarray, ratio: float = 1.3) -> bool:
    """True if the mask has a clear long axis (a square object has no grasp direction to correct)."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] < 1:
        return False
    a, b, c = m["mu20"] / m["m00"], m["mu11"] / m["m00"], m["mu02"] / m["m00"]
    l1 = (a + c) / 2 + math.sqrt(((a - c) / 2) ** 2 + b * b)
    l2 = (a + c) / 2 - math.sqrt(((a - c) / 2) ** 2 + b * b)
    return l2 > 0 and math.sqrt(l1 / l2) >= ratio


WRIST_SAM_SIDE = 384  # px; the wrist view is close up, so a small upload keeps SAM fast
SETTLE_WAYPOINT = 0.1  # s: short pause at pass-through points
SETTLE_PRECISE = 0.3  # s: settle where position matters (grasp, place, photos)


class SkillError(RuntimeError):
    """A skill couldn't be done (not found, out of reach, missed after retries)."""


@dataclass
class Seen:
    """An object found in the overhead view, located in the arm frame."""

    prompt: str
    det: Detection
    grasp: GraspTarget
    x: float
    y: float
    z: float  # table height under it
    close_dir: float  # degrees, arm frame
    frame: np.ndarray

    @property
    def reach(self) -> float:
        return math.hypot(self.x, self.y)


class Bench:
    def __init__(
        self,
        port: str = "/dev/ttyACM0",
        robot_id: str = "my_follower",
        camera: str | None = None,
        calib_path: str = str(CALIB_FILE),
        urdf: str = str(URDF_FILE),
        speed: float = 70.0,
        grasp_height_mm: float = 5.0,
        approach_mm: float = 50.0,
        gripper_open: float = 60.0,
        gripper_closed: float = 5.0,
        empty_margin: float = 2.0,  # jaw units above the measured empty reading that mean "holding"
        yaw_offset: float = 0.0,
        dry_run: bool = False,
        stream_overhead: bool = True,
        log=print,
    ):
        self.calib = load_calib(calib_path)
        self.camera = camera or self.calib["camera"]
        # The overhead resolution the calibration was made at (pixel positions only match at that size).
        self.res = tuple(int(v) for v in self.calib.get("image_size", [640, 480]))
        self.sam_side = min(OVERHEAD_SIDE, max(self.res))
        self.kin = Kinematics(urdf)
        self.arm = None if dry_run else Arm(port, robot_id, speed_dps=speed)
        self.gh = grasp_height_mm / 1000
        self.hov = approach_mm / 1000
        self.g_open, self.g_closed, self.empty_margin = gripper_open, gripper_closed, empty_margin
        self.yaw_offset = yaw_offset
        self.dry_run = dry_run
        self._log = log
        self.holding: Seen | None = None
        # Wrist camera alignment (sag-calibrate --wrist). Without it, picks use the overhead estimate only.
        self.wrist = json.loads(WRIST_FILE.read_text()) if WRIST_FILE.exists() else None
        self.name_hint = None  # optional: prompt -> a fuller name it was seen under before (memory)
        self.locator = None  # optional: (frame, prompt) -> rough (u, v) of it in the frame (Muse Spark)
        # What the wrist camera keeps correcting the overhead estimate by, learned pick by pick.
        try:
            self.bias = json.loads(BIAS_FILE.read_text())
        except Exception:
            self.bias = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0, "n": 0}
        self.align_tol = 0.006  # good enough below 6 mm: well inside the open jaws' clearance
        self.align_max = 0.04  # beyond this a correction is "following" a moved object, not fine-tuning
        self.follow_max = 0.12  # the wrist camera seeing it further than this is probably something else
        self.align_tol_deg = 10.0  # jaws within 10 deg of square is fine
        self.trust = 0.015  # a correction smaller than this is applied without checking it again
        self.lean = 0.0
        self.wrist_stream: CameraStream | None = None
        self.overhead_stream: CameraStream | None = None
        # Keep the overhead camera open: autofocus has time to settle (sharp photos) and each photo
        # is instant. Also needed when recording episodes.
        self._want_overhead_stream = stream_overhead

    _STATUS = {"look": "Looking", "align": "Wrist camera", "pick": "Grasp", "hand": "Looking for your hand",
               "offer": "Handover", "place": "Put down", "put away": "Tidying", "point": "Pointing"}

    def log(self, msg: str) -> None:
        """Print, and show the skill's progress in the brain view."""
        self._log(msg)
        if msg.startswith("["):
            tag, _, rest = msg[1:].partition("]")
            if tag in self._STATUS and tag != "look":
                events.publish("status", text=f"{self._STATUS[tag]}: {rest.strip()}")

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if self.arm:
            self.arm.connect()
            self.arm.go_home(gripper=self.g_closed)
            self._learn_empty_grip()
        if self.wrist and not self.dry_run:
            try:
                self.wrist_stream = CameraStream(self.wrist["camera"])
            except Exception as e:  # fall back to opening it per photo
                self.log(f"[wrist] streaming unavailable ({e}); using single shots")
                self.wrist_stream = None
        if self._want_overhead_stream and not self.dry_run:
            try:
                self.overhead_stream = CameraStream(self.camera, width=self.res[0], height=self.res[1])
            except Exception as e:
                self.log(f"[overhead] streaming unavailable ({e}); using single shots")
                self.overhead_stream = None

    def close(self) -> None:
        for st in (self.wrist_stream, self.overhead_stream):
            if st:
                st.close()
        if self.arm:
            self.arm.disconnect()

    def wrist_photo(self) -> np.ndarray:
        """A wrist-camera frame taken after this call (i.e. after the arm has settled)."""
        if self.wrist_stream:
            try:
                return self.wrist_stream.latest()
            except RuntimeError:
                self.log("[wrist] stream stalled; switching to single shots")
                self.wrist_stream.close()
                self.wrist_stream = None
        return capture(self.wrist["camera"])

    def go_home(self) -> None:
        if self.arm:
            self.arm.go_home(gripper=self.g_closed if not self.holding else None)

    # ------------------------------------------------------------------ perception
    def snapshot(self) -> np.ndarray:
        """Overhead picture. The arm should be home (out of view) when this is called."""
        if self.overhead_stream:
            try:
                return self.overhead_stream.latest()
            except RuntimeError:
                self.log("[overhead] stream stalled; switching to single shots")
                self.overhead_stream.close()
                self.overhead_stream = None
        return capture(self.camera, width=self.res[0], height=self.res[1])

    def overhead_peek(self) -> np.ndarray | None:
        return self.overhead_stream.peek() if self.overhead_stream else None

    def wrist_peek(self) -> np.ndarray | None:
        return self.wrist_stream.peek() if self.wrist_stream else None

    def look(self, prompt: str, frame: np.ndarray | None = None) -> Seen:
        """Find `prompt` in the overhead view; pick the most confident/largest match."""
        frame = self.snapshot() if frame is None else frame
        dets, used = segment_robust(frame, prompt, max_side=self.sam_side)
        if not dets:
            # SAM occasionally misses on one frame (glare, exposure); one fresh photo before giving up.
            time.sleep(0.5)
            frame = self.snapshot()
            dets, used = segment_robust(frame, prompt, max_side=self.sam_side)
        if not dets and self.name_hint:
            # The name it was asked for came up empty ("brush"); try the fuller name this object had
            # when it was last seen ("black brush").
            alt = self.name_hint(prompt)
            if alt and alt.lower() != prompt.lower():
                self.log(f"[look] nothing for '{prompt}'; trying '{alt}', as I last saw it")
                dets, used = segment_robust(frame, alt, max_side=self.sam_side)
        if not dets and self.locator:
            dets, used = self._zoom_look(frame, prompt)
        if not dets:
            raise SkillError(f"I can't see any {prompt} on the bench.")
        dets.sort(key=lambda d: (d.confidence or 0, d.mask.sum()), reverse=True)
        det = dets[0]
        g = grasp_from_mask(det.mask)
        x, y, z = pixel_to_base(g.u, g.v, self.calib)
        close_dir = pixel_dir_to_base(g.u, g.v, g.yaw_deg, self.calib) + self.yaw_offset
        if self.bias.get("n"):  # the offset the wrist camera has been correcting on earlier picks
            x, y = x + self.bias["dx"], y + self.bias["dy"]
            close_dir += self.bias["dyaw"]
        seen = Seen(prompt, det, g, x, y, z, close_dir, frame)
        vis = draw_overlay(shade_reach(frame, self.calib), [det], [g])
        cv2.imwrite(str(output("look.png")), vis)
        events.publish("overhead_sam", image=vis, label=f"SAM 3.1 found '{used}'")
        self.log(f"[look] {prompt}: pixel ({g.u:.0f},{g.v:.0f}) -> ({x*1000:.0f}, {y*1000:.0f}) mm, "
                 f"reach {seen.reach*1000:.0f} mm" + (f" ({len(dets)} matches)" if len(dets) > 1 else ""))
        return seen

    def _zoom_look(self, frame: np.ndarray, prompt: str) -> tuple[list, str]:
        """Last try for small or thin things SAM misses in the whole view (tweezers seen end-on):
        Muse Spark points out roughly where it is, then SAM 3.1 looks again at a close-up crop of
        that spot, where the object is several times bigger."""
        try:
            spot = self.locator(frame, prompt)
        except Exception as e:
            self.log(f"[look] couldn't ask where the {prompt} is ({e.__class__.__name__})")
            return [], prompt
        if spot is None:
            return [], prompt
        h, w = frame.shape[:2]
        side = int(min(h, w) * 0.45)
        u, v = spot
        x0 = int(np.clip(u - side / 2, 0, w - side))
        y0 = int(np.clip(v - side / 2, 0, h - side))
        crop = frame[y0:y0 + side, x0:x0 + side]
        self.log(f"[look] Muse Spark says the {prompt} is near pixel ({u:.0f},{v:.0f}); zooming in")
        dets, used = segment_robust(crop, prompt, max_side=side)
        out = []
        for d in dets:
            m = np.zeros((h, w), dtype=bool)
            m[y0:y0 + side, x0:x0 + side] = d.mask
            l, t, r, b = d.box
            out.append(Detection(d.object_id, (l + x0, t + y0, r + x0, b + y0), m, d.confidence))
        if out:
            self.log(f"[look] found it in the close-up")
        return out, used

    # ------------------------------------------------------------------ planning
    def _solve(self, name: str, x: float, y: float, z: float, close_dir: float,
               lean: float | None = None) -> np.ndarray:
        """Joints for the grasp centre at (x, y, z). Straight down where possible, leaning in or out
        near the edges of reach. `lean` pins the lean (so hover and grasp match); it falls back to
        any lean that works. The lean used is left in self.lean."""
        reach = math.hypot(x, y)
        r = self.kin.grasp(x, y, z, close_dir, lean=lean)
        if not r.ok and lean is not None:
            r = self.kin.grasp(x, y, z, close_dir)
        if not r.ok:
            where = "too close to my base" if reach < 0.15 else "out of my reach"
            raise SkillError(f"{name}: {reach*100:.0f} cm from the base is {where}.")
        self.lean = r.tilt_deg if r.tilt_deg > 1 else 0.0
        return r.joints

    def _plan(self, seen: "Seen") -> tuple[np.ndarray, np.ndarray]:
        """(hover, grasp) joints for a seen object, with the same lean for both so the descent is
        a straight drop along the gripper."""
        grasp = self._solve("grasp", seen.x, seen.y, seen.z + self.gh, seen.close_dir)
        lean = self.lean
        above = self._solve("above object", seen.x, seen.y, seen.z + self.hov, seen.close_dir, lean=lean)
        if self.lean != lean:  # hover needed a different lean: re-solve the grasp to match it
            grasp = self._solve("grasp", seen.x, seen.y, seen.z + self.gh, seen.close_dir, lean=self.lean)
        if self.lean:
            self.log(f"[plan] leaning the gripper {self.lean:.0f} deg to reach it")
        return above, grasp

    def _camera_centre(self, joints: np.ndarray, height: float) -> np.ndarray:
        """Where, on the object's level, the wrist camera's 'centred' point is relative to the grasp
        centre. Zero when the gripper points straight down; when it leans, the camera looks along
        the lean and its centre lands that much further along."""
        T = self.kin.fk_tcp(joints)
        z = T[:3, 2]
        if z[2] > -0.2:
            return np.zeros(2)
        return height * z[:2] / -z[2]

    # ------------------------------------------------------------------ wrist camera
    def tool_xy_axes(self, joints: np.ndarray) -> np.ndarray:
        """2x2: columns are the tool frame's x and y axes in the table plane (gripper pointing down)."""
        return self.kin.fk(joints)[:2, :2]

    def wrist_offset(self, prompt: str, joints: np.ndarray, save_as: str = "wrist.png", scale: float = 1.0):
        """From a wrist photo at hover: where the object is relative to the camera's centre point
        (base frame, metres), and how far the gripper should turn (degrees) so the jaws close across it.
        Returns (offset_xy, turn_deg) or None if the object isn't visible. turn_deg is None when the
        wrist calibration has no angle data or the object is too square to have a direction.
        `scale` < 1 when looking from higher than the calibrated hover (things look smaller)."""
        if not self.wrist:
            return None
        img = self.wrist_photo()
        dets, _ = segment_robust(img, prompt, max_side=WRIST_SAM_SIDE)  # close-up: small image is plenty
        target = np.array(self.wrist["target_px"], dtype=float)
        if not dets:
            cv2.imwrite(str(output(save_as)), img)
            return None
        pairs = [(d, grasp_from_mask(d.mask)) for d in dets]
        pairs = [(d, g) for d, g in pairs if g]
        if not pairs:
            return None
        # the object we're over is the one nearest where a centred object appears
        det, g = min(pairs, key=lambda p: np.hypot(p[1].u - target[0], p[1].v - target[1]))
        vis = draw_overlay(img, [d for d, _ in pairs], [g for _, g in pairs])
        cv2.drawMarker(vis, tuple(int(t) for t in target), (0, 255, 0), cv2.MARKER_CROSS, 24, 2)
        cv2.imwrite(str(output(save_as)), vis)
        err_px = np.array([g.u, g.v]) - target
        px_per_m = np.array(self.wrist["px_per_m"], dtype=float) * scale  # 2x2: pixel error per metre
        offset = self.tool_xy_axes(joints) @ np.linalg.solve(px_per_m, err_px)
        wrist_label = f"SAM 3.1 wrist: {np.linalg.norm(offset)*1000:.0f} mm off centre"

        turn = None
        if "target_yaw_px" in self.wrist and _elongated(det.mask):
            # how far the object's grasp direction is rotated from where the jaws close, in the image
            e = _wrap90(g.yaw_deg - self.wrist["target_yaw_px"])
            turn = -e / self.wrist["yaw_gain"]  # gripper turn that cancels it (gain = image deg per gripper deg)
            wrist_label += f", {turn:+.0f} deg"
        events.publish("wrist_sam", image=vis, label=wrist_label)
        return offset, turn

    def _align(self, seen: Seen, above: np.ndarray, rounds: int = 3) -> tuple[float, float, np.ndarray, str]:
        """Hovering over the object: measure with the wrist camera, shift/turn, re-measure.

        Returns (x, y, hover_joints, status), status one of:
          "centred"  jaws centred and square (or a small last correction, trusted): grasp
          "close"    not perfect but within ~12 mm / 20 deg after all rounds: grasp
          "lost"     the wrist camera can't see it even from higher up (or it's beyond follow range)
          "stuck"    it keeps ending up off-centre: don't grasp blind
        Speed matters: a correction under 15 mm is applied and trusted without another photo, and
        if the object isn't in view it rises for a wider look instead of going back for an overhead
        photo. If it sees the object further away (someone moved it), it follows it.
        """
        x, y = seen.x, seen.y
        widened = False
        for i in range(rounds + 1):
            wide = False
            m = self.wrist_offset(seen.prompt, above, save_as=f"wrist_{i + 1}.png")
            if m is None and not widened:
                # Not in view: the overhead estimate was further off than the close-up view is wide.
                # Rise a little for a wider look rather than going all the way back, keeping the same
                # gripper angle (a different lean would point the camera somewhere else).
                widened = wide = True
                high = None
                for rise in (0.04, 0.03, 0.02):
                    r = self.kin.grasp(x, y, seen.z + self.hov + rise, seen.close_dir, lean=self.lean or None)
                    if r.ok and abs((r.tilt_deg if r.tilt_deg > 1 else 0.0) - self.lean) < 1:
                        high = r.joints
                        break
                if high is not None:
                    self.log("[align] not in view; rising for a wider look")
                    self.arm.move(high, settle_s=SETTLE_PRECISE)
                    d0 = float(self.wrist.get("cam_height", 0.10))  # camera to table at normal hover
                    m = self.wrist_offset(seen.prompt, high, save_as=f"wrist_{i + 1}w.png", scale=d0 / (d0 + rise))
                    look_from, look_h = high, self.hov + rise
            else:
                look_from, look_h = above, self.hov
            if m is None:
                self.log("[align] can't see it in the wrist camera")
                return x, y, above, "lost"
            d, turn = m
            e = d + self._camera_centre(look_from, look_h)  # where it is relative to the grasp centre
            n = float(np.hypot(*e))
            t = abs(turn) if turn is not None else 0.0
            status = f"{n*1000:.0f} mm off" + (f", {turn:+.0f} deg" if turn is not None else "")
            if not wide and n < self.align_tol and t < self.align_tol_deg:
                self.log(f"[align] centred and square ({status})")
                return x, y, above, "centred"
            if n > self.follow_max:
                self.log(f"[align] what I see is {n*100:.0f} cm away; probably not it")
                return x, y, above, "lost"
            if i == rounds:
                ok = n < 2 * self.align_tol and t < 2 * self.align_tol_deg
                self.log(f"[align] {status} after {rounds} corrections; " + ("close enough" if ok else "not grabbing blind"))
                return x, y, above, "close" if ok else "stuck"
            nx, ny = x + e[0], y + e[1]
            cd = seen.close_dir + turn if (turn is not None and self.align_tol_deg <= t <= 60) else seen.close_dir
            try:
                new_above = self._solve("above object", nx, ny, seen.z + self.hov, cd, lean=self.lean)
            except SkillError as err:
                self.log(f"[align] it's moved out of reach ({err})")
                return x, y, above, "lost"
            x, y, seen.close_dir, above = nx, ny, cd, new_above
            trusted = not wide and n < self.trust and t < 12
            self.log(f"[align] {status}; " + ("following it" if n > self.align_max else
                                             "small correction, going for it" if trusted else "correcting"))
            self.arm.move(above, settle_s=SETTLE_WAYPOINT if trusted else SETTLE_PRECISE)
            if trusted:
                return x, y, above, "centred"
        return x, y, above, "stuck"

    # ------------------------------------------------------------------ learning from corrections
    def _learn_bias(self, before: tuple[float, float, float], seen: "Seen") -> None:
        """The wrist camera's correction tells us how far off the overhead estimate was. When that
        error is consistent (a slightly shifted camera, a small calibration offset), fold it into
        future overhead estimates, so the arm arrives closer and needs fewer correction rounds.
        Large corrections are skipped: they mean someone moved the object, not a camera offset."""
        dx, dy = seen.x - before[0], seen.y - before[1]
        dyaw = _wrap90(seen.close_dir - before[2])
        if math.hypot(dx, dy) > 0.06 or abs(dyaw) > 35:
            return
        b = self.bias
        w = 0.5 if b.get("n", 0) < 3 else 0.3  # learn fast at first, then settle
        b["dx"] = float(np.clip(b["dx"] + w * dx, -0.05, 0.05))
        b["dy"] = float(np.clip(b["dy"] + w * dy, -0.05, 0.05))
        b["dyaw"] = float(np.clip(b["dyaw"] + w * dyaw, -30, 30))
        b["n"] = b.get("n", 0) + 1
        try:
            BIAS_FILE.write_text(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in b.items()},
                                            indent=2))
        except OSError:
            pass
        self.log(f"[learn] overhead estimate was off by {math.hypot(dx, dy)*1000:.0f} mm, {dyaw:+.0f} deg; "
                 f"correction now ({b['dx']*1000:+.0f}, {b['dy']*1000:+.0f}) mm, {b['dyaw']:+.0f} deg")

    # ------------------------------------------------------------------ actions
    def _learn_empty_grip(self) -> None:
        """What 'closed on nothing' looks like on this gripper: jaw reading and motor load. Thin
        things (tweezers on their side, a wire) stop the jaws only a hair short of that, so the
        grip check compares against this instead of a fixed margin."""
        time.sleep(0.3)
        pos = [self.arm.gripper_pos() for _ in range(5)]
        loads = [ld[5] for ld in (self.arm.read_load() for _ in range(5)) if ld is not None]
        self.empty_width = float(np.median(pos))
        self.empty_load = float(np.median(np.abs(loads))) if loads else None
        self.log(f"[grip] empty gripper closes to {self.empty_width:.1f}"
                 + (f", load {self.empty_load:.0f}" if self.empty_load is not None else ""))

    def _holding_something(self) -> tuple[bool, str]:
        """After closing: is something between the jaws? Returns (holding, explanation)."""
        width = float(np.median([self.arm.gripper_pos() for _ in range(3)]))
        empty = getattr(self, "empty_width", self.g_closed)
        gap = width - empty
        ld = self.arm.read_load()
        load = abs(float(ld[5])) if ld is not None else None
        base = getattr(self, "empty_load", None)
        info = f"gripper at {width:.1f} (empty {empty:.1f})" + (
            f", load {load:.0f}" + (f" (empty {base:.0f})" if base is not None else "") if load is not None else "")
        if gap > self.empty_margin:
            return True, info
        # Thin object: the jaws stop just short of closed, and the motor is pushing against it.
        if gap > 0.6 and load is not None and base is not None and load > base + 60:
            return True, info + ", thin object"
        return False, info

    def pick(self, prompt: str, retries: int = 2, pause: bool = False, for_hand: bool = False,
             frame: np.ndarray | None = None) -> Seen:
        """Find and grasp `prompt`, as one fluid motion.

        The overhead photo is taken while the arm is home (out of view); the arm starts lifting
        while SAM works on it. At hover the wrist camera centres and squares the jaws, then it
        grasps. If the grasp misses, it rises back to hover and looks again with the wrist camera
        right there (the object has usually only been nudged), re-aligns and tries again. Only if
        the object has left the wrist camera's view does it step back for a fresh overhead look.
        """
        t0 = time.time()
        given = frame is not None  # e.g. a clean photo from the start of a tidy-up
        frame = self.snapshot() if frame is None else frame
        if self.dry_run:
            seen = self.look(prompt, frame)
            self._plan(seen)
            self.log("[pick] dry run: plan OK, not moving")
            return seen

        arm = self.arm
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(self.look, prompt, frame)
                arm.leave_home(gripper=self.g_open)  # lift the wrist while SAM is thinking
                seen = fut.result()
        except SkillError:
            # Maybe the arm itself was hiding it (mid tidy-up): move out of the way and look again.
            if not given and np.max(np.abs(arm.q_cmd - load_home())) < 15:
                raise
            self.log("[pick] not found; moving out of the camera's way and looking again")
            arm.go_clear(gripper=self.g_open)
            seen = self.look(prompt)
        above, _ = self._plan(seen)
        arm.move(above, gripper=self.g_open, settle_s=SETTLE_PRECISE)

        for attempt in range(retries + 1):
            if attempt:
                self.log(f"[pick] attempt {attempt + 1}/{retries + 1}")
            if self.wrist:
                before = (seen.x, seen.y, seen.close_dir)
                x, y, above, status = self._align(seen, above)  # may also turn seen.close_dir
                seen.x, seen.y = x, y  # so put_back() returns it to where it really was
                if attempt == 0 and status in ("centred", "close"):
                    self._learn_bias(before, seen)
                if status in ("lost", "stuck"):
                    # Don't close on nothing: step back and find it again from above.
                    self.log("[pick] lost track of it; stepping back for an overhead look")
                    arm.go_clear(gripper=self.g_open)
                    seen = self.look(prompt)
                    above, _ = self._plan(seen)
                    arm.move(above, settle_s=SETTLE_PRECISE)
                    x, y, above, status = self._align(seen, above)
                    seen.x, seen.y = x, y
                    if status in ("lost", "stuck"):
                        raise SkillError(f"I can see the {prompt} but I can't line up with it. "
                                         "Could you move it a little closer to the middle of my reach?")
            grasp = self._solve("grasp", seen.x, seen.y, seen.z + self.gh, seen.close_dir, lean=self.lean)
            if pause and input("[pick] hovering; Enter to grasp, q to abort: ").strip().lower() == "q":
                arm.go_home()
                raise SkillError("aborted")
            arm.move(grasp, settle_s=SETTLE_PRECISE)
            arm.set_gripper(self.g_closed)
            held, info = self._holding_something()
            if held:
                if for_hand:
                    self._start_hand_search((seen.x, seen.y))  # look for your hand while it lifts
                arm.move(above, settle_s=SETTLE_WAYPOINT)  # lift
                self.log(f"[pick] holding {prompt} ({info}) after {time.time() - t0:.1f}s")
                self.holding = seen
                return seen
            self.log(f"[pick] missed ({info}); re-checking from above")
            arm.set_gripper(self.g_open)
            arm.move(above, settle_s=SETTLE_PRECISE)  # back up to hover, right where we were
        raise SkillError(f"I tried {retries + 1} times but couldn't grab the {prompt}.") from None

    def place_xy(self, x: float, y: float) -> None:
        """Put down what we're holding at (x, y) on the table, then step back."""
        if self.holding is None and not self.dry_run:
            raise SkillError("I'm not holding anything.")
        cd = self.holding.close_dir if self.holding else 0.0
        z = table_z_at(x, y, self.calib)
        above = self._solve("above place", x, y, z + self.hov, cd)
        down = self._solve("place", x, y, z + self.gh + 0.003, cd)
        if self.dry_run:
            self.log(f"[place] dry run: ({x*1000:.0f}, {y*1000:.0f}) mm OK")
            return
        self.arm.move(above, settle_s=SETTLE_WAYPOINT)
        self.arm.move(down, settle_s=SETTLE_PRECISE)
        self.arm.set_gripper(self.g_open)
        self.arm.move(above, settle_s=SETTLE_WAYPOINT)
        self.log(f"[place] put down at ({x*1000:.0f}, {y*1000:.0f}) mm")
        self.holding = None

    # ------------------------------------------------------------------ hand-to-hand
    def find_hand(self, frame: np.ndarray | None = None,
                  exclude_xy: tuple[float, float] | None = None) -> tuple[float, float] | None:
        """Look for the user's hand in the overhead view -> (x, y) in the arm frame, or None.

        Works with the arm in view: anything within 8 cm of `exclude_xy` (where the gripper is) is
        ignored, so the robot's own gripper is never mistaken for a hand. The hand is held above
        the table, so its position is a little off (parallax); the drop is forgiving of that."""
        frame = self.snapshot() if frame is None else frame
        best = None
        for p in ("hand", "open hand", "palm"):
            for det in segment(frame, p, max_side=self.sam_side):
                ys, xs = np.nonzero(det.mask)
                if len(xs) < 50:
                    continue
                u, v = float(xs.mean()), float(ys.mean())
                x, y, _ = pixel_to_base(u, v, self.calib)
                if exclude_xy is not None and math.hypot(x - exclude_xy[0], y - exclude_xy[1]) < 0.08:
                    continue
                if best is None or len(xs) > best[0]:
                    best = (len(xs), x, y, u, v, det.mask)
            if best:
                break
        if best is None:
            return None
        _, x, y, u, v, mask = best
        overlay = frame.copy()
        overlay[mask] = (0.5 * overlay[mask] + (0, 110, 255)).clip(0, 255).astype(np.uint8)
        cv2.circle(overlay, (int(u), int(v)), 8, (0, 0, 255), -1)
        cv2.imwrite(str(output("hand.png")), overlay)
        events.publish("overhead_sam", image=overlay, label="SAM 3.1 found your hand")
        self.log(f"[hand] pixel ({u:.0f},{v:.0f}) -> ({x*1000:.0f}, {y*1000:.0f}) mm")
        return x, y

    def _start_hand_search(self, exclude_xy: tuple[float, float]) -> None:
        """Start looking for the user's hand in the background (the first look of offer())."""
        import threading

        box: dict = {}

        def run():
            try:
                box["hand"] = self.find_hand(exclude_xy=exclude_xy)
            except Exception:
                box["hand"] = None

        th = threading.Thread(target=run, daemon=True)
        th.start()
        self._hand_search = (th, box, time.time())

    def _rise(self, q: np.ndarray, x: float, y: float, height: float) -> None:
        """After letting go: lift straight up from pose q, keeping the wrist's turn exactly as it is
        (the jaws are empty, so re-solving their angle would only make the wrist spin)."""
        try:
            up, _, _ = self._offer_pose(x, y, height)
        except SkillError:
            return
        up = up.copy()
        up[4] = q[4]  # wrist roll unchanged
        if np.max(np.abs(up - q)) < 60:  # only a short lift, never a big swing
            self.arm.move(up, settle_s=0.05)

    def _offer_pose(self, x: float, y: float, height: float) -> tuple[np.ndarray, float, float]:
        """Joints to hold the tool `height` above the table at (x, y), leaning if needed; if it's
        still too far, as close as it can get along the line towards it."""
        cd = self.holding.close_dir if self.holding else 0.0
        r = math.hypot(x, y)
        for k in range(12):  # shrink the reach 2 cm at a time until it can get there
            s = max(0.0, r - 0.02 * k) / max(r, 1e-6)
            px, py = x * s, y * s
            if math.hypot(px, py) < 0.10:
                break
            z0 = table_z_at(px, py, self.calib)
            for h in (height, height - 0.02, height - 0.04):
                res = self.kin.grasp(px, py, z0 + h, cd)
                if res.ok:
                    return res.joints, px, py
        raise SkillError("I can't hold it out that far.")

    def offer(self, wait_hand_s: float = 8.0, height_mm: float = 90.0, say=print,
              fallback_xy: tuple[float, float] | None = None) -> str:
        """Drop the held tool into the user's open hand.

        Straight from the pick (no trip home): looks for a hand in the overhead view while still
        holding the tool, moves over it and lets go a few centimetres above the palm. If no hand
        shows up in `wait_hand_s` seconds, puts the tool down at `fallback_xy` instead.
        Returns "hand" or "bench"."""
        if self.holding is None and not self.dry_run:
            raise SkillError("I'm not holding anything.")
        here = (self.holding.x, self.holding.y) if self.holding else None
        t0 = time.time()
        found = None
        asked = closer = False
        far = 0.42  # beyond this (even leaning) a drop would miss the palm
        pending = getattr(self, "_hand_search", None)
        self._hand_search = None
        while time.time() - t0 < wait_hand_s:
            if pending and time.time() - pending[2] < 6:  # the look started while it was lifting
                pending[0].join(timeout=5)
                found = pending[1].get("hand")
                pending = None
            else:
                found = self.find_hand(exclude_xy=here)
            if found and math.hypot(*found) <= far:
                break
            if found:
                self.log(f"[offer] hand at {math.hypot(*found)*100:.0f} cm: out of reach")
                if not closer:
                    say("Bring your hand a little closer.")
                    closer = True
                found = None
            elif not asked:
                say("Hold out your hand.")
                asked = True
            time.sleep(0.2)
        if not found:
            self.log("[offer] no hand seen; putting it down instead")
            if fallback_xy is None:
                raise SkillError("I couldn't see your hand.")
            self.place_xy(*fallback_xy)
            return "bench"
        q, ox, oy = self._offer_pose(*found, height_mm / 1000)
        if self.dry_run:
            self.log(f"[offer] dry run: would drop it at ({ox*1000:.0f}, {oy*1000:.0f}) mm")
            self.holding = None
            return "hand"
        arm = self.arm
        old = arm.speed
        arm.speed = min(old, 60)  # moving towards a person: a touch gentler
        try:
            arm.move(q, settle_s=0.3)
        finally:
            arm.speed = old
        time.sleep(0.3)
        arm.set_gripper(self.g_open)  # drop it into the palm
        self.holding = None
        time.sleep(0.2)
        self._rise(q, ox, oy, height_mm / 1000 + 0.05)  # straight up, clear of your hand
        self.log(f"[offer] dropped into your hand at ({ox*1000:.0f}, {oy*1000:.0f}) mm, "
                 f"{time.time() - t0:.1f}s after the pick")
        return "hand"

    # ------------------------------------------------------------------ point / put away
    def point_at(self, x: float, y: float, height_mm: float = 80.0) -> bool:
        """Hover over (x, y) and give a little wiggle: "it's right here". Returns False when the spot
        is out of reach (it then points from the edge of its reach, towards the spot)."""
        q, px, py = self._offer_pose(x, y, height_mm / 1000)
        reachable = math.hypot(px - x, py - y) < 0.01
        if self.dry_run:
            self.log(f"[point] dry run: ({px*1000:.0f}, {py*1000:.0f}) mm")
            return reachable
        arm = self.arm
        arm.leave_home()
        arm.move(q, settle_s=0.2)
        for d in (10, -10, 10, -10, 0):  # wiggle the wrist
            w = q.copy()
            w[3] += d
            arm.move(w, settle_s=0)
        self.log(f"[point] at ({px*1000:.0f}, {py*1000:.0f}) mm" + ("" if reachable else " (edge of reach)"))
        return reachable

    def drop_into(self, container: str, rim_mm: float = 50.0, clearance_mm: float = 40.0,
                  box: "Seen | None" = None) -> Seen:
        """Drop what we're holding into a container found by name ("tray", "cup", "bin").

        The container is found fresh each time with SAM 3.1, so it can sit anywhere in reach. The
        tool is let go `clearance_mm` above the container's rim (`rim_mm` above the table)."""
        if self.holding is None and not self.dry_run:
            raise SkillError("I'm not holding anything.")
        box = box or self.look(container)  # a tidy-up passes the tray it already found
        q, qx, qy = self._offer_pose(box.x, box.y, (rim_mm + clearance_mm) / 1000)
        if math.hypot(qx - box.x, qy - box.y) > 0.03:
            raise SkillError(f"The {container} is out of my reach; can you move it closer?")
        if self.dry_run:
            self.log(f"[put away] dry run: into the {container} at ({box.x*1000:.0f}, {box.y*1000:.0f}) mm")
            self.holding = None
            return box
        arm = self.arm
        arm.move(q, settle_s=0.15)
        arm.set_gripper(self.g_open)
        self.holding = None
        time.sleep(0.15)
        self._rise(q, box.x, box.y, (rim_mm + clearance_mm + 40) / 1000)
        self.log(f"[put away] dropped into the {container} at ({box.x*1000:.0f}, {box.y*1000:.0f}) mm")
        return box

    def place_pixel(self, u: float, v: float) -> None:
        x, y, _ = pixel_to_base(u, v, self.calib)
        self.place_xy(x, y)

    def put_back(self) -> None:
        """Put the held object back where it was picked from."""
        if self.holding is None:
            raise SkillError("I'm not holding anything.")
        self.place_xy(self.holding.x, self.holding.y)
