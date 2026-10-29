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

from .arm import Arm
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
from .paths import CALIB_FILE, URDF_FILE, WRIST_FILE, output
from .sam import Detection, GraspTarget, draw_overlay, grasp_from_mask, segment_robust


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
SETTLE_WAYPOINT = 0.25  # s: short pause at pass-through points
SETTLE_PRECISE = 0.5  # s: full settle where position matters (grasp, place, photos)


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
        speed: float = 45.0,
        grasp_height_mm: float = 5.0,
        approach_mm: float = 50.0,
        gripper_open: float = 60.0,
        gripper_closed: float = 5.0,
        empty_margin: float = 3.0,
        yaw_offset: float = 0.0,
        dry_run: bool = False,
        stream_overhead: bool = False,
        log=print,
    ):
        self.calib = load_calib(calib_path)
        self.camera = camera or self.calib["camera"]
        self.kin = Kinematics(urdf)
        self.arm = None if dry_run else Arm(port, robot_id, speed_dps=speed)
        self.gh = grasp_height_mm / 1000
        self.hov = approach_mm / 1000
        self.g_open, self.g_closed, self.empty_margin = gripper_open, gripper_closed, empty_margin
        self.yaw_offset = yaw_offset
        self.dry_run = dry_run
        self.log = log
        self.holding: Seen | None = None
        # Wrist camera alignment (sag-calibrate --wrist). Without it, picks use the overhead estimate only.
        self.wrist = json.loads(WRIST_FILE.read_text()) if WRIST_FILE.exists() else None
        self.align_tol = 0.003  # good enough below 3 mm: inside the jaw clearance, and about the servos' slack
        self.align_max = 0.04  # beyond this a correction is "following" a moved object, not fine-tuning
        self.follow_max = 0.12  # the wrist camera seeing it further than this is probably something else
        self.align_tol_deg = 6.0  # jaws within 6 deg of square is fine
        self.wrist_stream: CameraStream | None = None
        self.overhead_stream: CameraStream | None = None
        self._want_overhead_stream = stream_overhead  # needed when recording episodes

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if self.arm:
            self.arm.connect()
            self.arm.go_home()
        if self.wrist and not self.dry_run:
            try:
                self.wrist_stream = CameraStream(self.wrist["camera"])
            except Exception as e:  # fall back to opening it per photo
                self.log(f"[wrist] streaming unavailable ({e}); using single shots")
                self.wrist_stream = None
        if self._want_overhead_stream and not self.dry_run:
            try:
                self.overhead_stream = CameraStream(self.camera)
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
        return capture(self.camera)

    def overhead_peek(self) -> np.ndarray | None:
        return self.overhead_stream.peek() if self.overhead_stream else None

    def wrist_peek(self) -> np.ndarray | None:
        return self.wrist_stream.peek() if self.wrist_stream else None

    def look(self, prompt: str, frame: np.ndarray | None = None) -> Seen:
        """Find `prompt` in the overhead view; pick the most confident/largest match."""
        frame = self.snapshot() if frame is None else frame
        dets, used = segment_robust(frame, prompt)
        if not dets:
            # SAM occasionally misses on one frame (glare, exposure); one fresh photo before giving up.
            time.sleep(0.5)
            frame = self.snapshot()
            dets, used = segment_robust(frame, prompt)
        if not dets:
            raise SkillError(f"I can't see any {prompt} on the bench.")
        dets.sort(key=lambda d: (d.confidence or 0, d.mask.sum()), reverse=True)
        det = dets[0]
        g = grasp_from_mask(det.mask)
        x, y, z = pixel_to_base(g.u, g.v, self.calib)
        close_dir = pixel_dir_to_base(g.u, g.v, g.yaw_deg, self.calib) + self.yaw_offset
        seen = Seen(prompt, det, g, x, y, z, close_dir, frame)
        cv2.imwrite(str(output("look.png")), draw_overlay(shade_reach(frame, self.calib), [det], [g]))
        self.log(f"[look] {prompt}: pixel ({g.u:.0f},{g.v:.0f}) -> ({x*1000:.0f}, {y*1000:.0f}) mm, "
                 f"reach {seen.reach*1000:.0f} mm" + (f" ({len(dets)} matches)" if len(dets) > 1 else ""))
        return seen

    # ------------------------------------------------------------------ planning
    def _solve(self, name: str, x: float, y: float, z: float, close_dir: float) -> np.ndarray:
        reach = math.hypot(x, y)
        if reach < REACH_MIN:
            raise SkillError(f"{name}: {reach*100:.0f} cm from the base is too close for me (min {REACH_MIN*100:.0f} cm).")
        r = self.kin.topdown(x, y, z, close_dir)
        if not r.ok:
            raise SkillError(f"{name}: {reach*100:.0f} cm from the base is out of my reach (max ~{REACH_MAX*100:.0f} cm).")
        return r.joints

    # ------------------------------------------------------------------ wrist camera
    def tool_xy_axes(self, joints: np.ndarray) -> np.ndarray:
        """2x2: columns are the tool frame's x and y axes in the table plane (gripper pointing down)."""
        return self.kin.fk(joints)[:2, :2]

    def wrist_offset(self, prompt: str, joints: np.ndarray, save_as: str = "wrist.png"):
        """From a wrist photo at hover: how far the object is from the jaw centre (base frame, metres),
        and how far the gripper should turn (degrees) so the jaws close across it.
        Returns (offset_xy, turn_deg) or None if the object isn't visible. turn_deg is None when the
        wrist calibration has no angle data or the object is too square to have a direction."""
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
        px_per_m = np.array(self.wrist["px_per_m"], dtype=float)  # 2x2: pixel error per metre of object offset
        offset = self.tool_xy_axes(joints) @ np.linalg.solve(px_per_m, err_px)

        turn = None
        if "target_yaw_px" in self.wrist and _elongated(det.mask):
            # how far the object's grasp direction is rotated from where the jaws close, in the image
            e = _wrap90(g.yaw_deg - self.wrist["target_yaw_px"])
            turn = -e / self.wrist["yaw_gain"]  # gripper turn that cancels it (gain = image deg per gripper deg)
        return offset, turn

    def _align(self, seen: Seen, above: np.ndarray, rounds: int = 4) -> tuple[float, float, np.ndarray, str]:
        """Hovering over the object: measure with the wrist camera, shift/turn, re-measure.

        Returns (x, y, hover_joints, status), status one of:
          "centred"  jaws centred and square: grasp
          "close"    not perfect but within ~6 mm / 12 deg after all rounds: grasp
          "lost"     the wrist camera can't see it (or it's beyond follow range)
          "stuck"    it keeps ending up off-centre: don't grasp blind
        If it sees the object further away (someone moved it), it follows it.
        Small moves are about the size of the servos' slack, so one correction can land short or
        long; re-measuring after every move (and accumulating corrections) converges anyway.
        """
        x, y = seen.x, seen.y
        for i in range(rounds + 1):
            m = self.wrist_offset(seen.prompt, above, save_as=f"wrist_{i + 1}.png")
            if m is None:
                self.log("[align] can't see it in the wrist camera")
                return x, y, above, "lost"
            d, turn = m
            n = float(np.hypot(*d))
            t = abs(turn) if turn is not None else 0.0
            status = f"{n*1000:.0f} mm off" + (f", {turn:+.0f} deg" if turn is not None else "")
            if n < self.align_tol and t < self.align_tol_deg:
                self.log(f"[align] centred and square ({status})")
                return x, y, above, "centred"
            if n > self.follow_max:
                self.log(f"[align] what I see is {n*100:.0f} cm away; probably not it")
                return x, y, above, "lost"
            if i == rounds:
                ok = n < 2 * self.align_tol and t < 2 * self.align_tol_deg
                self.log(f"[align] {status} after {rounds} corrections; " + ("close enough" if ok else "not grabbing blind"))
                return x, y, above, "close" if ok else "stuck"
            nx, ny = x + d[0], y + d[1]
            try:
                if turn is not None and self.align_tol_deg <= t <= 60:
                    cd = seen.close_dir + turn
                else:
                    cd = seen.close_dir
                new_above = self._solve("above object", nx, ny, seen.z + self.hov, cd)
            except SkillError as e:
                self.log(f"[align] it's moved out of reach ({e})")
                return x, y, above, "lost"
            x, y, seen.close_dir, above = nx, ny, cd, new_above
            self.log(f"[align] {status}; " + ("following it" if n > self.align_max else "correcting"))
            self.arm.move(above, settle_s=SETTLE_PRECISE)  # settled = sharp next photo
        return x, y, above, "stuck"

    # ------------------------------------------------------------------ actions
    def pick(self, prompt: str, retries: int = 2, pause: bool = False) -> Seen:
        """Find and grasp `prompt`, as one fluid motion.

        The overhead photo is taken while the arm is home (out of view); the arm starts lifting
        while SAM works on it. At hover the wrist camera centres and squares the jaws, then it
        grasps. If the grasp misses, it rises back to hover and looks again with the wrist camera
        right there (the object has usually only been nudged), re-aligns and tries again. Only if
        the object has left the wrist camera's view does it step back for a fresh overhead look.
        """
        t0 = time.time()
        frame = self.snapshot()
        if self.dry_run:
            seen = self.look(prompt, frame)
            self._solve("above object", seen.x, seen.y, seen.z + self.hov, seen.close_dir)
            self._solve("grasp", seen.x, seen.y, seen.z + self.gh, seen.close_dir)
            self.log("[pick] dry run: plan OK, not moving")
            return seen

        arm = self.arm
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(self.look, prompt, frame)
            arm.leave_home(gripper=self.g_open)  # lift the wrist while SAM is thinking
            seen = fut.result()
        above = self._solve("above object", seen.x, seen.y, seen.z + self.hov, seen.close_dir)
        arm.move(above, gripper=self.g_open, settle_s=SETTLE_PRECISE)

        for attempt in range(retries + 1):
            if attempt:
                self.log(f"[pick] attempt {attempt + 1}/{retries + 1}")
            if self.wrist:
                x, y, above, status = self._align(seen, above)  # may also turn seen.close_dir
                seen.x, seen.y = x, y  # so put_back() returns it to where it really was
                if status in ("lost", "stuck"):
                    # Don't close on nothing: step back and find it again from above.
                    self.log("[pick] lost track of it; stepping back for an overhead look")
                    arm.go_clear(gripper=self.g_open)
                    seen = self.look(prompt)
                    above = self._solve("above object", seen.x, seen.y, seen.z + self.hov, seen.close_dir)
                    arm.move(above, settle_s=SETTLE_PRECISE)
                    x, y, above, status = self._align(seen, above)
                    seen.x, seen.y = x, y
                    if status in ("lost", "stuck"):
                        raise SkillError(f"I can see the {prompt} but I can't line up with it. "
                                         "Could you move it a little closer to the middle of my reach?")
            grasp = self._solve("grasp", seen.x, seen.y, seen.z + self.gh, seen.close_dir)
            if pause and input("[pick] hovering; Enter to grasp, q to abort: ").strip().lower() == "q":
                arm.go_home()
                raise SkillError("aborted")
            arm.move(grasp, settle_s=SETTLE_PRECISE)
            arm.set_gripper(self.g_closed)
            width = arm.gripper_pos()
            if width > self.g_closed + self.empty_margin:
                arm.move(above, settle_s=SETTLE_WAYPOINT)  # lift
                self.log(f"[pick] holding {prompt} (gripper at {width:.0f}) after {time.time() - t0:.1f}s")
                self.holding = seen
                return seen
            self.log(f"[pick] missed (gripper closed fully at {width:.0f}); re-checking from above")
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

    def place_pixel(self, u: float, v: float) -> None:
        x, y, _ = pixel_to_base(u, v, self.calib)
        self.place_xy(x, y)

    def put_back(self) -> None:
        """Put the held object back where it was picked from."""
        if self.holding is None:
            raise SkillError("I'm not holding anything.")
        self.place_xy(self.holding.x, self.holding.y)
