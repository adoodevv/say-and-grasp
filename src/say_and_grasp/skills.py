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

import math
import time
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
from .camera import capture
from .kinematics import Kinematics
from .paths import CALIB_FILE, URDF_FILE, output
from .sam import Detection, GraspTarget, draw_overlay, grasp_from_mask, segment_robust


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
        speed: float = 30.0,
        grasp_height_mm: float = 5.0,
        approach_mm: float = 50.0,
        gripper_open: float = 60.0,
        gripper_closed: float = 5.0,
        empty_margin: float = 3.0,
        yaw_offset: float = 0.0,
        dry_run: bool = False,
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

    # ------------------------------------------------------------------ lifecycle
    def connect(self) -> None:
        if self.arm:
            self.arm.connect()
            self.arm.go_home()

    def close(self) -> None:
        if self.arm:
            self.arm.disconnect()

    def go_home(self) -> None:
        if self.arm:
            self.arm.go_home(gripper=self.g_closed if not self.holding else None)

    # ------------------------------------------------------------------ perception
    def snapshot(self) -> np.ndarray:
        """Overhead picture. The arm should be home (out of view) when this is called."""
        return capture(self.camera)

    def look(self, prompt: str, frame: np.ndarray | None = None) -> Seen:
        """Find `prompt` in the overhead view; pick the most confident/largest match."""
        frame = self.snapshot() if frame is None else frame
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

    # ------------------------------------------------------------------ actions
    def pick(self, prompt: str, retries: int = 2, pause: bool = False) -> Seen:
        """Find and grasp `prompt`. Verifies the grasp and retries with a fresh look."""
        last_err = None
        for attempt in range(retries + 1):
            if attempt:
                self.log(f"[pick] attempt {attempt + 1}/{retries + 1}")
            seen = self.look(prompt)
            above = self._solve("above object", seen.x, seen.y, seen.z + self.hov, seen.close_dir)
            grasp = self._solve("grasp", seen.x, seen.y, seen.z + self.gh, seen.close_dir)
            if self.dry_run:
                self.log("[pick] dry run: plan OK, not moving")
                return seen
            arm = self.arm
            arm.leave_home()  # no-op when we're already at the clear pose after a miss
            arm.move(above, gripper=self.g_open)
            if pause and input("[pick] hovering; Enter to grasp, q to abort: ").strip().lower() == "q":
                arm.go_home()
                raise SkillError("aborted")
            arm.move(grasp)
            arm.set_gripper(self.g_closed)
            time.sleep(0.3)
            width = float(arm.robot.get_observation()["gripper.pos"])
            arm.move(above)  # lift
            if width > self.g_closed + self.empty_margin:
                self.log(f"[pick] holding {prompt} (gripper at {width:.0f})")
                self.holding = seen
                return seen
            last_err = f"missed {prompt} (gripper closed fully at {width:.0f})"
            self.log(f"[pick] {last_err}; stepping back to look again")
            # The miss usually nudges the object, so look again rather than guessing a new spot.
            # Step back only as far as the clear pose (out of the tool area), not all the way home.
            arm.set_gripper(self.g_open)
            arm.go_clear()
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
        self.arm.move(above)
        self.arm.move(down)
        self.arm.set_gripper(self.g_open)
        self.arm.move(above)
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
