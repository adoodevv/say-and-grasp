"""SO-101 kinematics: straight-down (top-down) grasps via placo on the SO-101 URDF.

Same solver LeRobot's RobotKinematics wraps, with joint limits on, the gripper masked out,
and an analytic seed for pan and wrist roll so top-down poses converge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .paths import URDF_FILE

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
TIP_FRAME = "gripper_frame_link"

# In the URDF the jaws close along the tool frame's x-axis. With the tool pointing down,
# that axis sits at roughly (177 - pan + roll) degrees in the base frame.
_ROLL_ZERO_DEG = 177.0


def _wrap90(a: float) -> float:
    while a > 90:
        a -= 180
    while a < -90:
        a += 180
    return a


@dataclass
class IKResult:
    joints: np.ndarray  # degrees, ARM_JOINTS order
    pos_err: float  # metres
    tilt_deg: float  # 0 = tool straight down
    ok: bool


class Kinematics:
    def __init__(self, urdf_path: str = str(URDF_FILE)):
        import placo

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.mask_dof("gripper")
        self.task = self.solver.add_frame_task(TIP_FRAME, np.eye(4))

    def _set(self, joints_deg) -> None:
        for name, v in zip(ARM_JOINTS, np.deg2rad(joints_deg)):
            self.robot.set_joint(name, v)
        self.robot.update_kinematics()

    def fk(self, joints_deg) -> np.ndarray:
        self._set(joints_deg)
        return self.robot.get_T_world_frame(TIP_FRAME).copy()

    def topdown(self, x: float, y: float, z: float, close_dir_deg: float, iters: int = 150) -> IKResult:
        """Tip at (x, y, z), pointing down, jaws closing along close_dir_deg (base frame, mod 180)."""
        pan = -math.degrees(math.atan2(y, x))
        roll = _wrap90(close_dir_deg - _ROLL_ZERO_DEG + pan)
        th = math.radians(_ROLL_ZERO_DEG - pan + roll)
        X = np.array([math.cos(th), math.sin(th), 0.0])
        Z = np.array([0.0, 0.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = np.c_[X, np.cross(Z, X), Z]
        T[:3, 3] = [x, y, z]

        best = None
        # Strict orientation first; relax it (allow some tilt) if the pose is out of reach.
        for ow in (1.0, 0.3, 0.1):
            self._set([pan, 0.0, 0.0, 90.0, roll])
            self.task.T_world_frame = T
            self.task.configure(TIP_FRAME, "soft", 1.0, ow)
            for _ in range(iters):
                self.solver.solve(True)
                self.robot.update_kinematics()
            Tr = self.robot.get_T_world_frame(TIP_FRAME)
            pe = float(np.linalg.norm(Tr[:3, 3] - T[:3, 3]))
            tilt = math.degrees(math.acos(float(np.clip(-Tr[2, 2], -1, 1))))
            q = np.rad2deg([self.robot.get_joint(n) for n in ARM_JOINTS])
            res = IKResult(q, pe, tilt, pe < 0.004 and tilt < 25)
            if res.ok:
                return res
            if best is None or pe < best.pos_err:
                best = res
        return best
