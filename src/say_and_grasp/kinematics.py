"""SO-101 kinematics: straight-down (top-down) grasps via placo on the SO-101 URDF.

Same solver LeRobot's RobotKinematics wraps, with joint limits on, the gripper masked out,
and an analytic seed for pan and wrist roll so top-down poses converge.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np

from .paths import TOOL_FILE, URDF_FILE

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


def load_tcp_offset() -> np.ndarray:
    """Grasp centre relative to the URDF tip frame, in that frame (metres). Zero if not measured."""
    if TOOL_FILE.exists():
        return np.array(json.loads(TOOL_FILE.read_text())["tcp_offset"], dtype=float)
    return np.zeros(3)


def solve_tcp_offset(poses: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the grasp centre from several tip-frame poses that all touched the SAME point.

    Each pose i satisfies p_i + R_i d = c, with d the unknown offset (tool frame, x/y only:
    the vertical part can't be seen from wrist twists and the table height absorbs it) and
    c the touched point. Returns (d, c, per-pose residual in metres).
    """
    A, b = [], []
    for T in poses:
        R, p = T[:3, :3], T[:3, 3]
        A.append(np.c_[R[:, :2], -np.eye(3)])
        b.append(-p)
    A, b = np.vstack(A), np.concatenate(b)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    d = np.array([sol[0], sol[1], 0.0])
    c = sol[2:]
    res = np.array([np.linalg.norm(T[:3, 3] + T[:3, :3] @ d - c) for T in poses])
    return d, c, res


class Kinematics:
    def __init__(self, urdf_path: str = str(URDF_FILE), tcp_offset: np.ndarray | None = None):
        import placo

        # Where the gripper actually grasps, relative to the URDF tip frame (see sag-calibrate --tcp).
        self.tcp = load_tcp_offset() if tcp_offset is None else np.asarray(tcp_offset, dtype=float)

        self.robot = placo.RobotWrapper(urdf_path)
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        self.solver.enable_joint_limits(True)
        self.solver.mask_dof("gripper")
        self.task = self.solver.add_frame_task(TIP_FRAME, np.eye(4))
        self._set([0.0] * len(ARM_JOINTS))
        self.pan_axis = self.robot.get_T_world_frame("shoulder_link")[:3, 3].copy()  # the arm turns about this
        self.limits = np.rad2deg([self.robot.get_joint_limits(n) for n in ARM_JOINTS])  # (lo, hi) per joint

    def at_limit(self, q: np.ndarray, margin: float = 2.0) -> bool:
        """A joint within `margin` degrees of its end stop: the pose is only just reachable, and the
        arm can't move any further that way to correct (so the wrist camera's corrections stall)."""
        q = np.asarray(q, dtype=float)
        return bool(np.any(q <= self.limits[:, 0] + margin) or np.any(q >= self.limits[:, 1] - margin))

    def _set(self, joints_deg) -> None:
        for name, v in zip(ARM_JOINTS, np.deg2rad(joints_deg)):
            self.robot.set_joint(name, v)
        self.robot.update_kinematics()

    def fk(self, joints_deg) -> np.ndarray:
        """URDF tip-frame pose."""
        self._set(joints_deg)
        return self.robot.get_T_world_frame(TIP_FRAME).copy()

    def fk_tcp(self, joints_deg) -> np.ndarray:
        """Pose of the grasp centre (tip frame shifted by the measured TCP offset)."""
        T = self.fk(joints_deg)
        T[:3, 3] = T[:3, 3] + T[:3, :3] @ self.tcp
        return T

    def _solve_T(self, T: np.ndarray, seed: list[float], ow: float = 0.2,
                 iters: int = 150) -> tuple[np.ndarray, float, float, float]:
        """IK for a tip-frame pose, position first. A 5-joint arm can't hit every orientation
        exactly, so orientation is soft. Returns (joints, position error m, tool-axis error deg,
        jaw-direction error deg)."""
        self._set(seed)
        self.task.T_world_frame = T
        self.task.configure(TIP_FRAME, "soft", 1.0, ow)
        for _ in range(iters):
            self.solver.solve(True)
            self.robot.update_kinematics()
        Tr = self.robot.get_T_world_frame(TIP_FRAME)
        target = T[:3, 3] + T[:3, :3] @ self.tcp
        pe = float(np.linalg.norm((Tr[:3, 3] + Tr[:3, :3] @ self.tcp) - target))
        z_err = math.degrees(math.acos(float(np.clip(Tr[:3, 2] @ T[:3, 2], -1, 1))))
        jx, tx = Tr[:2, 0], T[:2, 0]
        jaw_err = abs(_wrap90(math.degrees(math.atan2(jx[1], jx[0]) - math.atan2(tx[1], tx[0]))))
        return np.rad2deg([self.robot.get_joint(n) for n in ARM_JOINTS]), pe, z_err, jaw_err

    def grasp(self, x: float, y: float, z: float, close_dir_deg: float, lean: float | None = None) -> IKResult:
        """Grasp centre at (x, y, z), jaws closing along close_dir_deg, anywhere the arm can reach.

        Straight down when it can (most accurate, what the wrist camera was calibrated for). Close
        to the base or far out it leans the gripper about the jaws' own axis, so the jaws stay level
        with the table: leaning in reaches ~6 cm from the base, leaning out ~40 cm. How far a lean
        helps depends on the tool's direction (leaning sideways to the arm doesn't add reach).
        `tilt_deg` in the result is the lean."""
        r = self.topdown(x, y, z, close_dir_deg)
        if not lean and not r.ok and r.pos_err < 0.006 and r.tilt_deg < 5 and not self.at_limit(r.joints, 0.3):
            # Straight down only just within reach (a joint near its stop, a few mm short): on the
            # real arm this grasps much better than leaning, so take it.
            r = IKResult(r.joints, r.pos_err, r.tilt_deg, True)
        if r.ok and not lean:
            return r
        leans = (15, 30, 45, 60) if lean is None else (lean,)
        # Lean in or out relative to the pan axis (a few cm in front of the base origin), not the origin.
        dx, dy = x - self.pan_axis[0], y - self.pan_axis[1]
        reach = math.hypot(dx, dy)
        az = math.atan2(dy, dx)
        radial = np.array([math.cos(az), math.sin(az), 0.0])
        pan = -math.degrees(math.atan2(y, x))
        best = r
        for flip in (0.0, 180.0):  # the parallel jaws are symmetric: either way round
            th = math.radians(close_dir_deg + flip)
            X = np.array([math.cos(th), math.sin(th), 0.0])
            side = np.cross(np.array([0.0, 0.0, 1.0]), X)  # horizontal, perpendicular to the jaws
            outward = 1.0 if reach > 0.12 else -1.0
            if float(side @ radial) * outward < 0:
                side = -side
            if abs(float(side @ radial)) < 0.35:  # jaws pointing along the arm: lean straight in/out instead
                side = radial * outward
            for lean in leans:
                a = math.radians(lean)
                lean_dir, Xl = side, X
                seed = [pan, 0.0, 0.0, 60.0, 0.0]
                for _ in range(4):
                    # The arm leans within its own vertical plane, which doesn't pass through the base
                    # axis exactly: re-aim the lean along the plane the last solve ended up in.
                    Z = np.array([0.0, 0.0, -math.cos(a)]) + lean_dir * math.sin(a)
                    Xl = Xl - (Xl @ Z) * Z
                    Xl = Xl / np.linalg.norm(Xl)
                    T = np.eye(4)
                    T[:3, :3] = np.c_[Xl, np.cross(Z, Xl), Z]
                    T[:3, 3] = np.array([x, y, z]) - T[:3, :3] @ self.tcp
                    q, pe, z_err, jaw_err = self._solve_T(T, seed)
                    jaw_err = abs(_wrap90(math.degrees(math.atan2(Xl[1], Xl[0])) - (close_dir_deg + flip)))
                    jaw_tilt = math.degrees(math.asin(min(1.0, abs(float(Xl[2])))))  # one jaw lower than the other
                    if pe < 0.003 and z_err < 10 and jaw_err < 15 and jaw_tilt < 20 and not self.at_limit(q):
                        return IKResult(q, pe, float(lean), True)
                    if pe < best.pos_err:
                        best = IKResult(q, pe, float(lean), False)
                    zr = self.robot.get_T_world_frame(TIP_FRAME)[:3, 2]
                    h = np.array([zr[0], zr[1], 0.0])
                    if np.linalg.norm(h) < 1e-3:
                        break
                    lean_dir = h / np.linalg.norm(h)
                    Xl = X.copy()  # keep the jaws' heading; the wrist roll can turn them about the leaning axis
                    seed = list(q)
        return best

    def topdown(self, x: float, y: float, z: float, close_dir_deg: float, iters: int = 150) -> IKResult:
        """Grasp centre at (x, y, z), pointing down, jaws closing along close_dir_deg (base frame, mod 180)."""
        pan = -math.degrees(math.atan2(y, x))
        roll = _wrap90(close_dir_deg - _ROLL_ZERO_DEG + pan)
        th = math.radians(_ROLL_ZERO_DEG - pan + roll)
        X = np.array([math.cos(th), math.sin(th), 0.0])
        Z = np.array([0.0, 0.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = np.c_[X, np.cross(Z, X), Z]
        T[:3, 3] = np.array([x, y, z]) - T[:3, :3] @ self.tcp  # put the TIP FRAME where the grasp centre needs it

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
            pe = float(np.linalg.norm((Tr[:3, 3] + Tr[:3, :3] @ self.tcp) - np.array([x, y, z])))
            tilt = math.degrees(math.acos(float(np.clip(-Tr[2, 2], -1, 1))))
            q = np.rad2deg([self.robot.get_joint(n) for n in ARM_JOINTS])
            res = IKResult(q, pe, tilt, pe < 0.003 and tilt < 25 and not self.at_limit(q))
            if res.ok:
                return res
            if best is None or pe < best.pos_err:
                best = res
        return best
