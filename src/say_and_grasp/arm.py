"""SO-101 arm control through LeRobot: smooth rate-limited joint moves, gripper, home pose."""
from __future__ import annotations

import json
import time

import numpy as np

from .kinematics import ARM_JOINTS
from .paths import HOME_FILE

DEFAULT_HOME = [0.0, -100.0, 95.0, 70.0, 0.0]  # folded, roughly LeRobot's rest pose


class Arm:
    """Thin wrapper over LeRobot's SO101Follower with smooth, rate-limited joint moves."""

    def __init__(self, port: str, robot_id: str = "my_follower", speed_dps: float = 45.0):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        # max_relative_target: per-command safety clamp (degrees) inside LeRobot.
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, max_relative_target=10.0))
        self.speed = speed_dps
        self.gripper = 50.0

    def connect(self) -> None:
        self.robot.connect(calibrate=False)
        obs = self.robot.get_observation()
        self.gripper = float(obs["gripper.pos"])

    def disconnect(self) -> None:
        self.robot.disconnect()

    def joints(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([obs[f"{j}.pos"] for j in ARM_JOINTS], dtype=float)

    def _send(self, q: np.ndarray, gripper: float) -> None:
        action = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q)}
        action["gripper.pos"] = float(gripper)
        self.robot.send_action(action)

    def move(self, target: np.ndarray, gripper: float | None = None, hz: float = 30.0) -> None:
        """Smooth (ease-in/out) joint-space move at roughly self.speed deg/s."""
        start = self.joints()
        g0 = self.gripper
        g1 = self.gripper if gripper is None else gripper
        span = max(np.max(np.abs(target - start)), abs(g1 - g0) * 0.5, 1.0)
        steps = max(int(span / self.speed * hz), 5)
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)  # smoothstep
            self._send(start + (target - start) * s, g0 + (g1 - g0) * s)
            time.sleep(1.0 / hz)
        # settle on the final target
        for _ in range(int(0.4 * hz)):
            self._send(target, g1)
            time.sleep(1.0 / hz)
        self.gripper = g1

    def set_gripper(self, value: float, seconds: float = 0.8) -> None:
        self.move(self.joints(), gripper=value)
        time.sleep(seconds * 0.3)


def load_home() -> np.ndarray:
    if HOME_FILE.exists():
        return np.array(json.loads(HOME_FILE.read_text())["joints"], dtype=float)
    return np.array(DEFAULT_HOME, dtype=float)


def save_home(joints: np.ndarray) -> None:
    HOME_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOME_FILE.write_text(json.dumps({"joints": [round(float(v), 2) for v in joints]}, indent=2))
