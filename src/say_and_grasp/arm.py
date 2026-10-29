"""SO-101 arm control through LeRobot: smooth rate-limited joint moves, gripper, home pose."""
from __future__ import annotations

import json
import time

import numpy as np

from .kinematics import ARM_JOINTS
from .paths import CLEAR_FILE, HOME_FILE

DEFAULT_HOME = [0.0, -100.0, 95.0, 70.0, 0.0]  # folded, roughly LeRobot's rest pose


class Arm:
    """Thin wrapper over LeRobot's SO101Follower with smooth, rate-limited joint moves."""

    def __init__(self, port: str, robot_id: str = "my_follower", speed_dps: float = 45.0):
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        import logging

        # max_relative_target: per-command safety clamp (degrees) inside LeRobot. Our moves are
        # already smooth; this only catches a runaway command. The clamp warning fires every time
        # a joint lags (e.g. the gripper squeezing an object), so keep it out of the console.
        logging.getLogger("lerobot.robots.utils").setLevel(logging.ERROR)
        self.robot = SO101Follower(SO101FollowerConfig(port=port, id=robot_id, max_relative_target=25.0))
        self.speed = speed_dps
        self.gripper = 50.0
        self.q_cmd: np.ndarray | None = None  # last COMMANDED arm joints (not the sagged reading)
        # Latest reading / command, for the episode recorder. The recorder must never talk to the
        # serial bus itself (two readers garble each other's packets), so the arm caches them.
        import threading

        self._state_lock = threading.Lock()
        self.last_state: np.ndarray | None = None  # 5 arm joints + gripper, as read
        self.last_action: np.ndarray | None = None  # 5 arm joints + gripper, as commanded
        self.track_state = False  # read the bus every control tick (only while recording)
        self.corr = np.zeros(len(ARM_JOINTS))  # learned gravity-sag compensation, degrees

    def connect(self, attempts: int = 3) -> None:
        # The Feetech bus sometimes returns a garbled status packet right after power-up or a
        # replug ("Incorrect status packet"); a clean retry nearly always works.
        for i in range(attempts):
            try:
                self.robot.connect(calibrate=False)
                break
            except ConnectionError as e:
                if i == attempts - 1:
                    raise ConnectionError(
                        f"{e}\nArm bus keeps failing. Check: arm power supply on; arm USB adapter plugged "
                        "straight into the laptop (not a shared hub); motor cables seated; port "
                        "(lerobot-find-port)."
                    ) from None
                print(f"[arm] bus glitch on connect ({e.__class__.__name__}); retrying {i + 2}/{attempts}...")
                try:
                    self.robot.bus.disconnect(disable_torque=False)
                except Exception:
                    pass
                time.sleep(1.0)
        s = self.read()
        self.gripper = float(s[5])
        self.q_cmd = s[:5]
        with self._state_lock:
            self.last_action = s.copy()

    def disconnect(self) -> None:
        self.robot.disconnect()

    def read(self) -> np.ndarray:
        """Read all 6 motors (5 arm joints + gripper) and cache them for the recorder."""
        obs = self.robot.get_observation()
        s = np.array([obs[f"{j}.pos"] for j in ARM_JOINTS] + [obs["gripper.pos"]], dtype=float)
        with self._state_lock:
            self.last_state = s
        return s

    def joints(self) -> np.ndarray:
        return self.read()[:5]

    def gripper_pos(self) -> float:
        return float(self.read()[5])

    def snapshot_state(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        with self._state_lock:
            return (None if self.last_state is None else self.last_state.copy(),
                    None if self.last_action is None else self.last_action.copy())

    def _send(self, q: np.ndarray, gripper: float) -> None:
        action = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q)}
        action["gripper.pos"] = float(gripper)
        self.robot.send_action(action)
        with self._state_lock:
            self.last_action = np.array(list(q) + [gripper], dtype=float)
        if self.track_state:
            self.read()

    def move(self, target: np.ndarray, gripper: float | None = None, hz: float = 30.0,
             settle_s: float = 0.6, max_corr: float = 6.0) -> None:
        """Smooth (ease-in/out) joint-space move at roughly self.speed deg/s, then settle.

        Interpolates from the last COMMANDED pose (re-reading the sagged position and commanding
        it would make the arm creep down every time). While settling, it nudges the command by
        the measured error so gravity sag doesn't leave the gripper short/low; the nudge is
        capped at max_corr degrees per joint.
        """
        target = np.asarray(target, dtype=float)
        start = self.q_cmd if self.q_cmd is not None else self.joints()
        g0 = self.gripper
        g1 = self.gripper if gripper is None else gripper
        span = max(np.max(np.abs(target - start)), abs(g1 - g0) * 0.5, 1.0)
        steps = max(int(span / self.speed * hz), 5)
        corr0 = self.corr.copy()
        for k in range(1, steps + 1):
            s = k / steps
            s = s * s * (3 - 2 * s)  # smoothstep
            self._send(start + (target - start) * s + corr0, g0 + (g1 - g0) * s)
            time.sleep(1.0 / hz)
        # settle: integrate the remaining error into the sag compensation
        corr = corr0
        for _ in range(int(settle_s * hz)):
            err = target - self.joints()
            corr = np.clip(corr + 0.3 * err, -max_corr, max_corr)
            self._send(target + corr, g1)
            time.sleep(1.0 / hz)
        self.corr = corr
        self.q_cmd = target
        self.gripper = g1

    def leave_home(self, gripper: float | None = None) -> None:
        """If we're at home, first lift the wrist to the clear pose so nothing scrapes the base."""
        clear = load_clear()
        if clear is not None and np.max(np.abs(self.joints() - load_home())) < 15:
            self.move(clear, gripper=gripper, settle_s=0.1)  # pass-through point

    def go_clear(self, gripper: float | None = None) -> None:
        """Step back to the clear pose (near home, wrist up): out of the tool area, quicker than
        folding all the way home. Falls back to home if no clear pose was saved."""
        clear = load_clear()
        self.move(clear if clear is not None else load_home(), gripper=gripper, settle_s=0.3)

    def go_home(self, gripper: float | None = None) -> None:
        """Return home via the clear pose (wrist lifted), then fold down."""
        clear, home = load_clear(), load_home()
        if clear is not None and np.max(np.abs(self.joints() - home)) >= 15:  # skip if already home
            self.move(clear, gripper=gripper, settle_s=0.1)
        self.move(home, gripper=gripper, settle_s=0.3)

    def set_gripper(self, value: float) -> None:
        """Open/close in place: holds the last commanded arm pose, so the arm doesn't sink."""
        self.move(self.q_cmd if self.q_cmd is not None else self.joints(), gripper=value, settle_s=0.35)


def load_home() -> np.ndarray:
    if HOME_FILE.exists():
        return np.array(json.loads(HOME_FILE.read_text())["joints"], dtype=float)
    return np.array(DEFAULT_HOME, dtype=float)


def save_home(joints: np.ndarray) -> None:
    HOME_FILE.parent.mkdir(parents=True, exist_ok=True)
    HOME_FILE.write_text(json.dumps({"joints": [round(float(v), 2) for v in joints]}, indent=2))


def load_clear() -> np.ndarray | None:
    if CLEAR_FILE.exists():
        return np.array(json.loads(CLEAR_FILE.read_text())["joints"], dtype=float)
    return None


def save_clear(joints: np.ndarray) -> None:
    CLEAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CLEAR_FILE.write_text(json.dumps({"joints": [round(float(v), 2) for v in joints]}, indent=2))
