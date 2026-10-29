"""Record every fetch as a LeRobot dataset episode: the robot's own training data.

Each episode holds, at a fixed rate:
  observation.state           5 arm joints + gripper, as read from the motors
  action                      5 arm joints + gripper, as commanded
  observation.images.overhead overhead camera
  observation.images.wrist    wrist camera
  task                        e.g. "pick up the blue block and hand it over"

Feature names match LeRobot's own SO-101 recordings, so `lerobot-train` (ACT, SmolVLA, ...) can
use the dataset directly. Each episode also gets a label in labels.jsonl (Muse Spark's verdict on
whether the handover worked), so training can keep only the successes.

The zero-shot system (Muse Spark + SAM 3.1) does the task; every run becomes a demonstration.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from .kinematics import ARM_JOINTS
from .paths import ROOT

DATA_DIR = ROOT / "data"
DEFAULT_REPO_ID = "adoodevv/say-and-grasp-handovers"
FPS = 15
IMAGE_SHAPE = (480, 640, 3)
MOTOR_NAMES = [f"{j}.pos" for j in ARM_JOINTS] + ["gripper.pos"]


def dataset_root(repo_id: str = DEFAULT_REPO_ID) -> Path:
    return DATA_DIR / repo_id.split("/")[-1]


def features() -> dict:
    vec = {"dtype": "float32", "shape": (len(MOTOR_NAMES),), "names": MOTOR_NAMES}
    img = {"dtype": "video", "shape": IMAGE_SHAPE, "names": ["height", "width", "channels"]}
    return {
        "observation.state": dict(vec),
        "action": dict(vec),
        "observation.images.overhead": dict(img),
        "observation.images.wrist": dict(img),
    }


def _rgb(frame: np.ndarray | None) -> np.ndarray:
    """BGR camera frame -> RGB at the dataset's image size (black if the camera gave nothing)."""
    h, w, _ = IMAGE_SHAPE
    if frame is None:
        return np.zeros(IMAGE_SHAPE, dtype=np.uint8)
    if frame.shape[:2] != (h, w):
        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


class EpisodeRecorder:
    """Background recorder. start(task) ... stop(success, info) around each fetch."""

    def __init__(self, arm, overhead_frame, wrist_frame, repo_id: str = DEFAULT_REPO_ID, fps: int = FPS,
                 log=print):
        """
        arm            say_and_grasp.arm.Arm (only its cached state is read; never the bus)
        overhead_frame / wrist_frame
                       callables returning the latest BGR frame (or None) without blocking
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.arm, self.overhead_frame, self.wrist_frame = arm, overhead_frame, wrist_frame
        self.fps, self.log = fps, log
        self.root = dataset_root(repo_id)
        self.labels_path = self.root / "labels.jsonl"
        if (self.root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=self.root, image_writer_threads=4)
        else:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self.ds = LeRobotDataset.create(repo_id, fps=fps, features=features(), root=self.root,
                                            robot_type="so101_follower", use_videos=True,
                                            image_writer_threads=4)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._save_lock = threading.Lock()  # an episode's video encoding finishes before the next starts
        self.task = ""
        self.frames = 0
        self.t_start = 0.0

    # ------------------------------------------------------------------ recording loop
    def _loop(self) -> None:
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            state, action = self.arm.snapshot_state()
            if state is not None and action is not None:
                self.ds.add_frame({
                    "observation.state": state.astype(np.float32),
                    "action": action.astype(np.float32),
                    "observation.images.overhead": _rgb(self.overhead_frame()),
                    "observation.images.wrist": _rgb(self.wrist_frame()),
                    "task": self.task,
                })
                self.frames += 1
            nxt += period
            delay = nxt - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.perf_counter()  # fell behind: don't try to catch up with a burst

    def start(self, task: str) -> None:
        self._save_lock.acquire()  # wait if the previous episode is still encoding
        self._save_lock.release()
        self.task, self.frames, self.t_start = task, 0, time.time()
        self.arm.track_state = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        self.log(f"[record] episode started: {task!r}")

    def stop(self, success: bool, info: dict | None = None, background: bool = True) -> None:
        """End the episode, save it (video encoding in the background by default) and label it."""
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        self.arm.track_state = False
        duration = time.time() - self.t_start
        if self.frames < 5:
            self.ds.clear_episode_buffer()
            self.log("[record] episode too short; discarded")
            return
        label = {"task": self.task, "success": bool(success), "frames": self.frames,
                 "duration_s": round(duration, 1), "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                 **(info or {})}

        def save():
            with self._save_lock:
                index = self.ds.meta.total_episodes
                self.ds.save_episode()
                self.labels_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.labels_path, "a") as f:
                    f.write(json.dumps({"episode_index": index, **label}) + "\n")
                self.log(f"[record] saved episode {index} ({self.frames} frames, "
                         f"{'success' if success else 'failure'})")

        if background:
            threading.Thread(target=save, daemon=True).start()
        else:
            save()

    def close(self) -> None:
        if self._thread is not None:
            self.stop(success=False, info={"note": "interrupted"}, background=False)
        with self._save_lock:
            self.ds.finalize()


def read_labels(repo_id: str = DEFAULT_REPO_ID) -> list[dict]:
    p = dataset_root(repo_id) / "labels.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
