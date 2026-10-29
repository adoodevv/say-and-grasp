"""Grab single frames from a USB camera."""
from __future__ import annotations

import time

import cv2
import numpy as np


def capture(camera: str | int, mjpg: bool = False, width: int = 640, height: int = 480) -> np.ndarray:
    """Open the camera, grab one fresh frame, close it again.

    Opening per shot avoids the stalls ("select() timeout") a camera can hit when it sits
    open but unread. MJPG is opt-in: only needed when two cameras share USB bandwidth.
    """
    src = int(camera) if str(camera).isdigit() else camera
    for _ in range(2):
        cap = cv2.VideoCapture(src)
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        frame = None
        if cap.isOpened():
            for _ in range(8):  # let auto-exposure settle, drop stale frames
                ok, f = cap.read()
                if ok:
                    frame = f
        cap.release()
        if frame is not None:
            return frame
        time.sleep(1.0)
    raise RuntimeError(f"Could not read a frame from camera {camera}. Check: v4l2-ctl --list-devices")


class CameraStream:
    """Keeps a camera open and reads frames in the background, so a fresh frame is instant.

    Used for the wrist camera: every alignment check needs a new photo, and re-opening the camera
    each time costs ~1 s (open + auto-exposure). MJPG keeps the USB load low enough to share the
    bus with the overhead camera.
    """

    def __init__(self, camera: str | int, mjpg: bool = True, width: int = 640, height: int = 480):
        import threading

        src = int(camera) if str(camera).isdigit() else camera
        self.cap = cv2.VideoCapture(src)
        if mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera}")
        self._frame: np.ndarray | None = None
        self._t = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self.latest(timeout=3.0)  # make sure it actually delivers frames
        time.sleep(0.5)  # let auto-exposure settle once

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, f = self.cap.read()
            if ok:
                with self._lock:
                    self._frame, self._t = f, time.time()
            else:
                time.sleep(0.01)

    def latest(self, after: float | None = None, timeout: float = 2.0) -> np.ndarray:
        """Newest frame, waiting for one captured after `after` (e.g. once the arm has stopped)."""
        after = time.time() if after is None else after
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._frame is not None and self._t > after:
                    return self._frame.copy()
            time.sleep(0.005)
        raise RuntimeError("Camera stream stopped delivering frames")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.cap.release()
