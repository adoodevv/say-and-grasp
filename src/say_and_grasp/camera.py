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
