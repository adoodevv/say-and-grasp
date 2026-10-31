"""Brain view: a split-screen video of what Wiggle sees and thinks, for demos.

    +-------------------------------+-------------------------------+
    |  OVERHEAD CAMERA              |  WRIST CAMERA                 |
    |  live, or SAM 3.1's masks for |  live, or SAM 3.1's mask with |
    |  a few seconds after a look   |  the jaw centre and the error |
    +-------------------------------+-------------------------------+
    |  You: "Wiggle, pass me the screwdriver"                       |
    |  Muse Spark -> fetch("screwdriver")                           |
    |  Wiggle: "On it!"                     status / verdict ...    |
    +---------------------------------------------------------------+

It listens to say_and_grasp.events, so the robot code doesn't know it exists. Frames are written
at a steady rate to outputs/brainview/<date-time>.mp4 while the assistant runs.
"""
from __future__ import annotations

import textwrap
import threading
import time
from collections import deque

import cv2
import numpy as np

from . import events
from .paths import OUTPUT_DIR

W, H = 1280, 720
PANEL_W, PANEL_H = 640, 480
HOLD_S = 3.0  # how long a SAM result stays on screen before going back to live video

BG = (24, 24, 28)
FG = (235, 235, 235)
DIM = (150, 150, 160)
BLUE = (255, 160, 70)    # Muse Spark (BGR)
GREEN = (120, 220, 120)  # Wiggle / success
ORANGE = (60, 170, 255)  # you
RED = (90, 90, 240)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _fit(img: np.ndarray | None, w: int = PANEL_W, h: int = PANEL_H) -> np.ndarray:
    out = np.full((h, w, 3), 40, np.uint8)
    if img is None:
        cv2.putText(out, "no signal", (w // 2 - 70, h // 2), FONT, 0.8, DIM, 2, cv2.LINE_AA)
        return out
    ih, iw = img.shape[:2]
    s = min(w / iw, h / ih)
    r = cv2.resize(img, (int(iw * s), int(ih * s)), interpolation=cv2.INTER_AREA)
    y, x = (h - r.shape[0]) // 2, (w - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def _tag(img: np.ndarray, text: str, color=FG, y: int = 30) -> None:
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.65, 2)
    cv2.rectangle(img, (8, y - th - 10), (8 + tw + 16, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (16, y), FONT, 0.65, color, 2, cv2.LINE_AA)


class BrainView:
    def __init__(self, overhead_frame, wrist_frame, fps: float = 10.0, path=None, log=print):
        """overhead_frame / wrist_frame: callables returning the latest BGR frame (or None)."""
        self.overhead_frame, self.wrist_frame = overhead_frame, wrist_frame
        self.fps = fps
        d = OUTPUT_DIR / "brainview"
        d.mkdir(parents=True, exist_ok=True)
        self.path = path or d / time.strftime("%Y%m%d-%H%M%S.mp4")
        self.writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        self.lines: deque = deque(maxlen=6)  # (who, text, colour)
        self.status = ""
        self.verdict: tuple[str, tuple] | None = None
        self.over: tuple[np.ndarray, str, float] | None = None
        self.wrist: tuple[np.ndarray, str, float] | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self.log = log

    # ------------------------------------------------------------------ events
    def _on(self, kind: str, d: dict) -> None:
        now = time.time()
        with self._lock:
            if kind == "heard":
                self.lines.append(("You", f'"{d["text"]}"', ORANGE))
                self.verdict = None
            elif kind == "tool":
                args = ", ".join(f'{v!r}' if isinstance(v, str) else str(v) for v in d.get("args", {}).values())
                self.lines.append(("Muse Spark", f"-> {d['name']}({args})", BLUE))
            elif kind == "say":
                self.lines.append(("Wiggle", f'"{d["text"]}"', GREEN))
            elif kind == "overhead_sam":
                self.over = (d["image"].copy(), d.get("label", "SAM 3.1"), now)
            elif kind == "wrist_sam":
                self.wrist = (d["image"].copy(), d.get("label", "SAM 3.1"), now)
            elif kind == "verify":
                ok = d.get("ok")
                by = d.get("by", "Muse Spark check")
                self.verdict = (f"{by}: {'success' if ok else 'not sure'} - {d.get('reason', '')}",
                                GREEN if ok else RED)
            elif kind == "status":
                self.status = d.get("text", "")

    # ------------------------------------------------------------------ drawing
    def frame(self) -> np.ndarray:
        now = time.time()
        canvas = np.full((H, W, 3), BG, np.uint8)
        with self._lock:
            over, wrist = self.over, self.wrist
            lines, status, verdict = list(self.lines), self.status, self.verdict
        if over and now - over[2] < HOLD_S:
            left, ltag, lcol = over[0], over[1], GREEN
        else:
            left, ltag, lcol = self.overhead_frame(), "Overhead camera (live)", FG
        if wrist and now - wrist[2] < HOLD_S:
            right, rtag, rcol = wrist[0], wrist[1], GREEN
        else:
            right, rtag, rcol = self.wrist_frame(), "Wrist camera (live)", FG
        lp, rp = _fit(left), _fit(right)
        _tag(lp, ltag, lcol)
        _tag(rp, rtag, rcol)
        canvas[:PANEL_H, :PANEL_W] = lp
        canvas[:PANEL_H, PANEL_W:] = rp
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, PANEL_H), BG, 4)

        # conversation strip
        y = PANEL_H + 34
        for who, text, col in lines[-5:]:
            cv2.putText(canvas, f"{who}:", (20, y), FONT, 0.6, col, 2, cv2.LINE_AA)
            for k, part in enumerate(textwrap.wrap(text, 62)[:1]):
                cv2.putText(canvas, part, (150, y), FONT, 0.6, FG, 1, cv2.LINE_AA)
            y += 36
        # right column: what it's doing / the verdict
        x0 = 860
        cv2.putText(canvas, "SAY-AND-GRASP", (x0, PANEL_H + 34), FONT, 0.6, DIM, 2, cv2.LINE_AA)
        if status:
            for k, part in enumerate(textwrap.wrap(status, 30)[:2]):
                cv2.putText(canvas, part, (x0, PANEL_H + 74 + 30 * k), FONT, 0.6, FG, 1, cv2.LINE_AA)
        if verdict:
            for k, part in enumerate(textwrap.wrap(verdict[0], 30)[:3]):
                cv2.putText(canvas, part, (x0, PANEL_H + 150 + 28 * k), FONT, 0.55, verdict[1], 1, cv2.LINE_AA)
        cv2.putText(canvas, "Muse Voice Transcribe  |  Muse Spark  |  SAM 3.1", (x0, H - 16), FONT, 0.45, DIM,
                    1, cv2.LINE_AA)
        return canvas

    def _loop(self) -> None:
        period = 1.0 / self.fps
        nxt = time.perf_counter()
        while not self._stop.is_set():
            try:
                self.writer.write(self.frame())
            except Exception as e:
                self.log(f"[brainview] frame error: {e}")
            nxt += period
            time.sleep(max(0.0, nxt - time.perf_counter()))

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> "BrainView":
        events.subscribe(self._on)
        self._thread.start()
        self.log(f"[brainview] recording to {self.path}")
        return self

    def close(self) -> None:
        events.unsubscribe(self._on)
        self._stop.set()
        self._thread.join(timeout=2)
        self.writer.release()
        self.log(f"[brainview] saved {self.path}")
