"""Overhead camera <-> arm base frame: a homography from image pixels to the table plane."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .paths import CALIB_FILE

# Annulus the SO-101 can grasp straight down in (metres from the base). Inside ~16 cm the arm
# has to fold so far that a parallel gripper's base hits the arm; beyond ~28 cm the wrist runs
# out of range.
REACH_MIN, REACH_MAX = 0.16, 0.28


def load_calib(path: str | Path = CALIB_FILE) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No calibration at {path}. Run: sag-calibrate --camera N")
    return json.loads(path.read_text())


def save_calib(calib: dict, path: str | Path = CALIB_FILE) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calib, indent=2))


def fit_table_plane(xyz: np.ndarray) -> np.ndarray:
    """Least-squares plane z = a*x + b*y + c through the calibration touches."""
    A = np.c_[xyz[:, 0], xyz[:, 1], np.ones(len(xyz))]
    coef, *_ = np.linalg.lstsq(A, xyz[:, 2], rcond=None)
    return coef


def table_z_at(x: float, y: float, calib: dict) -> float:
    """Table height under (x, y) in the arm frame. The table is rarely level relative to the
    arm model, so use the fitted plane when there is one."""
    if "table_plane" in calib:
        a, b, c = calib["table_plane"]
        return float(a * x + b * y + c)
    return float(calib["table_z"])


def pixel_to_base(u: float, v: float, calib: dict) -> tuple[float, float, float]:
    """Overhead pixel -> (x, y, z) on the table surface, metres, arm base frame."""
    H = np.array(calib["homography"], dtype=float)
    p = H @ np.array([u, v, 1.0])
    x, y = float(p[0] / p[2]), float(p[1] / p[2])
    return x, y, table_z_at(x, y, calib)


def pixel_dir_to_base(u: float, v: float, yaw_px_deg: float, calib: dict) -> float:
    """A direction in the image (degrees) at pixel (u, v) -> the same direction in the arm frame."""
    x0, y0, _ = pixel_to_base(u, v, calib)
    d = math.radians(yaw_px_deg)
    x1, y1, _ = pixel_to_base(u + 15 * math.cos(d), v + 15 * math.sin(d), calib)
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def fit_homography(pixels: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Least-squares homography plus per-point fit error and leave-one-out error (metres)."""
    H, _ = cv2.findHomography(pixels.astype(np.float64), xy.astype(np.float64), 0)

    def apply(Hm, pts):
        p = np.c_[pts, np.ones(len(pts))] @ Hm.T
        return p[:, :2] / p[:, 2:3]

    fit_err = np.linalg.norm(apply(H, pixels) - xy, axis=1)
    loo_err = np.full(len(pixels), np.nan)
    if len(pixels) >= 5:  # need 4 left over to refit
        for i in range(len(pixels)):
            keep = np.arange(len(pixels)) != i
            Hi, _ = cv2.findHomography(pixels[keep].astype(np.float64), xy[keep].astype(np.float64), 0)
            loo_err[i] = np.linalg.norm(apply(Hi, pixels[i : i + 1])[0] - xy[i])
    return H, fit_err, loo_err


def reach_mask(shape: tuple[int, int], calib: dict, r_min: float = REACH_MIN, r_max: float = REACH_MAX) -> np.ndarray:
    """Boolean image mask of pixels whose table point is within top-down reach."""
    h, w = shape[:2]
    us, vs = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
    H = np.array(calib["homography"], dtype=float)
    q = H @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    r = np.hypot(q[0] / q[2], q[1] / q[2]).reshape(us.shape)
    zone = ((r > r_min) & (r < r_max)).astype(np.uint8)
    return cv2.resize(zone, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def shade_reach(img: np.ndarray, calib: dict) -> np.ndarray:
    """Tint the reachable part of the overhead image green."""
    zone = reach_mask(img.shape, calib)
    out = img.copy()
    out[zone] = (0.6 * out[zone] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
    return out
