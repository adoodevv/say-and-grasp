"""Overhead camera <-> arm base frame: a homography from image pixels to the table plane."""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np

from .paths import CALIB_FILE

# Where the SO-101 can grasp in any direction (metres from the base); further out for some tool
# directions by leaning the gripper. Close to the base, watch that the gripper clears the arm.
REACH_MIN, REACH_MAX = 0.10, 0.36
PAN_AXIS = (0.0388, 0.0)  # the shoulder pan axis in the base frame (SO-101 URDF)


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
    x, y = _radial(np.array([[p[0] / p[2], p[1] / p[2]]]), calib.get("radial"))[0]
    x, y = float(x), float(y)
    return x, y, table_z_at(x, y, calib)


def _radial(xy: np.ndarray, coef, sign: float = 1.0) -> np.ndarray:
    """Distance-from-the-pan-axis correction: r -> r + sign*(c1*r + c0). It absorbs the arm's own
    reach error (folded close to the base its joint readings are a centimetre or so off), so
    the mapping matches where the arm actually goes."""
    if coef is None or len(coef) == 0:
        return xy
    a = np.array(PAN_AXIS)
    rel = xy - a
    r = np.linalg.norm(rel, axis=1, keepdims=True)
    return a + rel / np.maximum(r, 1e-9) * (r + sign * np.polyval(coef, r))


def pixel_dir_to_base(u: float, v: float, yaw_px_deg: float, calib: dict) -> float:
    """A direction in the image (degrees) at pixel (u, v) -> the same direction in the arm frame."""
    x0, y0, _ = pixel_to_base(u, v, calib)
    d = math.radians(yaw_px_deg)
    x1, y1, _ = pixel_to_base(u + 15 * math.cos(d), v + 15 * math.sin(d), calib)
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


def _apply_h(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.c_[pts, np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3]


def _fit_once(pixels: np.ndarray, xy: np.ndarray, radial: bool) -> tuple[np.ndarray, list | None]:
    H, _ = cv2.findHomography(pixels.astype(np.float64), xy.astype(np.float64), 0)
    if not radial or len(pixels) < 7:
        return H, None
    a = np.array(PAN_AXIS)
    coef = np.zeros(2)
    for _ in range(6):  # alternate: radial error given H, then H given the radial correction
        pred = _apply_h(H, pixels)
        rel = pred - a
        r = np.linalg.norm(rel, axis=1)
        r_true = np.einsum("ij,ij->i", xy - a, rel / r[:, None])
        coef = np.linalg.lstsq(np.c_[r, np.ones_like(r)], r_true - r, rcond=None)[0]
        H, _ = cv2.findHomography(pixels.astype(np.float64), _radial(xy, coef, sign=-1.0), 0)
    return H, [float(c) for c in coef]


def fit_homography(pixels: np.ndarray, xy: np.ndarray, radial: bool = True):
    """Homography (+ a distance-from-base correction when there are 7+ points). Returns
    (H, radial coefficients or None, per-point fit error, leave-one-out error), errors in metres."""
    H, coef = _fit_once(pixels, xy, radial)

    def predict(Hm, c, pts):
        return _radial(_apply_h(Hm, pts), c)

    fit_err = np.linalg.norm(predict(H, coef, pixels) - xy, axis=1)
    loo_err = np.full(len(pixels), np.nan)
    if len(pixels) >= 5:  # need 4 left over to refit
        for i in range(len(pixels)):
            keep = np.arange(len(pixels)) != i
            Hi, ci = _fit_once(pixels[keep], xy[keep], radial and coef is not None)
            loo_err[i] = np.linalg.norm(predict(Hi, ci, pixels[i : i + 1])[0] - xy[i])
    return H, coef, fit_err, loo_err


def reach_mask(shape: tuple[int, int], calib: dict, r_min: float = REACH_MIN, r_max: float = REACH_MAX) -> np.ndarray:
    """Boolean image mask of pixels whose table point is within top-down reach."""
    h, w = shape[:2]
    us, vs = np.meshgrid(np.arange(0, w, 4), np.arange(0, h, 4))
    H = np.array(calib["homography"], dtype=float)
    q = H @ np.stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    xy = _radial(np.c_[q[0] / q[2], q[1] / q[2]], calib.get("radial"))
    r = np.hypot(xy[:, 0], xy[:, 1]).reshape(us.shape)
    zone = ((r > r_min) & (r < r_max)).astype(np.uint8)
    return cv2.resize(zone, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)


def shade_reach(img: np.ndarray, calib: dict) -> np.ndarray:
    """Tint the reachable part of the overhead image green."""
    zone = reach_mask(img.shape, calib)
    out = img.copy()
    out[zone] = (0.6 * out[zone] + 0.4 * np.array([60, 200, 60])).astype(np.uint8)
    return out
