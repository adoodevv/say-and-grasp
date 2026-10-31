"""SAM 3.1 through Meta Model API: text prompt -> masks -> top-down grasp targets.

segment(image, "side cutters") returns Detections with full-frame masks in the original
image's coordinates; grasp_from_mask() turns a mask into a centroid + gripper yaw.
"""
from __future__ import annotations

import base64
import math
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
from meta_sam_parser import (
    CompletedOutcome,
    SegmentationBoxRecord,
    SegmentationMaskRecord,
    decode_mask_to_raster,
    image_segmentation_format,
)

BASE_URL = "https://api.meta.ai/v1"
MODEL = "sam-3.1"
MAX_SIDE = 640  # downscale before upload: faster round trips, cheaper, enough for grasping
OVERHEAD_SIDE = 1280  # overhead photos: thin tools (tweezers, a brush) need the extra pixels


@dataclass
class Detection:
    object_id: str
    box: tuple[float, float, float, float]  # left, top, right, bottom in ORIGINAL image pixels
    mask: np.ndarray  # bool array, same HxW as the original image
    confidence: float | None = None


@dataclass
class GraspTarget:
    u: float  # pixel x of the mask centroid
    v: float  # pixel y of the mask centroid
    yaw_deg: float  # gripper yaw: close the jaws ACROSS the object's short axis
    width_px: float  # object extent along the closing direction (sanity check vs. jaw opening)


_use_jpeg = True  # ~7x smaller uploads than PNG; falls back to PNG if the API ever rejects it


def _encode_image(img_bgr: np.ndarray, max_side: int = MAX_SIDE, jpeg: bool = True) -> tuple[str, float]:
    h, w = img_bgr.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    if jpeg:
        ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        mime = "image/jpeg"
    else:
        ok, buf = cv2.imencode(".png", img_bgr)
        mime = "image/png"
    if not ok:
        raise RuntimeError("Could not encode image")
    return f"data:{mime};base64," + base64.b64encode(buf.tobytes()).decode(), scale


def parse_output(output_text: str, sent_shape: tuple[int, int], orig_shape: tuple[int, int]) -> list[Detection]:
    """Turn SAM's output_text into Detections in original-image coordinates."""
    parser = image_segmentation_format().create_parser()
    parser.push(output_text if output_text.endswith("\n") else output_text + "\n")
    result = parser.finish(CompletedOutcome()).result
    for d in result.diagnostics:
        print(f"[sam] parser diagnostic: {d}", file=sys.stderr)

    sh, sw = sent_shape
    oh, ow = orig_shape
    sx, sy = ow / sw, oh / sh

    boxes: dict[str, SegmentationBoxRecord] = {}
    detections: list[Detection] = []
    for rec in result.records:
        if isinstance(rec, SegmentationBoxRecord):
            boxes[rec.object_id] = rec
        elif isinstance(rec, SegmentationMaskRecord):
            m = rec.mask
            local = np.frombuffer(decode_mask_to_raster(m), dtype=np.uint8).reshape(m.height, m.width)
            # The mask raster is box-local: stretch it over its bounds in the SENT frame.
            b = rec.bounds
            x0, y0 = int(round(b.left)), int(round(b.top))
            x1, y1 = int(round(b.right)), int(round(b.bottom))
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(sw, max(x1, x0 + 1)), min(sh, max(y1, y0 + 1))
            full = np.zeros((sh, sw), dtype=np.uint8)
            full[y0:y1, x0:x1] = cv2.resize(local, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
            # Back to the original resolution.
            full = cv2.resize(full, (ow, oh), interpolation=cv2.INTER_NEAREST).astype(bool)
            bx = boxes.get(rec.object_id)
            src = bx if bx is not None else b
            conf = rec.confidence if rec.confidence is not None else (bx.confidence if bx else None)
            detections.append(
                Detection(
                    object_id=rec.object_id,
                    box=(src.left * sx, src.top * sy, src.right * sx, src.bottom * sy),
                    mask=full,
                    confidence=conf,
                )
            )
    return detections


_client = None


def get_client():
    """One shared OpenAI-compatible client for Meta Model API (reads META_API_KEY)."""
    global _client
    if _client is None:
        from openai import OpenAI

        if "META_API_KEY" not in os.environ:
            raise RuntimeError("Set META_API_KEY first (export META_API_KEY=...)")
        _client = OpenAI(base_url=BASE_URL, api_key=os.environ["META_API_KEY"])
    return _client


def segment(img_bgr: np.ndarray, prompt: str, client=None, max_side: int = MAX_SIDE) -> list[Detection]:
    """Segment every instance of `prompt` in a BGR image. Returns [] when nothing matches.
    Detections come back in the ORIGINAL image's pixel coordinates, whatever max_side is."""
    global _use_jpeg
    client = client or get_client()
    oh, ow = img_bgr.shape[:2]

    def call(jpeg: bool):
        data_url, scale = _encode_image(img_bgr, max_side, jpeg)
        resp = client.responses.create(
            model=MODEL,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ]}],
        )
        return resp, scale

    t0 = time.time()
    try:
        resp, scale = call(_use_jpeg)
    except Exception as e:  # e.g. the API refusing JPEG: switch to PNG for the rest of the session
        if not _use_jpeg:
            raise
        print(f"[sam] JPEG upload failed ({e.__class__.__name__}); using PNG from now on", file=sys.stderr)
        _use_jpeg = False
        resp, scale = call(False)
    sent_shape = (round(oh * scale), round(ow * scale)) if scale < 1.0 else (oh, ow)
    text = resp.output_text or ""
    print(f"[sam] '{prompt}': {time.time() - t0:.2f}s round trip, {len(text)} chars", file=sys.stderr)
    if not text.strip():
        return []
    return parse_output(text, sent_shape, (oh, ow))


def prompt_variants(prompt: str) -> list[str]:
    """Simpler fallbacks for a prompt. SAM 3.1 can miss "blue lego block" yet find "blue block".

    Tries the prompt as given, then without brand/filler words, then the colour + last noun,
    then the last noun alone.
    """
    filler = {"lego", "a", "an", "the", "my", "small", "little", "big", "piece", "of"}
    words = prompt.lower().split()
    variants = [prompt]
    trimmed = [w for w in words if w not in filler]
    if trimmed:
        variants.append(" ".join(trimmed))
        if len(trimmed) > 2:
            variants.append(" ".join([trimmed[0], trimmed[-1]]))
        variants.append(trimmed[-1])
    seen, out = set(), []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def segment_robust(img_bgr: np.ndarray, prompt: str, max_side: int = MAX_SIDE) -> tuple[list[Detection], str]:
    """segment() with automatic prompt fallbacks. Returns (detections, the prompt that worked)."""
    for p in prompt_variants(prompt):
        dets = segment(img_bgr, p, max_side=max_side)
        if dets:
            if p != prompt:
                print(f"[sam] '{prompt}' found nothing; '{p}' worked", file=sys.stderr)
            return dets, p
    return [], prompt


def grasp_from_mask(mask: np.ndarray) -> GraspTarget | None:
    """Top-down grasp: centroid + principal axes from image moments."""
    m = cv2.moments(mask.astype(np.uint8), binaryImage=True)
    if m["m00"] < 1:
        return None
    u, v = m["m10"] / m["m00"], m["m01"] / m["m00"]
    mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
    long_axis = 0.5 * math.atan2(2 * mu11, mu20 - mu02)  # radians, image frame (y down)
    close_dir = long_axis + math.pi / 2  # squeeze across the short side
    # Extent of the mask along the closing direction.
    ys, xs = np.nonzero(mask)
    proj = (xs - u) * math.cos(close_dir) + (ys - v) * math.sin(close_dir)
    return GraspTarget(u=u, v=v, yaw_deg=math.degrees(close_dir) % 180, width_px=float(proj.max() - proj.min()))


def draw_overlay(img_bgr: np.ndarray, dets: list[Detection], grasps: list[GraspTarget | None]) -> np.ndarray:
    out = img_bgr.copy()
    colours = [(255, 120, 0), (0, 200, 255), (0, 255, 120), (200, 0, 255)]
    for i, (d, g) in enumerate(zip(dets, grasps)):
        c = colours[i % len(colours)]
        layer = out.copy()
        layer[d.mask] = c
        out = cv2.addWeighted(layer, 0.45, out, 0.55, 0)
        l, t, r, b = map(int, d.box)
        cv2.rectangle(out, (l, t), (r, b), c, 2)
        label = f"id {d.object_id}" + (f" {d.confidence:.2f}" if d.confidence is not None else "")
        cv2.putText(out, label, (l, max(12, t - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        if g:
            half = g.width_px / 2 + 8
            dx, dy = math.cos(math.radians(g.yaw_deg)) * half, math.sin(math.radians(g.yaw_deg)) * half
            p1 = (int(g.u - dx), int(g.v - dy))
            p2 = (int(g.u + dx), int(g.v + dy))
            cv2.line(out, p1, p2, (0, 0, 255), 2)  # red line = where the jaws close
            cv2.circle(out, (int(g.u), int(g.v)), 4, (0, 0, 255), -1)
    return out
