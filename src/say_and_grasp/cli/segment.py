"""Find objects by name with SAM 3.1 and show their grasp points.

    sag-segment --camera 5 --prompt "side cutters"
    sag-segment --image photo.jpg --prompt "screwdriver"

Saves outputs/segment.png (masks + red line where the jaws would close).
"""
from __future__ import annotations

import argparse
import sys

import cv2

from ..camera import capture
from ..paths import output
from ..sam import draw_overlay, grasp_from_mask, segment_robust


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", help="path to an image file")
    src.add_argument("--camera", help="camera index (e.g. 5) or /dev/v4l/by-id/... path")
    ap.add_argument("--prompt", required=True, help='short noun phrase, e.g. "side cutters"')
    ap.add_argument("--resolution", default="1920x1080", help="camera resolution, e.g. 1920x1080 or 640x480")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"Could not read {args.image}")
    else:
        w, h = (int(v) for v in args.resolution.lower().split("x"))
        img = capture(args.camera, width=w, height=h)

    dets, _ = segment_robust(img, args.prompt, max_side=min(1280, max(img.shape[:2])))
    if not dets:
        print(f"No '{args.prompt}' found.")
        cv2.imwrite(str(output("segment.png")), img)
        return
    grasps = [grasp_from_mask(d.mask) for d in dets]
    for d, g in zip(dets, grasps):
        conf = f"{d.confidence:.2f}" if d.confidence is not None else "n/a"
        print(f"object {d.object_id}: conf={conf} box={tuple(round(x) for x in d.box)} mask_px={int(d.mask.sum())}")
        if g:
            print(f"   grasp at pixel ({g.u:.0f}, {g.v:.0f}), yaw {g.yaw_deg:.0f} deg, width {g.width_px:.0f}px")
    path = output("segment.png")
    cv2.imwrite(str(path), draw_overlay(img, dets, grasps))
    print(f"Overlay saved to {path}")


if __name__ == "__main__":
    main()
