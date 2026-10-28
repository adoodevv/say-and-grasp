"""Named places on the bench, e.g. "handover" (where tools are put down for you).

Stored in config/spots.json as overhead-image pixels, converted to arm coordinates with the
calibration when used, so they survive a recalibration as long as the camera doesn't move.

    sag-spots set handover 420,280      # pixel in the overhead image
    sag-spots list
"""
from __future__ import annotations

import argparse
import json

from .calibration import load_calib, pixel_to_base
from .paths import CONFIG_DIR

SPOTS_FILE = CONFIG_DIR / "spots.json"


def load_spots() -> dict[str, list[float]]:
    if SPOTS_FILE.exists():
        return json.loads(SPOTS_FILE.read_text())
    return {}


def save_spot(name: str, u: float, v: float) -> None:
    spots = load_spots()
    spots[name] = [float(u), float(v)]
    SPOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPOTS_FILE.write_text(json.dumps(spots, indent=2))


def spot_xy(name: str, calib: dict | None = None) -> tuple[float, float]:
    spots = load_spots()
    if name not in spots:
        raise KeyError(f"No spot called '{name}'. Set it with: sag-spots set {name} U,V")
    x, y, _ = pixel_to_base(*spots[name], calib or load_calib())
    return x, y


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set", help="save a named spot at an overhead-image pixel")
    s.add_argument("name")
    s.add_argument("pixel", help="u,v")
    sub.add_parser("list")
    args = ap.parse_args()
    if args.cmd == "set":
        u, v = (float(t) for t in args.pixel.split(","))
        save_spot(args.name, u, v)
        x, y = spot_xy(args.name)
        print(f"Saved '{args.name}' at pixel ({u:.0f},{v:.0f}) -> arm ({x*1000:.0f}, {y*1000:.0f}) mm")
    else:
        calib = load_calib()
        for name, (u, v) in load_spots().items():
            x, y, _ = pixel_to_base(u, v, calib)
            print(f"{name:12s} pixel ({u:.0f},{v:.0f}) -> arm ({x*1000:.0f}, {y*1000:.0f}) mm")


if __name__ == "__main__":
    main()
