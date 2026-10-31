"""Named places on the bench, e.g. "handover" (where tools are put down for you).

Stored in config/spots.json as a position on the table (arm coordinates), plus the overhead pixel
it was set at. The table position is what's used, so spots survive a recalibration or a camera
resolution change.

    sag-spots set handover 420,280      # pixel in the overhead image
    sag-spots list
"""
from __future__ import annotations

import argparse
import json

from .calibration import load_calib, pixel_to_base
from .paths import CONFIG_DIR

SPOTS_FILE = CONFIG_DIR / "spots.json"


def load_spots() -> dict[str, dict]:
    if not SPOTS_FILE.exists():
        return {}
    raw = json.loads(SPOTS_FILE.read_text())
    # older files stored just [u, v] pixels
    return {k: (v if isinstance(v, dict) else {"pixel": v}) for k, v in raw.items()}


def _write(spots: dict) -> None:
    SPOTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPOTS_FILE.write_text(json.dumps(spots, indent=2))


def save_spot(name: str, u: float, v: float, calib: dict | None = None) -> None:
    calib = calib or load_calib()
    x, y, _ = pixel_to_base(u, v, calib)
    spots = load_spots()
    spots[name] = {"pixel": [float(u), float(v)], "image_size": calib.get("image_size", [640, 480]),
                   "xy": [round(x, 4), round(y, 4)]}
    _write(spots)


def freeze(calib: dict) -> int:
    """Store every spot's table position using `calib` (call before replacing the calibration)."""
    spots = load_spots()
    n = 0
    for name, s in spots.items():
        if "xy" not in s:
            x, y, _ = pixel_to_base(*s["pixel"], calib)
            s["xy"] = [round(x, 4), round(y, 4)]
            s.setdefault("image_size", calib.get("image_size", [640, 480]))
            n += 1
    if n:
        _write(spots)
    return n


def spot_xy(name: str, calib: dict | None = None) -> tuple[float, float]:
    spots = load_spots()
    if name not in spots:
        raise KeyError(f"No spot called '{name}'. Set it with: sag-spots set {name} U,V")
    s = spots[name]
    if "xy" in s:
        return tuple(s["xy"])
    x, y, _ = pixel_to_base(*s["pixel"], calib or load_calib())
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
        for name in load_spots():
            x, y = spot_xy(name)
            print(f"{name:12s} arm ({x*1000:.0f}, {y*1000:.0f}) mm")


if __name__ == "__main__":
    main()
