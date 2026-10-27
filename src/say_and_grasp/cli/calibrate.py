"""Calibrate the overhead camera to the SO-101's base frame.

You touch the gripper tip to a small marker (a coin works well) at 8 spots on the table.
For each spot the script records:
  * where the tip is, in the arm's base frame (forward kinematics from the joints)
  * where the marker is, in the overhead image (SAM 3.1 finds it)
It fits a homography (pixel -> table x, y in metres) plus the table height and saves
config/calib.json. Spread the points over the whole working area: the mapping is accurate
inside the area they cover and gets worse outside it.

    sag-calibrate --camera 5 --marker "blue block"
    sag-calibrate --refit --drop 7,8      # re-solve from saved points, leaving out bad ones

Torque is switched OFF so you can move the arm by hand. It sags when you let go, so keep
a hand on it.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from ..arm import Arm
from ..calibration import fit_homography, load_calib, save_calib
from ..camera import capture
from ..kinematics import Kinematics
from ..paths import CALIB_FILE, URDF_FILE, output
from ..sam import draw_overlay, grasp_from_mask, segment_robust


def locate_marker(frame: np.ndarray, prompt: str, save_as: Path) -> tuple[float, float] | None:
    """Find the marker with SAM 3.1 and return its mask centroid. Asks you if it's ambiguous."""
    dets, _ = segment_robust(frame, prompt)
    grasps = [grasp_from_mask(d.mask) for d in dets]
    cv2.imwrite(str(save_as), draw_overlay(frame, dets, grasps) if dets else frame)

    if not dets:
        print(f"  SAM found no '{prompt}' (image saved to {save_as}).")
        ans = input("  Type a different prompt to retry, 'u,v' to enter the pixel by hand, or Enter to redo: ").strip()
        if not ans:
            return None
        if "," in ans:
            u, v = (float(x) for x in ans.split(","))
            return u, v
        return locate_marker(frame, ans, save_as)

    if len(dets) == 1:
        g = grasps[0]
        print(f"  marker found at pixel ({g.u:.0f}, {g.v:.0f}) -> check {save_as}")
        return g.u, g.v

    print(f"  SAM found {len(dets)} '{prompt}' objects (see {save_as}):")
    for d, g in zip(dets, grasps):
        print(f"    id {d.object_id}: pixel ({g.u:.0f}, {g.v:.0f})")
    ans = input("  Which id is the marker? (Enter to redo this point) ").strip()
    for d, g in zip(dets, grasps):
        if d.object_id == ans:
            return g.u, g.v
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="overhead camera index (e.g. 5) or /dev/v4l/by-id/... path")
    ap.add_argument("--marker", default="coin", help='SAM prompt for your marker, e.g. "coin", "bottle cap"')
    ap.add_argument("--port", default="/dev/ttyACM0", help="follower arm serial port")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--points", type=int, default=8, help="number of touch points (min 4, 8 recommended)")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG (only if two cameras fight over USB bandwidth)")
    ap.add_argument("--urdf", default=str(URDF_FILE))
    ap.add_argument("--out", default=str(CALIB_FILE))
    ap.add_argument("--refit", action="store_true", help="re-solve from the points saved in --out")
    ap.add_argument("--drop", default="", help="with --refit: point numbers to leave out, e.g. 7,8")
    args = ap.parse_args()

    if args.refit:
        old = load_calib(args.out)
        drop = {int(x) for x in args.drop.split(",") if x.strip()}
        pts = [p for k, p in enumerate(old.get("all_points", old["points"]), start=1) if k not in drop]
        if len(pts) < 4:
            sys.exit("Need at least 4 points left.")
        print(f"Refitting with {len(pts)} points (dropped {sorted(drop) or 'none'}).")
        solve_and_save([p["pixel"] for p in pts], [p["base_xyz"] for p in pts], [p["tilt_deg"] for p in pts],
                       old["camera"], old["image_size"], old["urdf"], args.out)
        # keep the full original set so point numbers stay stable across refits
        new = load_calib(args.out)
        new["all_points"] = old.get("all_points", old["points"])
        new["dropped"] = sorted(drop)
        save_calib(new, args.out)
        return
    if not args.camera:
        sys.exit("--camera is required")

    if not Path(args.urdf).exists():
        sys.exit(f"URDF not found at {args.urdf}. See the README (assets/SO101).")

    kin = Kinematics(args.urdf)
    arm = Arm(args.port, args.robot_id)
    n_points = max(4, args.points)

    print("Checking the camera...")
    capture(args.camera, args.mjpg)

    print("\nConnecting to the arm. Hold it: torque will be switched OFF in a moment.")
    input("Press Enter when you are holding the arm... ")
    arm.connect()
    arm.robot.bus.disable_torque()

    def tip_pose() -> tuple[np.ndarray, float]:
        T = kin.fk(arm.joints())
        tilt = math.degrees(math.acos(float(np.clip(-T[2, 2], -1, 1))))  # 0 = tool straight down
        return T[:3, 3].copy(), tilt

    print(
        "\nFor each point:\n"
        "  1. Put the marker somewhere new on the table (spread them out: corners + middle of the\n"
        "     area the arm should work in, all in the overhead camera's view).\n"
        "  2. Close the jaws. Guide the gripper tip down onto the marker centre, tool pointing\n"
        "     as straight down as you can. Press Enter.\n"
        "  3. Lift the arm up and OUT of the camera view (leave the marker). Press Enter.\n"
        "     SAM 3.1 finds the marker; check outputs/calib_point_N.png if unsure.\n"
    )

    pixels, points, tilts = [], [], []
    image_size = None
    try:
        i = 0
        while i < n_points:
            print(f"--- point {i + 1}/{n_points} ---")
            input("  Tip on marker? Enter to record... ")
            xyz, tilt = tip_pose()
            print(f"  tip at x={xyz[0]*1000:.0f} y={xyz[1]*1000:.0f} z={xyz[2]*1000:.0f} mm, tilt {tilt:.0f} deg")
            if tilt > 25:
                print("  ! Tool is tilted a lot; the tip position is less reliable. Redo? [y/N] ", end="")
                if input().strip().lower() == "y":
                    continue
            input("  Arm out of view? Enter to take the picture... ")
            frame = capture(args.camera, args.mjpg)
            image_size = [frame.shape[1], frame.shape[0]]
            pt = locate_marker(frame, args.marker, output(f"calib_point_{i + 1}.png"))
            if pt is None:
                print("  redoing this point")
                continue
            pixels.append(pt)
            points.append(xyz)
            tilts.append(tilt)
            i += 1
    finally:
        arm.disconnect()

    solve_and_save(pixels, points, tilts, args.camera, image_size, args.urdf, args.out)


def solve_and_save(pixels, points, tilts, camera, image_size, urdf, out) -> None:
    pixels_a = np.array(pixels, dtype=float)
    points_a = np.array(points, dtype=float)
    H, fit_err, loo_err = fit_homography(pixels_a, points_a[:, :2])
    table_z = float(np.median(points_a[:, 2]))

    print("\nPoint   pixel        base x,y (mm)      fit err   leave-one-out err")
    for k, ((u, v), p, e, l) in enumerate(zip(pixels_a, points_a, fit_err, loo_err)):
        loo = f"{l*1000:5.1f} mm" if not np.isnan(l) else "   n/a"
        print(f"{k+1:>4}   ({u:4.0f},{v:4.0f})   ({p[0]*1000:6.0f},{p[1]*1000:6.0f})   {e*1000:5.1f} mm   {loo}")
    z_spread = (points_a[:, 2].max() - points_a[:, 2].min()) * 1000
    print(f"\nTable height z = {table_z*1000:.0f} mm (spread {z_spread:.0f} mm across touches)")
    # Leave-one-out at a corner point measures extrapolation, so judge by the median.
    typical = np.nanmedian(loo_err) if not np.all(np.isnan(loo_err)) else float(np.median(fit_err))
    print(f"Typical error (median leave-one-out): {typical*1000:.1f} mm  ->  "
          + ("good" if typical < 0.005 else "OK, usable" if typical < 0.010 else "too high: refit with --drop, or redo"))
    print("A point with a much larger error than the rest was probably bumped or touched off-centre;")
    print("leave it out with: sag-calibrate --refit --drop N")

    save_calib(
        {
            "homography": H.tolist(),
            "table_z": table_z,
            "camera": str(camera),
            "image_size": image_size,
            "urdf": urdf,
            "points": [{"pixel": list(map(float, px)), "base_xyz": list(map(float, p)), "tilt_deg": float(t)}
                       for px, p, t in zip(pixels_a, points_a, tilts)],
            "fit_err_mm": (fit_err * 1000).round(2).tolist(),
            "loo_err_mm": [None if np.isnan(x) else round(float(x) * 1000, 2) for x in loo_err],
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        out,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
