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
    sag-calibrate --tcp                   # measure your gripper's grasp centre (do this FIRST)
    sag-calibrate --wrist --wrist-camera 2 --marker "blue block"   # wrist-camera alignment (last)
    sag-calibrate --refit --marker-height 10   # touched the TOP of a 10 mm block? correct the table height

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
from ..calibration import fit_homography, fit_table_plane, load_calib, save_calib
from ..camera import capture
from ..kinematics import Kinematics, solve_tcp_offset
from ..paths import CALIB_FILE, TOOL_FILE, URDF_FILE, WRIST_FILE, output
from ..sam import draw_overlay, grasp_from_mask, segment_robust


def locate_marker(frame: np.ndarray, prompt: str, save_as: Path) -> tuple[float, float] | None:
    """Find the marker with SAM 3.1 and return its mask centroid. Asks you if it's ambiguous."""
    dets, _ = segment_robust(frame, prompt, max_side=min(1280, max(frame.shape[:2])))
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


def measure_tcp(args) -> None:
    """Touch ONE fixed point with the grasp centre at several wrist twists; solve the offset."""
    import json

    if not Path(args.urdf).exists():
        sys.exit(f"URDF not found at {args.urdf}. See the README (assets/SO101).")
    kin = Kinematics(args.urdf, tcp_offset=np.zeros(3))  # raw tip frame
    arm = Arm(args.port, args.robot_id)
    print(
        "\nMeasuring the gripper's grasp centre.\n"
        "  Put a small, fixed target on the table (a dot drawn on paper taped down works best).\n"
        "  Each time: close the jaws, point the gripper straight down, and touch the target with\n"
        "  the point MIDWAY BETWEEN THE JAW TIPS. Twist the wrist to a different angle each time\n"
        "  (roughly -60, 0, +60 deg, and a fourth anywhere). Keep the target itself still.\n"
    )
    input("Hold the arm (torque goes OFF), then Enter... ")
    arm.connect()
    arm.robot.bus.disable_torque()
    poses = []
    try:
        while len(poses) < 4:
            input(f"  touch {len(poses) + 1}/4: grasp centre on the target? Enter... ")
            q = arm.joints()
            T = kin.fk(q)
            tilt = math.degrees(math.acos(float(np.clip(-T[2, 2], -1, 1))))
            print(f"    wrist_roll {q[4]:6.1f} deg, tilt {tilt:4.1f} deg")
            if poses:
                rolls = [p[1] for p in poses]
                if min(abs(q[4] - r) for r in rolls) < 20:
                    print("    ! too close to an earlier wrist angle; twist further and redo")
                    continue
            poses.append((T, q[4]))
    finally:
        arm.disconnect()

    d, c, res = solve_tcp_offset([T for T, _ in poses])
    print(f"\nGrasp centre offset (tool frame): x {d[0]*1000:+.1f} mm, y {d[1]*1000:+.1f} mm "
          f"(|d| = {np.linalg.norm(d)*1000:.1f} mm)")
    print("Consistency per touch: " + ", ".join(f"{r*1000:.1f}" for r in res) + " mm "
          + ("(good)" if res.max() < 0.004 else "(noisy: redo more carefully)"))
    TOOL_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOOL_FILE.write_text(json.dumps({"tcp_offset": d.round(5).tolist(), "residual_mm": (res * 1000).round(2).tolist(),
                                     "created": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=2))
    print(f"Saved {TOOL_FILE}. Now redo the camera calibration: sag-calibrate --camera N --marker \"blue block\"")


def calibrate_wrist(args) -> None:
    """Learn where a centred object appears in the wrist camera, and how pixels map to millimetres."""
    import json

    from ..skills import BIAS_FILE, Bench

    if not args.wrist_camera:
        sys.exit("--wrist-camera is required (e.g. 2, or a /dev/v4l/by-id/... path)")
    # The learned overhead-vs-wrist offset is measured against this calibration: start it afresh,
    # and don't let the old one shift where the arm hovers while calibrating.
    if BIAS_FILE.exists():
        BIAS_FILE.unlink()
        print("Reset the learned overhead offset (it relearns over the first few picks).")
    bench = Bench(port=args.port, robot_id=args.robot_id, camera=args.camera)
    bench.wrist = None  # don't align while calibrating the aligner
    bench.bias = {"dx": 0.0, "dy": 0.0, "dyaw": 0.0, "n": 0}
    cam, prompt = args.wrist_camera, args.marker

    last_yaw = {}

    def wrist_centroid(near: np.ndarray | None, tag: str) -> np.ndarray:
        while True:
            img = capture(cam, args.mjpg)
            dets, _ = segment_robust(img, prompt)
            gs = [grasp_from_mask(d.mask) for d in dets] if dets else []
            gs = [g for g in gs if g]
            path = output(f"wrist_calib_{tag}.png")
            cv2.imwrite(str(path), draw_overlay(img, dets, [grasp_from_mask(d.mask) for d in dets]) if dets else img)
            if gs:
                ref = near if near is not None else np.array([img.shape[1] / 2, img.shape[0] / 2])
                g = min(gs, key=lambda g: np.hypot(g.u - ref[0], g.v - ref[1]))
                last_yaw[tag] = g.yaw_deg
                print(f"    wrist camera sees it at pixel ({g.u:.0f}, {g.v:.0f}), angle {g.yaw_deg:.0f} deg  [{path.name}]")
                return np.array([g.u, g.v])
            input(f"    SAM can't see '{prompt}' in the wrist camera ({path}). Fix the view, then Enter... ")

    try:
        bench.connect()
        seen = bench.look(prompt)
        above = bench._solve("above object", seen.x, seen.y, seen.z + bench.hov, seen.close_dir)
        arm = bench.arm
        arm.leave_home()
        arm.move(above, gripper=bench.g_open)
        print("\nThe arm is hovering over the block with the jaws open.")
        input("Look from the side and nudge the BLOCK (not the arm) until it sits centred between the jaws\n"
              "AND square to them (long side parallel to the jaw faces). Enter... ")
        p0 = wrist_centroid(None, "centre")

        R2 = bench.tool_xy_axes(above)
        step = 0.015
        cols = []
        for k, axis in enumerate(("x", "y")):
            d_tool = np.zeros(2)
            d_tool[k] = step
            dx, dy = R2 @ d_tool
            q = bench._solve(f"shift {axis}", seen.x + dx, seen.y + dy, seen.z + bench.hov, seen.close_dir)
            arm.move(q)
            time.sleep(0.3)
            p = wrist_centroid(p0, f"shift_{axis}")
            cols.append((p - p0) / step)
            arm.move(above)
        A = np.column_stack(cols)  # pixel shift per metre of gripper motion along tool x / y
        scale = np.sqrt(abs(np.linalg.det(A))) / 1000
        print(f"\nWrist camera: {scale:.1f} px per mm of gripper motion")
        if scale < 1:
            print("  ! very small: the block barely moved in the image. Is the wrist camera index right?")

        # Check: move to a known offset and see if the alignment math recovers it.
        test = np.array([0.008, -0.006])
        dx, dy = R2 @ test
        q = bench._solve("check", seen.x + dx, seen.y + dy, seen.z + bench.hov, seen.close_dir)
        arm.move(q)
        time.sleep(0.3)
        p = wrist_centroid(p0, "check")
        est = np.linalg.solve(A, p - p0)  # A maps gripper motion -> pixel shift, so invert it
        err = np.hypot(*(est - test)) * 1000
        print(f"Check: moved {test*1000} mm, wrist camera says {np.round(est*1000, 1)} mm -> error {err:.1f} mm "
              + ("(good)" if err < 3 else "(rough: redo with the block better centred)"))

        # Angle: turn the wrist 20 deg and see how the block's angle changes in the image.
        arm.move(above)
        q = bench._solve("turn", seen.x, seen.y, seen.z + bench.hov, seen.close_dir + 20.0)
        arm.move(q)
        time.sleep(0.3)
        wrist_centroid(p0, "turn")
        from ..skills import _wrap90
        yaw_gain = _wrap90(last_yaw["turn"] - last_yaw["centre"]) / 20.0
        print(f"Angle: turning the gripper 20 deg turned the block {yaw_gain * 20:+.0f} deg in the image")
        if not 0.5 < abs(yaw_gain) < 1.5:
            print("  ! expected about +/-20 deg; the block may be too square or SAM's mask was off. "
                  "Angle correction will be skipped.")
        arm.move(above)

        WRIST_FILE.parent.mkdir(parents=True, exist_ok=True)
        WRIST_FILE.write_text(json.dumps({
            "camera": str(cam),
            "target_px": p0.round(1).tolist(),
            "px_per_m": (-A).round(1).tolist(),  # sign: object offset (not gripper motion) -> pixel error
            **({"target_yaw_px": round(float(last_yaw["centre"]), 1), "yaw_gain": round(float(yaw_gain), 3)}
               if 0.5 < abs(yaw_gain) < 1.5 else {}),
            "hover_mm": bench.hov * 1000,
            "check_err_mm": round(float(err), 2),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2))
        print(f"Saved {WRIST_FILE}. Picks now centre the gripper with the wrist camera before descending.")
        bench.go_home()
    finally:
        bench.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="overhead camera index (e.g. 5) or /dev/v4l/by-id/... path")
    ap.add_argument("--marker", default="coin", help='SAM prompt for your marker, e.g. "coin", "bottle cap"')
    ap.add_argument("--marker-height", type=float, default=None,
                    help="height of the marker in mm, if you touch its TOP (e.g. 10 for a LEGO brick); "
                         "subtracted so the table height is the real table")
    ap.add_argument("--port", default="/dev/ttyACM0", help="follower arm serial port")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--points", type=int, default=8, help="number of touch points (min 4, 8 recommended)")
    ap.add_argument("--mjpg", action="store_true", help="force MJPG (only if two cameras fight over USB bandwidth)")
    ap.add_argument("--resolution", default="1920x1080",
                    help="overhead camera resolution to calibrate (and then run) at, e.g. 1920x1080 or 640x480")
    ap.add_argument("--urdf", default=str(URDF_FILE))
    ap.add_argument("--out", default=str(CALIB_FILE))
    ap.add_argument("--refit", action="store_true", help="re-solve from the points saved in --out")
    ap.add_argument("--drop", default="", help="with --refit: point numbers to leave out, e.g. 7,8")
    ap.add_argument("--tcp", action="store_true", help="measure the gripper's grasp centre (writes config/tool.json)")
    ap.add_argument("--wrist", action="store_true", help="calibrate wrist-camera alignment (writes config/wrist.json)")
    ap.add_argument("--wrist-camera", help="wrist camera index or /dev/v4l/by-id path (for --wrist)")
    args = ap.parse_args()

    if args.tcp:
        measure_tcp(args)
        return
    if args.wrist:
        calibrate_wrist(args)
        return

    if args.refit:
        old = load_calib(args.out)
        drop = {int(x) for x in args.drop.split(",") if x.strip()}
        pts = [p for k, p in enumerate(old.get("all_points", old["points"]), start=1) if k not in drop]
        if len(pts) < 4:
            sys.exit("Need at least 4 points left.")
        print(f"Refitting with {len(pts)} points (dropped {sorted(drop) or 'none'}).")
        mh = args.marker_height if args.marker_height is not None else old.get("marker_height_mm", 0.0)
        solve_and_save([p["pixel"] for p in pts], [p["base_xyz"] for p in pts], [p["tilt_deg"] for p in pts],
                       old["camera"], old["image_size"], old["urdf"], args.out, marker_height_mm=mh)
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

    res_w, res_h = (int(v) for v in args.resolution.lower().split("x"))
    print("Checking the camera...")
    test = capture(args.camera, args.mjpg, width=res_w, height=res_h)
    if (test.shape[1], test.shape[0]) != (res_w, res_h):
        print(f"  ! The camera gave {test.shape[1]}x{test.shape[0]}, not {res_w}x{res_h}; calibrating at "
              f"{test.shape[1]}x{test.shape[0]}.")
    else:
        print(f"  {res_w}x{res_h} OK")
    # Spots (e.g. handover) were set as pixels in the old image: keep their table positions.
    try:
        from ..spots import freeze

        n = freeze(load_calib(args.out))
        if n:
            print(f"  kept the table position of {n} saved spot(s) (handover etc.)")
    except Exception:
        pass
    # The overhead offset Wiggle learned belongs to the old calibration: start that afresh.
    from ..skills import BIAS_FILE

    if BIAS_FILE.exists():
        BIAS_FILE.unlink()
        print("  reset the learned overhead offset (it relearns over the first few picks)")

    print("\nConnecting to the arm. Hold it: torque will be switched OFF in a moment.")
    input("Press Enter when you are holding the arm... ")
    arm.connect()
    arm.robot.bus.disable_torque()

    def tip_pose() -> tuple[np.ndarray, float]:
        T = kin.fk_tcp(arm.joints())  # grasp centre, if config/tool.json exists
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
            frame = capture(args.camera, args.mjpg, width=res_w, height=res_h)
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

    solve_and_save(pixels, points, tilts, args.camera, image_size, args.urdf, args.out,
                   marker_height_mm=args.marker_height or 0.0)


def solve_and_save(pixels, points, tilts, camera, image_size, urdf, out, marker_height_mm: float = 0.0) -> None:
    pixels_a = np.array(pixels, dtype=float)
    points_a = np.array(points, dtype=float)  # raw touches, saved as-is
    surf = points_a.copy()
    surf[:, 2] -= marker_height_mm / 1000  # touched the marker's top -> the table is this much lower
    if marker_height_mm:
        print(f"Correcting for a {marker_height_mm:.0f} mm tall marker (touched on top).")
    H, radial, fit_err, loo_err = fit_homography(pixels_a, points_a[:, :2])
    if radial:
        print(f"Reach correction: {radial[0]*100:+.1f} mm per 10 cm from the base, {radial[1]*1000:+.1f} mm offset "
              "(absorbs the arm's own reach error)")
    table_z = float(np.median(surf[:, 2]))

    print("\nPoint   pixel        base x,y (mm)      fit err   leave-one-out err")
    for k, ((u, v), p, e, l) in enumerate(zip(pixels_a, points_a, fit_err, loo_err)):
        loo = f"{l*1000:5.1f} mm" if not np.isnan(l) else "   n/a"
        print(f"{k+1:>4}   ({u:4.0f},{v:4.0f})   ({p[0]*1000:6.0f},{p[1]*1000:6.0f})   {e*1000:5.1f} mm   {loo}")
    z_spread = (points_a[:, 2].max() - points_a[:, 2].min()) * 1000
    plane = fit_table_plane(surf)
    plane_res = surf[:, 2] - (np.c_[surf[:, :2], np.ones(len(surf))] @ plane)
    print(f"\nTable height z = {table_z*1000:.0f} mm (spread {z_spread:.0f} mm across touches)")
    print(f"Table plane: tilts {plane[0]*100:+.1f} mm per 10 cm in x, {plane[1]*100:+.1f} mm per 10 cm in y; "
          f"touches within {np.abs(plane_res).max()*1000:.1f} mm of it")
    # Leave-one-out at a corner point measures extrapolation, so judge by the median.
    typical = np.nanmedian(loo_err) if not np.all(np.isnan(loo_err)) else float(np.median(fit_err))
    print(f"Typical error (median leave-one-out): {typical*1000:.1f} mm  ->  "
          + ("good" if typical < 0.005 else "OK, usable" if typical < 0.010 else "too high: refit with --drop, or redo"))
    print("A point with a much larger error than the rest was probably bumped or touched off-centre;")
    print("leave it out with: sag-calibrate --refit --drop N")

    save_calib(
        {
            "homography": H.tolist(),
            "radial": radial,
            "table_z": table_z,
            "table_plane": plane.tolist(),
            "marker_height_mm": marker_height_mm,
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
