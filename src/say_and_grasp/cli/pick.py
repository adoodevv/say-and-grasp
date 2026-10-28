"""Pick up a named object and put it somewhere: SAM 3.1 + calibration + top-down IK.

    # first time: move the arm by hand to a folded pose clear of the camera
    sag-pick --save-home
    # then: from home, lift just the wrist until the gripper clears the base
    sag-pick --save-clear

    # see where the arm can reach (green) in the overhead view
    sag-pick --show-reach

    # plan only (no motion), then do it
    sag-pick --prompt "blue lego block" --place-pixel 200,250 --dry-run
    sag-pick --prompt "blue lego block" --place-pixel 200,250

--place-pixel is a spot in the overhead image (u,v); --place-xy is in mm in the arm frame.
Without either, the object is lifted, held for a moment and put back where it was.
Writes outputs/pick_plan.png: detection, grasp line, place spot and the reachable zone.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from ..arm import Arm, load_home, save_clear, save_home
from ..calibration import REACH_MAX, REACH_MIN, load_calib, pixel_dir_to_base, pixel_to_base, shade_reach, table_z_at
from ..camera import capture
from ..kinematics import Kinematics
from ..paths import CALIB_FILE, URDF_FILE, output
from ..sam import draw_overlay, grasp_from_mask, segment_robust


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help='what to pick, e.g. "side cutters"')
    ap.add_argument("--place-pixel", help="u,v in the overhead image")
    ap.add_argument("--place-xy", help="x,y in mm, arm frame")
    ap.add_argument("--camera", default=None, help="overhead camera (default: the one in calib.json)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--urdf", default=str(URDF_FILE))
    ap.add_argument("--calib", default=str(CALIB_FILE))
    ap.add_argument("--grasp-height", type=float, default=5,
                    help="jaw-tip height above the table at grasp, mm (about half the object height)")
    ap.add_argument("--approach", type=float, default=50, help="hover height above table, mm")
    ap.add_argument("--open", type=float, default=60, help="gripper value when open (0-100)")
    ap.add_argument("--closed", type=float, default=5, help="gripper value when closed (0-100)")
    ap.add_argument("--yaw-offset", type=float, default=0, help="degrees added to the grasp angle")
    ap.add_argument("--speed", type=float, default=30, help="joint speed, deg/s")
    ap.add_argument("--pause", action="store_true",
                    help="stop just above the object so you can check the jaw angle before grasping")
    ap.add_argument("--dry-run", action="store_true", help="plan and print, don't move")
    ap.add_argument("--yes", action="store_true", help="don't ask before moving")
    ap.add_argument("--save-home", action="store_true", help="record the current pose as home and exit")
    ap.add_argument("--save-clear", action="store_true",
                    help="record a 'clear' pose (home with the wrist lifted off the base) and exit")
    ap.add_argument("--show-reach", action="store_true", help="save outputs/reach.png (reachable zone) and exit")
    args = ap.parse_args()

    if args.save_home:
        arm = Arm(args.port, args.robot_id)
        arm.connect()
        arm.robot.bus.disable_torque()
        input("Torque is off. Move the arm to a folded pose OUT of the overhead camera's view, then Enter... ")
        q = arm.joints()
        save_home(q)
        print(f"Saved home: {np.round(q, 1).tolist()}")
        arm.disconnect()
        return

    if args.save_clear:
        arm = Arm(args.port, args.robot_id)
        arm.connect()
        print("Moving home first...")
        arm.move(load_home())
        arm.robot.bus.disable_torque()
        input("Torque is off. Lift ONLY the wrist until the gripper clears the robot's base, then Enter... ")
        q = arm.joints()
        save_clear(q)
        print(f"Saved clear pose: {np.round(q, 1).tolist()} (moves out of and back to home now go through it)")
        arm.disconnect()
        return

    if args.show_reach:
        calib = load_calib(args.calib)
        path = output("reach.png")
        cv2.imwrite(str(path), shade_reach(capture(args.camera or calib["camera"]), calib))
        print(f"Saved {path}: put tools and the handover spot inside the green zone.")
        return

    if not args.prompt:
        sys.exit("--prompt is required")

    calib = load_calib(args.calib)
    camera = args.camera or calib["camera"]
    kin = Kinematics(args.urdf)

    arm = None
    if not args.dry_run:
        arm = Arm(args.port, args.robot_id, speed_dps=args.speed)
        arm.connect()
        print("Moving home...")
        arm.go_home()

    # 1. Look
    frame = capture(camera)
    dets, _ = segment_robust(frame, args.prompt)
    if not dets:
        if arm:
            arm.disconnect()
        sys.exit(f"No '{args.prompt}' found in the overhead view.")
    if len(dets) > 1:
        dets.sort(key=lambda d: (d.confidence or 0, d.mask.sum()), reverse=True)
        print(f"{len(dets)} matches; taking the most confident/largest (id {dets[0].object_id}).")
    det = dets[0]
    g = grasp_from_mask(det.mask)

    # 2. Where, in the arm's frame
    gx, gy, gz = pixel_to_base(g.u, g.v, calib)
    close_dir = pixel_dir_to_base(g.u, g.v, g.yaw_deg, calib) + args.yaw_offset
    if args.place_pixel:
        pu, pv = (float(s) for s in args.place_pixel.split(","))
        px, py, pz = pixel_to_base(pu, pv, calib)
    elif args.place_xy:
        px, py = (float(s) / 1000 for s in args.place_xy.split(","))
        pz = table_z_at(px, py, calib)
        pu = pv = None
    else:
        px, py, pz, pu, pv = gx, gy, gz, g.u, g.v

    gh, hov = args.grasp_height / 1000, args.approach / 1000  # above the local table surface
    plan = [
        ("above object", gx, gy, gz + hov, gz),
        ("grasp", gx, gy, gz + gh, gz),
        ("lift", gx, gy, gz + hov, gz),
        ("above place", px, py, pz + hov, pz),
        ("place", px, py, pz + gh + 0.003, pz),
        ("retreat", px, py, pz + hov, pz),
    ]
    print(f"Object '{args.prompt}' at pixel ({g.u:.0f},{g.v:.0f}) -> arm ({gx*1000:.0f}, {gy*1000:.0f}) mm, "
          f"reach {math.hypot(gx, gy)*1000:.0f} mm, jaws close at {close_dir:.0f} deg")
    print(f"Place at arm ({px*1000:.0f}, {py*1000:.0f}) mm, reach {math.hypot(px, py)*1000:.0f} mm")

    solved = []
    for name, x, y, z, tz in plan:
        r = kin.topdown(x, y, z, close_dir)
        reach = math.hypot(x, y)
        if reach < REACH_MIN:
            r.ok = False
        flag = "ok" if r.ok else ("TOO CLOSE TO BASE" if reach < REACH_MIN else "OUT OF REACH")
        print(f"  {name:12s} ({x*1000:4.0f},{y*1000:4.0f},{(z - tz)*1000:3.0f}mm up)  err {r.pos_err*1000:4.1f}mm "
              f"tilt {r.tilt_deg:4.1f}  roll {r.joints[4]:5.0f}  {flag}")
        solved.append((name, r))

    vis = draw_overlay(shade_reach(frame, calib), [det], [g])
    if pu is not None:
        cv2.drawMarker(vis, (int(pu), int(pv)), (0, 200, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.imwrite(str(output("pick_plan.png")), vis)
    print(f"Plan image: {output('pick_plan.png')}")

    if not all(r.ok for _, r in solved):
        if arm:
            arm.go_home()
            arm.disconnect()
        sys.exit(f"Part of the plan is out of reach. Keep objects and the place spot {REACH_MIN*100:.0f}-"
                 f"{REACH_MAX*100:.0f} cm from the arm base (green zone: sag-pick --show-reach).")
    if args.dry_run:
        print("Dry run: not moving.")
        return

    if not args.yes:
        input("Enter to execute (Ctrl+C to abort)... ")

    q = {name: r.joints for name, r in solved}
    try:
        arm.leave_home()
        arm.move(q["above object"], gripper=args.open)
        if args.pause:
            print(f"Hovering. The jaws should close ACROSS the red line in {output('pick_plan.png')}")
            print(f"(planned wrist_roll {q['grasp'][4]:.0f} deg). If they're along it, rerun with --yaw-offset 90.")
            if input("Enter to grasp, or type q to go home: ").strip().lower() == "q":
                arm.go_home()
                return
        arm.move(q["grasp"])
        arm.set_gripper(args.closed)
        time.sleep(0.3)
        arm.move(q["lift"])
        if args.place_pixel or args.place_xy:
            arm.move(q["above place"])
        else:
            time.sleep(1.0)
        arm.move(q["place"])
        arm.set_gripper(args.open)
        arm.move(q["retreat"])
        arm.go_home(gripper=args.closed)
        print("Done.")
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
