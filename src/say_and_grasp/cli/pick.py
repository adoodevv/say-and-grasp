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
Every grasp is checked (did the jaws close on something?) and retried with a fresh look.
Writes outputs/look.png: detection, grasp line and the reachable zone.
"""
from __future__ import annotations

import argparse
import math
import sys
import time

import cv2
import numpy as np

from ..arm import Arm, load_home, save_clear, save_home
from ..calibration import load_calib, shade_reach
from ..camera import capture
from ..paths import CALIB_FILE, URDF_FILE, output
from ..skills import Bench, SkillError


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
    ap.add_argument("--retries", type=int, default=2, help="re-look and retry this many times after a missed grasp")
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

    bench = Bench(
        port=args.port, robot_id=args.robot_id, camera=args.camera, calib_path=args.calib, urdf=args.urdf,
        speed=args.speed, grasp_height_mm=args.grasp_height, approach_mm=args.approach,
        gripper_open=args.open, gripper_closed=args.closed, yaw_offset=args.yaw_offset, dry_run=args.dry_run,
    )
    try:
        bench.connect()
        if not args.yes and not args.dry_run:
            input("Enter to start (Ctrl+C to abort)... ")
        seen = bench.pick(args.prompt, retries=args.retries, pause=args.pause)
        if args.place_pixel:
            u, v = (float(s) for s in args.place_pixel.split(","))
            bench.place_pixel(u, v)
        elif args.place_xy:
            x, y = (float(s) / 1000 for s in args.place_xy.split(","))
            bench.place_xy(x, y)
        else:
            time.sleep(1.0)
            bench.put_back()
        bench.go_home()
        print("Done." if not args.dry_run else f"Dry run OK (see {output('look.png')}).")
    except SkillError as e:
        print(f"Couldn't do it: {e}")
        bench.go_home()
    finally:
        bench.close()


if __name__ == "__main__":
    main()
