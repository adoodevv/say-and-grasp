"""Kojo's body: save its awake pose and try the gestures.

    sag-kojo --save-awake        # pose the arm by hand: up, "looking" at you, out of the camera's view
    sag-kojo wake wave nod dance sleep      # play gestures in order
    sag-kojo all                 # the whole show

Gestures: wake, wave, nod, shake_head, tilt_head, dance, sleep.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from ..arm import Arm, load_clear, load_home
from ..character import GESTURES, Character, save_awake
from ..paths import output
from ..skills import Bench
from ..voice import speak


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("gestures", nargs="*", help=f"any of: {', '.join(GESTURES)}, or 'all'")
    ap.add_argument("--save-awake", action="store_true", help="record Kojo's awake pose by hand")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--quiet", action="store_true", help="don't speak")
    args = ap.parse_args()

    if args.save_awake:
        arm = Arm(args.port, args.robot_id)
        arm.connect()
        print("Moving home first...")
        arm.move(load_home())
        clear = load_clear()
        if clear is not None:
            arm.move(clear, settle_s=0.1)
        arm.robot.bus.disable_torque()
        input("Torque is off. Lift the arm into Kojo's 'awake' pose: up off the base, the gripper\n"
              "(its head) looking towards where you sit. Keep it OUT of the overhead camera's view.\n"
              "Hold it there and press Enter... ")
        q = arm.joints()
        arm.robot.bus.enable_torque()
        arm.q_cmd = q
        arm.corr[:] = 0
        arm.move(q, settle_s=0.5)
        save_awake(q)
        print(f"Saved awake pose: {np.round(q, 1).tolist()}")
        try:
            from ..calibration import load_calib
            from ..camera import capture

            path = output("awake_view.png")
            cv2.imwrite(str(path), capture(load_calib()["camera"]))
            print(f"Overhead photo in this pose: {path}. The arm shouldn't cover the bench in it.")
        except Exception as e:
            print(f"(couldn't take the overhead photo: {e})")
        if clear is not None:
            arm.move(clear, settle_s=0.1)
        arm.move(load_home(), settle_s=0.3)
        arm.disconnect()
        return

    names = args.gestures or ["wake", "wave", "sleep"]
    if names == ["all"]:
        names = ["wake", "wave", "nod", "shake_head", "tilt_head", "dance", "sleep"]
    bench = Bench(port=args.port, robot_id=args.robot_id)
    bench.arm.connect()
    bench.arm.go_home()
    kojo = Character(bench, say=(lambda t: print(f"  Kojo: {t}")) if args.quiet else speak)
    try:
        for n in names:
            print(f"[{n}]")
            if n == "wake":
                kojo.wake()
            else:
                kojo.gesture(n)
    except KeyboardInterrupt:
        pass
    finally:
        if kojo.awake:
            kojo.sleep(goodnight=False)
        bench.go_home()
        bench.arm.disconnect()


if __name__ == "__main__":
    main()
