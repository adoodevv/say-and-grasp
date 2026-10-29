"""Talk to the workbench assistant.

    sag-agent                 # push-to-talk: Enter, speak, Enter
    sag-agent --text          # type requests instead (no microphone needed)
    sag-agent --text --dry-run   # plan only, the arm doesn't move

Needs a handover spot first:  sag-spots set handover U,V
"""
from __future__ import annotations

import argparse

from ..agent import Assistant
from ..skills import Bench
from ..voice import listen, speak, transcribe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="store_true", help="type instead of speaking")
    ap.add_argument("--dry-run", action="store_true", help="don't move the arm")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--open", type=float, default=60)
    ap.add_argument("--closed", type=float, default=5)
    ap.add_argument("--grasp-height", type=float, default=5)
    ap.add_argument("--effort", default="minimal", help="Muse Spark reasoning effort: minimal/low/medium/high")
    args = ap.parse_args()

    bench = Bench(port=args.port, robot_id=args.robot_id, camera=args.camera, dry_run=args.dry_run,
                  gripper_open=args.open, gripper_closed=args.closed, grasp_height_mm=args.grasp_height)
    bot = Assistant(bench, reasoning_effort=args.effort, say=speak)
    bench.connect()
    print("Say-and-Grasp ready. Ctrl+C to quit.")
    try:
        while True:
            if args.text:
                text = input("\nyou> ").strip()
            else:
                wav = listen("\nEnter to talk... ")
                text = transcribe(wav, keywords=["side cutters", "screwdriver", "multimeter", "pliers"])
                print(f"you> {text}")
            if not text:
                continue
            speak(bot.handle(text))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        bench.go_home()
        bench.close()


if __name__ == "__main__":
    main()
