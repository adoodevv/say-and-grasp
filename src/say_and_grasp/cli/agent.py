"""Talk to Kojo, the workbench assistant.

    sag-agent --hands-free    # just talk: say "Kojo" to wake it, then ask
    sag-agent                 # push-to-talk: Enter, speak, Enter
    sag-agent --text          # type requests instead (no microphone needed)
    sag-agent --text --dry-run   # plan only, the arm doesn't move
    sag-agent --no-record        # don't save episodes to the dataset
    sag-agent --push             # upload the dataset to the Hugging Face Hub when you quit

Kojo sleeps folded up until you call its name ("Kojo!", "hey Kojo, pass me the pliers"), then
wakes up and stays awake, looking at you, until it's been idle a while (--sleep-after) or you say
goodnight. In hands-free mode it only listens for its name while asleep.

Every fetch is recorded as a LeRobot episode in data/ and labelled by Muse Spark (sag-dataset).

Needs a handover spot first:  sag-spots set handover U,V
"""
from __future__ import annotations

import argparse
import time

from ..agent import Assistant
from ..character import Character, heard_name
from ..skills import Bench
from ..voice import listen, listen_auto, speak, transcribe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", action="store_true", help="type instead of speaking")
    ap.add_argument("--hands-free", action="store_true", help="no Enter key: say 'Kojo' to wake it")
    ap.add_argument("--sleep-after", type=float, default=60, help="seconds idle before Kojo naps (hands-free)")
    ap.add_argument("--dry-run", action="store_true", help="don't move the arm")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--camera", default=None)
    ap.add_argument("--open", type=float, default=60)
    ap.add_argument("--closed", type=float, default=5)
    ap.add_argument("--grasp-height", type=float, default=5)
    ap.add_argument("--effort", default="minimal", help="Muse Spark reasoning effort: minimal/low/medium/high")
    ap.add_argument("--no-record", action="store_true", help="don't record episodes")
    ap.add_argument("--no-hand", action="store_true",
                    help="don't hand tools into your hand; always put them on the handover spot")
    ap.add_argument("--no-verify", action="store_true", help="skip Muse Spark's check after each handover")
    ap.add_argument("--repo-id", default=None, help="dataset id (default adoodevv/say-and-grasp-handovers)")
    ap.add_argument("--push", action="store_true", help="push the dataset to the Hugging Face Hub on exit")
    ap.add_argument("--private", action="store_true", help="with --push: make the Hub dataset private")
    args = ap.parse_args()

    record = not args.no_record and not args.dry_run
    bench = Bench(port=args.port, robot_id=args.robot_id, camera=args.camera, dry_run=args.dry_run,
                  gripper_open=args.open, gripper_closed=args.closed, grasp_height_mm=args.grasp_height,
                  stream_overhead=record)
    bench.connect()
    recorder = None
    if record:
        from ..recorder import DEFAULT_REPO_ID, EpisodeRecorder

        prompt = "you> " if args.text else "Enter to talk... "

        def bg_log(msg: str) -> None:
            # Background saves finish while you're at the prompt: print above it, then redraw it.
            print(f"\r\033[K{msg}\n{prompt}", end="", flush=True)

        recorder = EpisodeRecorder(bench.arm, bench.overhead_peek, bench.wrist_peek,
                                   repo_id=args.repo_id or DEFAULT_REPO_ID, log=bg_log)
        print(f"Recording episodes to {recorder.root}")
    kojo = Character(bench, say=speak, sleep_after_s=args.sleep_after)
    bot = Assistant(bench, reasoning_effort=args.effort, say=speak, recorder=recorder,
                    verify=not args.no_verify, character=kojo, hand_to_hand=not args.no_hand)
    hands_free = args.hands_free and not args.text
    print("Kojo is asleep. Say its name to wake it" + (" (just talk)." if hands_free else ".")
          + " Ctrl+C to quit.")

    def hear() -> str | None:
        """Next thing the user said (None: nobody spoke before it was time to nap)."""
        if args.text:
            return input("\nyou> ").strip()
        if hands_free:
            timeout = max(1.0, kojo.sleep_after_s - (time.time() - kojo.last_active)) if kojo.awake else None
            print("\n  [listening" + ("" if kojo.awake else " for 'Kojo'") + "...]", flush=True)
            wav = listen_auto(timeout=timeout)
            if wav is None:
                return None
        else:
            wav = listen("\nEnter to talk... ")
        text = transcribe(wav)
        print(f"you> {text}")
        return text

    try:
        while True:
            text = hear()
            if text is None:  # idle too long
                kojo.sleep()
                continue
            if not text:
                continue
            called, rest = heard_name(text)
            if not called and hands_free and not kojo.awake:
                continue  # asleep: only its name wakes it
            if called and not rest:  # "Kojo!" -> wake up (or "still here")
                kojo.wake()
                continue
            if called and not kojo.awake:
                kojo.wake(quick=True, greet=False)  # "Kojo, pass me ..." -> perk up and get on with it
            kojo.touch()
            speak(bot.handle(rest if called else text))
            kojo.touch()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
    finally:
        if kojo.awake:
            try:
                kojo.sleep()
            except Exception:
                pass
        bench.go_home()
        if recorder:
            print("Finishing the last episode...")
            recorder.close()
        bench.close()
        if recorder and args.push:
            from .dataset import push

            push(args.repo_id or DEFAULT_REPO_ID, private=args.private)


if __name__ == "__main__":
    main()
