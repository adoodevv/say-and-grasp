"""Close the self-training loop: train a policy on the episodes Wiggle recorded itself, then run it.

    sag-train --task tray                  # show what it would train on (successful 'tray' episodes)
    sag-train --task tray --go             # train ACT on HF Jobs (A10G GPU, billed to your HF account)
    sag-policy --task "put the screwdriver in the tray"   # run the trained policy on the arm

Only episodes Muse Spark (or the handover) marked as successful are used, so the policy learns
from Wiggle's good runs. No human demonstrations anywhere. Push the dataset first:
    sag-dataset --push

ACT is a small policy that learns one task at a time without language, so train it on one kind of
job (e.g. putting tools in the tray); the more consistent the episodes, the better it learns.
"""
from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys

from ..recorder import DEFAULT_REPO_ID, FPS, read_labels


def _hf_user() -> str:
    from huggingface_hub import whoami

    try:
        return whoami()["name"]
    except Exception:
        sys.exit("Not logged in to Hugging Face: run `hf auth login` (a token with write access).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="tray",
                    help="train on successful episodes whose task contains this text (default: tray)")
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID, help="dataset on the Hub")
    ap.add_argument("--policy", default="act", help="LeRobot policy type (act is quick; smolvla is bigger)")
    ap.add_argument("--gpu", default="a10g-small", help="HF Jobs flavor (hf jobs hardware lists them)")
    ap.add_argument("--epochs", type=float, default=8, help="passes over the episodes")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--timeout", default="4h", help="stop the job after this long, whatever happens")
    ap.add_argument("--min-episodes", type=int, default=20)
    ap.add_argument("--go", action="store_true", help="actually submit the job (otherwise just print the plan)")
    ap.add_argument("--local", action="store_true", help="train on this machine instead (needs a GPU)")
    args = ap.parse_args()

    labels = read_labels(args.repo_id)
    if not labels:
        sys.exit("No episodes recorded yet (run sag-agent without --no-record).")
    want = args.task.lower()
    chosen = [r for r in labels if r.get("success") and want in r.get("task", "").lower()]
    other = [r for r in labels if want in r.get("task", "").lower() and not r.get("success")]
    episodes = sorted(int(r["episode_index"]) for r in chosen)
    frames = sum(int(r.get("frames", 0)) for r in chosen)
    tasks = sorted({r["task"] for r in chosen})
    print(f"Training set: {len(episodes)} successful '{args.task}' episodes "
          f"({frames} frames, {frames / FPS / 60:.1f} min); {len(other)} failed ones left out.")
    for t in tasks:
        print(f"  - {t}")
    if len(episodes) < args.min_episodes:
        print(f"\nThat's under {args.min_episodes} episodes; ACT usually needs ~40-50 consistent ones. "
              "Record more first (or --min-episodes to override).")
        if args.go:
            return

    steps = max(2000, math.ceil(args.epochs * frames / args.batch))
    user = _hf_user() if (args.go or not args.local) else "you"
    name = f"wiggle_{args.policy}_{want.replace(' ', '_')}"
    cmd = [
        "lerobot-train",
        f"--policy.type={args.policy}",
        f"--dataset.repo_id={args.repo_id}",
        f"--dataset.episodes={json.dumps(episodes, separators=(',', ':'))}",
        f"--policy.repo_id={user}/{name}",
        "--policy.push_to_hub=true",
        f"--batch_size={args.batch}",
        f"--steps={steps}",
        f"--save_freq={steps}",
        f"--job_name={name}",
        "--wandb.enable=false",
    ]
    if args.local:
        cmd.append("--policy.device=cuda")
    else:
        cmd += [f"--job.target={args.gpu}", f"--job.timeout={args.timeout}", "--policy.device=cuda"]
    print(f"\n{steps} steps (~{args.epochs:g} epochs, batch {args.batch}) "
          + ("on this machine" if args.local else f"on HF Jobs ({args.gpu}, stops after {args.timeout} at most)"))
    print("Trained policy goes to " + f"https://huggingface.co/{user}/{name}\n")
    print(" ".join(shlex.quote(c) for c in cmd))
    if not args.go:
        print("\nAdd --go to start it.")
        return
    if not shutil.which("lerobot-train"):
        sys.exit("lerobot-train not found: activate the lerobot environment.")
    sys.exit(subprocess.call(cmd))


def policy_main() -> None:
    """sag-policy: run the trained policy on the arm with lerobot-rollout."""
    from ..calibration import load_calib
    from ..paths import WRIST_FILE

    ap = argparse.ArgumentParser(description="Run a policy Wiggle trained on its own episodes.")
    ap.add_argument("--policy", default=None, help="Hub id or local path (default: <you>/wiggle_act_tray)")
    ap.add_argument("--task", default="put the blue block in the tray", help="what to do (logged with the run)")
    ap.add_argument("--seconds", type=float, default=40)
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="my_follower")
    ap.add_argument("--overhead", default=None, help="overhead camera (default: from calib.json)")
    ap.add_argument("--wrist", default=None, help="wrist camera (default: from wrist.json)")
    ap.add_argument("--go", action="store_true", help="run it (otherwise print the command)")
    args = ap.parse_args()

    policy = args.policy or f"{_hf_user()}/wiggle_act_tray"
    over = args.overhead or str(load_calib()["camera"])
    wrist = args.wrist or (str(json.loads(WRIST_FILE.read_text())["camera"]) if WRIST_FILE.exists() else "2")

    def cam(c: str) -> str:
        ref = int(c) if c.isdigit() else json.dumps(c)
        return f"{{type: opencv, index_or_path: {ref}, width: 640, height: 480, fps: 30}}"

    cmd = [
        "lerobot-rollout", "--strategy.type=base",
        f"--policy.path={policy}",
        "--robot.type=so101_follower", f"--robot.port={args.port}", f"--robot.id={args.robot_id}",
        f"--robot.cameras={{ overhead: {cam(over)}, wrist: {cam(wrist)} }}",
        "--robot.max_relative_target=10",  # safety: no sudden jumps
        f"--fps={FPS}",  # the rate the episodes were recorded at
        f"--task={args.task}", f"--duration={args.seconds:g}",
    ]
    print(" ".join(shlex.quote(c) for c in cmd))
    if not args.go:
        print("\nAdd --go to run it. Keep a hand near the arm's power switch the first time.")
        return
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
