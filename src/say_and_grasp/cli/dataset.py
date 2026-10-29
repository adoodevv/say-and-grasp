"""The robot's self-collected dataset: what's in it, and sharing it.

    sag-dataset                 # episodes, success rate, per-object breakdown
    sag-dataset --push          # upload to the Hugging Face Hub (needs `huggingface-cli login`)

Train on the successful episodes only, e.g.:
    lerobot-train --dataset.repo_id=adoodevv/say-and-grasp-handovers --dataset.root=data/say-and-grasp-handovers \\
        --dataset.episodes="[<printed list>]" --policy.type=act ...
"""
from __future__ import annotations

import argparse
from collections import Counter

from ..recorder import DEFAULT_REPO_ID, dataset_root, read_labels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    ap.add_argument("--push", action="store_true", help="push the dataset to the Hugging Face Hub")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    labels = read_labels(args.repo_id)
    root = dataset_root(args.repo_id)
    if not labels:
        print(f"No episodes yet in {root}. Run sag-agent and fetch some things.")
        return
    ok = [l for l in labels if l["success"]]
    secs = sum(l.get("duration_s", 0) for l in labels)
    frames = sum(l.get("frames", 0) for l in labels)
    print(f"Dataset {args.repo_id}  ({root})")
    print(f"  episodes: {len(labels)}   successful: {len(ok)} ({100 * len(ok) / len(labels):.0f}%)")
    print(f"  recorded: {secs / 60:.1f} min, {frames} frames, zero human demonstrations")
    by_obj = Counter(l.get("object", "?") for l in labels)
    by_obj_ok = Counter(l.get("object", "?") for l in ok)
    print("  per object:")
    for obj, n in by_obj.most_common():
        print(f"    {obj:20s} {by_obj_ok[obj]}/{n}")
    print(f"  successful episode indices: {[l['episode_index'] for l in ok]}")
    fails = [l for l in labels if not l["success"]]
    if fails:
        print("  recent failures:")
        for l in fails[-5:]:
            print(f"    #{l['episode_index']} {l.get('object','?')}: {l.get('verifier') or l.get('error')}")

    if args.push:
        push(args.repo_id, private=args.private)


def push(repo_id: str = DEFAULT_REPO_ID, private: bool = False) -> None:
    """Upload the dataset (videos, data, metadata and Muse Spark's labels.jsonl) to the Hub."""
    from huggingface_hub import whoami
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        user = whoami()["name"]
    except Exception:
        print("Not logged in to Hugging Face. Run: huggingface-cli login   (or: hf auth login)")
        return
    if repo_id.split("/")[0] != user:
        print(f"Note: you're logged in as '{user}', pushing to '{repo_id}'.")
    root = dataset_root(repo_id)
    if not (root / "meta" / "info.json").exists():
        print(f"No dataset at {root} yet.")
        return
    print(f"Pushing {root} to the Hub...")
    ds = LeRobotDataset(repo_id, root=root)
    ds.push_to_hub(private=private, tags=["say-and-grasp", "so101", "zero-shot", "meta-hackathon", "muse-spark", "sam3"])
    print(f"Done: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
