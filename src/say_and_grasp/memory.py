"""Wiggle's memory of the bench: where each thing was last seen, and what happened to it.

Every look, pick, handover and put-down is noted, so "where's my multimeter?" has an answer even
when the multimeter isn't in view any more ("I handed it to you 5 minutes ago").
Kept in data/sightings.json so it survives a restart.
"""
from __future__ import annotations

import json
import time

from .paths import ROOT

MEMORY_FILE = ROOT / "data" / "sightings.json"
_FILLER = {"the", "a", "an", "my", "of", "small", "little", "big", "pair"}


def _words(name: str) -> set[str]:
    return {w.rstrip("s") for w in name.lower().replace("-", " ").split() if w not in _FILLER}


def ago(t: float) -> str:
    s = time.time() - t
    if s < 60:
        return "just now"
    if s < 3600:
        m = round(s / 60)
        return f"{m} minute{'s' if m != 1 else ''} ago"
    h = round(s / 3600)
    if h < 24:
        return f"{h} hour{'s' if h != 1 else ''} ago"
    d = round(s / 86400)
    return f"{d} day{'s' if d != 1 else ''} ago"


class Memory:
    def __init__(self, path=MEMORY_FILE):
        self.path = path
        try:
            self.items: dict[str, dict] = json.loads(path.read_text())
        except Exception:
            self.items = {}

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.items, indent=1))
        except OSError:
            pass

    def note(self, obj: str, event: str, x: float | None = None, y: float | None = None) -> None:
        """event: 'seen on the bench', 'picked up', 'handed to the user', 'put in the tray', ..."""
        key = self.match(obj) or obj.lower().strip()
        rec = {"name": obj, "event": event, "t": time.time()}
        if x is not None:
            rec["xy"] = [round(x, 3), round(y, 3)]
        elif key in self.items and "xy" in self.items[key]:
            rec["xy"] = self.items[key]["xy"]
        self.items[key] = rec
        self._save()

    def match(self, obj: str) -> str | None:
        """Best stored name for `obj` ("multimeter" finds "yellow multimeter")."""
        want = _words(obj)
        best, score = None, 0.0
        for k in self.items:
            have = _words(k)
            if not want or not have:
                continue
            s = len(want & have) / len(want | have)
            if s > score:
                best, score = k, s
        return best if score >= 0.5 or (best and _words(best) >= want) else None

    def recall(self, obj: str) -> dict | None:
        k = self.match(obj)
        if k is None:
            return None
        r = dict(self.items[k])
        r["when"] = ago(r["t"])
        return r

    def summary(self, n: int = 8) -> str:
        """A few recent notes for Muse Spark, newest first."""
        recent = sorted(self.items.values(), key=lambda r: -r["t"])[:n]
        return "; ".join(f"{r['name']}: {r['event']} {ago(r['t'])}" for r in recent) or "nothing yet"
