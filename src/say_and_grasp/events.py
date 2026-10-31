"""A tiny event bus: the robot's skills and the assistant announce what they're doing, and anything
interested (the brain-view recorder) listens. Nothing listens by default, so this costs nothing.

    events.publish("overhead_sam", image=overlay, label="SAM 3.1: 'screwdriver'")
    events.subscribe(callback)   # callback(kind, data_dict)

Kinds used:
  heard          text                         what Muse Voice Transcribe heard
  tool           name, args                   a tool Muse Spark called
  result         name, result                 what the tool returned
  say            text                         what Wiggle says out loud
  overhead_sam   image, label                 an overhead photo with SAM 3.1 masks drawn on it
  wrist_sam      image, label                 a wrist photo with SAM 3.1 masks and the jaw centre
  verify         ok, reason                   Muse Spark's verdict on a handover
  status         text                         a short status line ("aligning", "dropping into your hand")
"""
from __future__ import annotations

import threading

_subs: list = []
_lock = threading.Lock()


def subscribe(cb) -> None:
    with _lock:
        _subs.append(cb)


def unsubscribe(cb) -> None:
    with _lock:
        if cb in _subs:
            _subs.remove(cb)


def publish(kind: str, **data) -> None:
    with _lock:
        subs = list(_subs)
    for cb in subs:
        try:
            cb(kind, data)
        except Exception:
            pass  # a viewer must never break the robot
