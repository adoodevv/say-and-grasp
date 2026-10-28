"""The workbench assistant: Muse Spark decides, the robot skills act.

Each request goes to Muse Spark together with a fresh overhead photo of the bench, so it can
resolve vague words ("the thing for cutting wire") to what is actually there, and turn them
into short names SAM 3.1 can find. It then calls tools (find, pick_up, hand_over, ...) until
the job is done and answers in one short spoken sentence.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sam import get_client
from .skills import Bench, SkillError
from .spots import load_spots, spot_xy

MODEL = "muse-spark-1.3"

SYSTEM = """You are Say-and-Grasp, a robot arm on the user's electronics workbench: a third hand.
The user is usually busy (soldering, wiring) and talks to you by voice.

You get a fresh overhead photo of the bench with every request. Use it to see what is there.

Tools move a real arm, so:
- Name objects for the vision model with a SHORT concrete phrase: colour + noun when it helps,
  e.g. "side cutters", "red screwdriver", "blue block", "solder spool". Never pass a description
  of purpose ("thing for cutting wire") to a tool; map it to the object you see in the photo.
- To give the user something: pick_up(object) then hand_over(). To tidy: put_back() after use, or
  put_down(spot).
- If the object isn't on the bench or out of reach, say so; don't guess.
- One thing at a time; you can only hold one object.

Reply in ONE short, friendly spoken sentence (no lists, no markdown). Don't narrate tool calls."""

TOOLS = [
    {"type": "function", "function": {
        "name": "find",
        "description": "Look for an object on the bench with the overhead camera. Returns whether it was found, "
                       "where (cm from the arm base), and whether the arm can reach it.",
        "parameters": {"type": "object", "properties": {
            "object": {"type": "string", "description": "short noun phrase, e.g. 'side cutters'"}},
            "required": ["object"]}}},
    {"type": "function", "function": {
        "name": "pick_up",
        "description": "Find and grasp an object. Checks the grasp and retries automatically.",
        "parameters": {"type": "object", "properties": {
            "object": {"type": "string", "description": "short noun phrase, e.g. 'side cutters'"}},
            "required": ["object"]}}},
    {"type": "function", "function": {
        "name": "hand_over",
        "description": "Put the held object down at the handover spot next to the user, then step back.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "put_down",
        "description": "Put the held object down at a named spot on the bench.",
        "parameters": {"type": "object", "properties": {
            "spot": {"type": "string", "description": "a saved spot name"}}, "required": ["spot"]}}},
    {"type": "function", "function": {
        "name": "put_back",
        "description": "Put the held object back exactly where it was picked up from.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "go_home",
        "description": "Fold the arm back to its rest pose, out of the way.",
        "parameters": {"type": "object", "properties": {}}}},
]


def _jpeg_data_url(img: np.ndarray, max_side: int = 640) -> str:
    h, w = img.shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


@dataclass
class Assistant:
    bench: Bench
    model: str = MODEL
    reasoning_effort: str = "low"
    history: list[dict] = field(default_factory=list)
    max_steps: int = 8
    log: callable = print

    # ------------------------------------------------------------------ tools
    def _tool(self, name: str, args: dict) -> dict:
        b = self.bench
        try:
            if name == "find":
                s = b.look(args["object"])
                return {"found": True, "position_cm": [round(s.x * 100, 1), round(s.y * 100, 1)],
                        "reachable": 0.16 <= s.reach <= 0.28}
            if name == "pick_up":
                s = b.pick(args["object"])
                return {"ok": True, "holding": s.prompt}
            if name == "hand_over":
                x, y = spot_xy("handover", b.calib)
                b.place_xy(x, y)
                b.go_home()
                return {"ok": True}
            if name == "put_down":
                x, y = spot_xy(args["spot"], b.calib)
                b.place_xy(x, y)
                b.go_home()
                return {"ok": True}
            if name == "put_back":
                b.put_back()
                b.go_home()
                return {"ok": True}
            if name == "go_home":
                b.go_home()
                return {"ok": True}
            return {"error": f"unknown tool {name}"}
        except (SkillError, KeyError) as e:
            b.go_home()
            return {"ok": False, "error": str(e).strip("'\"")}

    # ------------------------------------------------------------------ one request
    def handle(self, text: str) -> str:
        t0 = time.time()
        bench_photo = self.bench.snapshot()
        spots = ", ".join(load_spots()) or "none saved"
        holding = self.bench.holding.prompt if self.bench.holding else "nothing"
        user = {"role": "user", "content": [
            {"type": "text", "text": f"{text}\n\n(state: holding {holding}; saved spots: {spots})"},
            {"type": "image_url", "image_url": {"url": _jpeg_data_url(bench_photo)}},
        ]}
        # Keep earlier turns as text only (photos are big and stale).
        messages = [{"role": "system", "content": SYSTEM}, *self.history, user]
        client = get_client()
        reply = ""
        for _ in range(self.max_steps):
            resp = client.chat.completions.create(
                model=self.model, messages=messages, tools=TOOLS, tool_choice="auto",
                reasoning_effort=self.reasoning_effort,
            )
            msg = resp.choices[0].message
            if not msg.tool_calls:
                reply = (msg.content or "").strip()
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                self.log(f"  [tool] {tc.function.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                result = self._tool(tc.function.name, args)
                self.log(f"         -> {result}")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
        else:
            reply = "Sorry, that took too many steps; I've stopped."
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-12:]
        self.log(f"  [{time.time() - t0:.1f}s]")
        return reply
