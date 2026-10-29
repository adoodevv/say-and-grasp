"""The workbench assistant: Muse Spark decides, the robot skills act.

Each request goes to Muse Spark together with a fresh overhead photo of the bench, so it can
resolve vague words ("the thing for cutting wire") to what is actually there, and turn them
into short names SAM 3.1 can find. It then calls tools (find, pick_up, hand_over, ...) until
the job is done and answers in one short spoken sentence.
"""
from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sam import get_client
from .skills import Bench, SkillError
from .spots import load_spots, spot_xy

VERIFY_PROMPT = """This is an overhead photo of a robot's workbench. The green circle marks the handover spot.
Did the robot successfully put a {obj} down inside (or touching) the green circle?
Answer with JSON only: {{"success": true or false, "reason": "<at most 12 words>"}}"""

MODEL = "muse-spark-1.3"

SYSTEM = """You are Say-and-Grasp, a robot arm on the user's electronics workbench: a third hand.
The user is usually busy (soldering, wiring) and talks to you by voice.

You get a fresh overhead photo of the bench with every request. Use it to see what is there.

Tools move a real arm, so:
- Name objects for the vision model with a SHORT concrete phrase: colour + noun when it helps,
  e.g. "side cutters", "red screwdriver", "blue block", "solder spool". Never pass a description
  of purpose ("thing for cutting wire") to a tool; map it to the object you see in the photo.
- To give the user something, call fetch(object): it picks it up and hands it over in one go.
  Use pick_up/hand_over/put_down/put_back only for other jobs (tidying, moving things around).
- If the object isn't on the bench or out of reach, say so; don't guess.
- One thing at a time; you can only hold one object.

Reply in ONE short, friendly spoken sentence (no lists, no markdown). Don't narrate tool calls."""

TOOLS = [
    {"type": "function", "function": {
        "name": "fetch",
        "description": "Pick up an object and hand it to the user (put it down at the handover spot), "
                       "then fold out of the way. The normal way to answer 'pass me / give me X'.",
        "parameters": {"type": "object", "properties": {
            "object": {"type": "string", "description": "short noun phrase, e.g. 'side cutters'"}},
            "required": ["object"]}}},
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
    reasoning_effort: str = "minimal"
    history: list[dict] = field(default_factory=list)
    max_steps: int = 8
    log: callable = print
    say: callable = None  # speak a short line straight away (e.g. "On it") while the arm works
    recorder: object = None  # say_and_grasp.recorder.EpisodeRecorder: every fetch becomes an episode
    verify: bool = True  # ask Muse Spark whether each handover really worked (labels the episode)

    # ------------------------------------------------------------------ verification
    def verify_handover(self, obj: str) -> tuple[bool, str]:
        """Muse Spark looks at the overhead view and judges whether `obj` arrived at the handover spot."""
        from .spots import load_spots

        spots = load_spots()
        img = self.bench.snapshot()
        if "handover" in spots:
            u, v = (int(t) for t in spots["handover"])
            cv2.circle(img, (u, v), 45, (0, 220, 0), 3)
        resp = get_client().chat.completions.create(
            model=self.model, reasoning_effort="minimal",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": VERIFY_PROMPT.format(obj=obj)},
                {"type": "image_url", "image_url": {"url": _jpeg_data_url(img)}},
            ]}],
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            verdict = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            verdict = {}
        ok = bool(verdict.get("success", False))
        reason = str(verdict.get("reason", text.strip()[:80]))
        self.log(f"  [verify] {'success' if ok else 'FAILED'}: {reason}")
        return ok, reason

    # ------------------------------------------------------------------ tools
    def _tool(self, name: str, args: dict) -> dict:
        b = self.bench
        try:
            if name == "find":
                s = b.look(args["object"])
                return {"found": True, "position_cm": [round(s.x * 100, 1), round(s.y * 100, 1)],
                        "reachable": 0.16 <= s.reach <= 0.28}
            if name == "fetch":
                return self._fetch(args["object"])
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

    def _fetch(self, obj: str) -> dict:
        """Pick + hand over + home, recorded as one episode and checked by Muse Spark."""
        b = self.bench
        rec = self.recorder
        if rec:
            rec.start(f"pick up the {obj} and hand it over")
        t0 = time.time()
        try:
            s = b.pick(obj)
            x, y = spot_xy("handover", b.calib)
            b.place_xy(x, y)
            b.go_home()
        except (SkillError, KeyError) as e:
            if rec:
                rec.stop(False, {"object": obj, "error": str(e)})
            raise
        ok, reason = (True, "not verified")
        if self.verify and not b.dry_run:
            try:
                ok, reason = self.verify_handover(obj)
            except Exception as e:  # a verifier hiccup shouldn't break the handover
                ok, reason = True, f"verifier error: {e.__class__.__name__}"
        if rec:
            rec.stop(ok, {"object": obj, "verifier": reason, "seconds": round(time.time() - t0, 1)})
        # Report honestly but don't fail the tool: a failed check must not make the model re-run
        # the whole fetch (that would pick the object back up from next to the user).
        return {"ok": True, "handed_over": s.prompt, "verified": ok, **({} if ok else {"note": reason})}

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
            # Muse Spark often says what it's about to do alongside the tool call ("I'll grab the
            # blue block"). Say that NOW, as the arm starts moving, not after it's finished.
            if msg.content and msg.content.strip() and self.say:
                self.say(msg.content.strip())
            results = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                self.log(f"  [tool] {tc.function.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                result = self._tool(tc.function.name, args)
                self.log(f"         -> {result}")
                results.append((tc.function.name, args, result))
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            # A successful fetch is the whole job: answer right away instead of a second model call.
            if len(results) == 1 and results[0][0] == "fetch" and results[0][2].get("ok"):
                obj = results[0][1]["object"]
                if results[0][2].get("verified", True):
                    reply = f"Here you go, your {obj}."
                else:
                    reply = f"I've put the {obj} down, but I'm not sure it landed in the right spot."
                break
        else:
            reply = "Sorry, that took too many steps; I've stopped."
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-12:]
        self.log(f"  [{time.time() - t0:.1f}s]")
        return reply
