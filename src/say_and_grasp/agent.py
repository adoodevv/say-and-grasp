"""The workbench assistant: Muse Spark decides, the robot skills act.

Each request goes to Muse Spark together with a fresh overhead photo of the bench, so it can
resolve vague words ("the thing for cutting wire") to what is actually there, and turn them
into short names SAM 3.1 can find. It then calls tools (find, pick_up, hand_over, ...) until
the job is done and answers in one short spoken sentence.
"""
from __future__ import annotations

import base64
import json
import random
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .sam import get_client
from .skills import Bench, SkillError
from . import events
from .memory import Memory
from .spots import load_spots, spot_xy

VERIFY_PROMPT = """This is an overhead photo of a robot's workbench. The green circle marks the handover spot.
Did the robot successfully put a {obj} down inside (or touching) the green circle?
Answer with JSON only: {{"success": true or false, "reason": "<at most 12 words>"}}"""

MODEL = "muse-spark-1.3"

LOCATE_PROMPT = """This is an overhead photo of a workbench. Where is the {obj}?
Give the centre of the object as x and y from 0 to 1000 (x from the left edge, y from the top edge).
Answer with JSON only: {{"found": true, "x": <0-1000>, "y": <0-1000>}} or {{"found": false}}"""

TAKEN = ["All yours!", "There you go, one {obj}.", "Got it? Enjoy your {obj}.", "Careful, it's all yours now."]
HANDED = ["Here you go, your {obj}.", "One {obj}, delivered!", "There you go, {obj} right by your hand.",
          "Got it, your {obj} is ready."]

SYSTEM = """You are Wiggle, a small, cheerful robot arm on the user's electronics workbench: their third hand.
The user is usually busy (soldering, wiring) and talks to you by voice.

Personality: cheerful, warm and a little cheeky, like a helpful lab buddy. Light humour is welcome,
but keep it short; the user's hands are full. Your gripper is your mouth and your wrist is your head.
You can wave, nod, shake_head, tilt_head (puzzled) and dance with the gesture tool: use one when
asked ("wave at me", "do a dance", "say hi") or when it fits naturally (a nod for "thank you",
a dance when the user celebrates). Never gesture while holding something. Say "sleep" with the
gesture tool when the user says goodnight / go to sleep / that's all.

You get a fresh overhead photo of the bench with every request. Use it to see what is there.

Tools move a real arm, so:
- Name objects for the vision model with a SHORT concrete phrase: ALWAYS colour + noun as you see
  it in the photo, e.g. "red side cutters", "orange screwdriver", "blue block", "black brush". Never pass a description
  of purpose ("thing for cutting wire") to a tool; map it to the object you see in the photo.
- To give the user something, call fetch(object): it picks it up and hands it over in one go.
  Use pick_up/hand_over/put_down/put_back only for other jobs (tidying, moving things around).
- If the object isn't on the bench or out of reach, say so; don't guess.
- One thing at a time; you can only hold one object.
- "Where's my X?" -> locate(X): it points at it if it's on the bench, or tells you what last
  happened to it (handed to the user, put in the tray...). Answer like a friend would: "Right
  there, next to the solder spool" or "I handed it to you about ten minutes ago".
- "Tidy up" -> call tidy_up ONCE with every loose tool you see in the photo (colour + noun each),
  in the order to do them (nearest the container last is fine). Leave out the container itself,
  anything already inside it, and things too big to grip (a multimeter). Use the container you
  see (tray, cup, bin, box). The robot does them all in one go; reply once when it's done.
- The state line lists what you remember about the bench; use it.

Reply in ONE short, friendly spoken sentence (no lists, no markdown, no emoji), and don't say your
own name. Anything you write alongside a tool call is spoken immediately, so keep it to two to four
words ("On it!", "Let me look.") or write nothing; never repeat it in your final reply."""

TOOLS = [
    {"type": "function", "function": {
        "name": "fetch",
        "description": "Pick up an object and hand it to the user: dropped gently into their open hand, "
                       "or onto the handover spot if no hand is held out. The normal way to answer "
                       "'pass me / give me X'.",
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
        "description": "Give the held object to the user: into their hand, else on the handover spot.",
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
        "name": "locate",
        "description": "Answer 'where is my X': looks for it, points at it with the gripper if it's on the "
                       "bench, else returns what you last did with it and when.",
        "parameters": {"type": "object", "properties": {
            "object": {"type": "string", "description": "short noun phrase, e.g. 'multimeter'"}},
            "required": ["object"]}}},
    {"type": "function", "function": {
        "name": "tidy_up",
        "description": "Put several loose objects into a container, one after another, without stopping.",
        "parameters": {"type": "object", "properties": {
            "objects": {"type": "array", "items": {"type": "string"},
                        "description": "short noun phrases, e.g. ['blue brush', 'black tweezers']"},
            "container": {"type": "string", "description": "what to put them in, e.g. 'green tray'"}},
            "required": ["objects", "container"]}}},
    {"type": "function", "function": {
        "name": "put_away",
        "description": "Pick up an object and drop it into a container on the bench (for tidying up).",
        "parameters": {"type": "object", "properties": {
            "object": {"type": "string", "description": "short noun phrase, e.g. 'side cutters'"},
            "container": {"type": "string", "description": "what to put it in, e.g. 'tray', 'cup'"}},
            "required": ["object", "container"]}}},
    {"type": "function", "function": {
        "name": "gesture",
        "description": "Body language: wave, nod, shake_head, tilt_head, dance, or sleep (fold up and nap).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": ["wave", "nod", "shake_head", "tilt_head", "dance", "sleep"]}},
            "required": ["name"]}}},
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
    max_steps: int = 14  # tidying up is one put_away per tool
    log: callable = print
    say: callable = None  # speak a short line straight away (e.g. "On it") while the arm works
    recorder: object = None  # say_and_grasp.recorder.EpisodeRecorder: every fetch becomes an episode
    verify: bool = True  # ask Muse Spark whether each handover really worked (labels the episode)
    character: object = None  # say_and_grasp.character.Character: Wiggle's gestures, wake and sleep
    hand_to_hand: bool = True  # hold the tool out to the user's hand; else put it on the handover spot
    memory: Memory = field(default_factory=Memory)  # where things were last seen / what happened to them

    def __post_init__(self):
        mem = self.memory
        self.bench.name_hint = lambda p: (mem.items[k]["name"] if (k := mem.match(p)) else None)
        self.bench.locator = self.locate_in_photo

    def locate_in_photo(self, frame: np.ndarray, obj: str) -> tuple[float, float] | None:
        """Ask Muse Spark roughly where `obj` is in an overhead photo -> pixel (u, v), or None."""
        h, w = frame.shape[:2]
        resp = get_client().chat.completions.create(
            model=self.model, reasoning_effort="low",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": LOCATE_PROMPT.format(obj=obj)},
                {"type": "image_url", "image_url": {"url": _jpeg_data_url(frame, max_side=1280)}},
            ]}],
        )
        text = resp.choices[0].message.content or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            d = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            d = {}
        if not d.get("found") or "x" not in d or "y" not in d:
            self.log(f"  [locate] Muse Spark doesn't see a {obj} either")
            return None
        return float(d["x"]) / 1000 * w, float(d["y"]) / 1000 * h

    def _give(self) -> str:
        """Hand over what we hold: into the user's hand if we see one, else onto the handover spot.
        Returns "hand" or "bench"."""
        b = self.bench
        spot = None
        try:
            spot = spot_xy("handover", b.calib)
        except KeyError:
            pass
        if not self.hand_to_hand:
            if spot is None:
                raise SkillError("No handover spot saved (sag-spots set handover U,V).")
            b.place_xy(*spot)
            return "bench"
        # Straight from the pick: it looks for your hand while still holding the tool.
        return b.offer(say=self.say or print, fallback_xy=spot)

    def _rest(self) -> None:
        """After a job: back to the awake pose (looking at you) if awake, else fold up."""
        if self.character:
            self.character.rest()
        else:
            self.bench.go_home()

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
                {"type": "image_url", "image_url": {"url": _jpeg_data_url(img, max_side=1280)}},
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
        events.publish("verify", ok=ok, reason=reason)
        return ok, reason

    # ------------------------------------------------------------------ tools
    def _tool(self, name: str, args: dict) -> dict:
        b = self.bench
        try:
            if name == "find":
                s = b.look(args["object"])
                self.memory.note(args["object"], "seen on the bench", s.x, s.y)
                return {"found": True, "position_cm": [round(s.x * 100, 1), round(s.y * 100, 1)],
                        "reachable": b.kin.grasp(s.x, s.y, s.z + b.gh, s.close_dir).ok}
            if name == "fetch":
                return self._fetch(args["object"])
            if name == "pick_up":
                s = b.pick(args["object"])
                self.memory.note(args["object"], "picked up (holding it)", s.x, s.y)
                return {"ok": True, "holding": s.prompt}
            if name == "locate":
                return self._locate(args["object"])
            if name == "put_away":
                return self._put_away(args["object"], args.get("container") or "tray")
            if name == "tidy_up":
                return self._tidy_up(list(args.get("objects") or []), args.get("container") or "tray")
            if name == "hand_over":
                held = b.holding.prompt if b.holding else "it"
                where = self._give()
                self.memory.note(held, "handed to the user" if where == "hand" else "put on the handover spot")
                self._rest()
                return {"ok": True, "handed": "into your hand" if where == "hand" else "on the bench"}
            if name == "put_down":
                x, y = spot_xy(args["spot"], b.calib)
                held = b.holding.prompt if b.holding else "it"
                b.place_xy(x, y)
                self.memory.note(held, f"put down at the {args['spot']} spot", x, y)
                self._rest()
                return {"ok": True}
            if name == "put_back":
                held = b.holding
                b.put_back()
                if held:
                    self.memory.note(held.prompt, "put back where it was", held.x, held.y)
                self._rest()
                return {"ok": True}
            if name == "gesture":
                if b.holding and args["name"] != "sleep":
                    return {"ok": False, "error": "can't gesture while holding something"}
                if self.character:
                    self.character.gesture(args["name"])
                return {"ok": True, "done": args["name"]}
            if name == "go_home":
                b.go_home()
                return {"ok": True}
            return {"error": f"unknown tool {name}"}
        except (SkillError, KeyError) as e:
            self._rest()
            if self.character and self.character.awake and not b.holding:
                self.character.tilt_head()  # puzzled
            err = {"ok": False, "error": str(e).strip("'\"")}
            if "can't see" in err["error"] and args.get("object"):
                last = self.memory.recall(args["object"])
                if last:
                    err["last_seen"] = f"{last['event']} {last['when']}"
            return err

    def _locate(self, obj: str) -> dict:
        """Point at it if it's on the bench; otherwise say what last happened to it."""
        b = self.bench
        try:
            s = b.look(obj)
        except SkillError:
            last = self.memory.recall(obj)
            if last:
                return {"found": False, "last_seen": f"{last['event']} {last['when']}"}
            return {"found": False, "last_seen": "never seen it"}
        self.memory.note(obj, "seen on the bench", s.x, s.y)
        reachable = b.point_at(s.x, s.y)
        self._rest()
        return {"found": True, "pointed_at_it": True, "within_reach": reachable,
                "position_cm": [round(s.x * 100, 1), round(s.y * 100, 1)]}

    def _tidy_up(self, objects: list[str], container: str) -> dict:
        """Every object into the container, back to back; a failed one is skipped, not fatal.

        One clean overhead photo is taken first, with the arm parked: later looks would have the
        arm (hovering over the tray) in the picture, hiding tools. The tray is found once."""
        b = self.bench
        clean = b.snapshot()
        box = None
        done, failed = [], {}
        for obj in objects:
            try:
                self._put_away(obj, container, frame=clean, box=box)
                box = getattr(self, "_last_box", None) or box
                done.append(obj)
            except (SkillError, KeyError) as e:
                failed[obj] = str(e).strip("'\"")
                self.log(f"  [tidy] skipped {obj}: {failed[obj]}")
        self._rest()
        return {"ok": bool(done), "put_away": done, "couldnt": failed, "into": container}

    def _put_away(self, obj: str, container: str, frame=None, box=None) -> dict:
        """Pick it up and drop it into the container; recorded as an episode like a fetch."""
        b = self.bench
        rec = self.recorder
        if rec:
            rec.start(f"put the {obj} in the {container}")
        t0 = time.time()
        try:
            s = b.pick(obj, frame=frame)
            box = b.drop_into(container, box=box)
        except (SkillError, KeyError) as e:
            if b.holding:  # couldn't find the container: don't keep holding it
                try:
                    b.put_back()
                except SkillError:
                    pass
            if rec:
                rec.stop(False, {"object": obj, "container": container, "error": str(e)})
            raise
        self.memory.note(obj, f"put in the {container}", box.x, box.y)
        if rec:
            rec.stop(True, {"object": obj, "container": container, "seconds": round(time.time() - t0, 1)})
        # Next tool runs straight from here (no rest pose in between): tidying up keeps moving.
        self._last_box = box  # a tidy-up reuses where the container is
        return {"ok": True, "put_away": s.prompt, "into": container}

    def _fetch(self, obj: str) -> dict:
        """Pick + hand over + home, recorded as one episode and checked by Muse Spark."""
        b = self.bench
        rec = self.recorder
        if rec:
            rec.start(f"pick up the {obj} and hand it over")
        t0 = time.time()
        try:
            s = b.pick(obj, for_hand=self.hand_to_hand)
            where = self._give()
            self.memory.note(obj, "handed to the user" if where == "hand" else "put on the handover spot")
            self._rest()
        except (SkillError, KeyError) as e:
            if rec:
                rec.stop(False, {"object": obj, "error": str(e)})
            raise
        ok, reason = (True, "not verified")
        if where == "hand":
            ok, reason = True, "taken from the gripper by the user"
            events.publish("verify", ok=True, reason="dropped into your hand", by="Handover")
        elif self.verify and not b.dry_run:
            try:
                ok, reason = self.verify_handover(obj)
            except Exception as e:  # a verifier hiccup shouldn't break the handover
                ok, reason = True, f"verifier error: {e.__class__.__name__}"
        if rec:
            rec.stop(ok, {"object": obj, "verifier": reason, "handed": where,
                          "seconds": round(time.time() - t0, 1)})
        # Report honestly but don't fail the tool: a failed check must not make the model re-run
        # the whole fetch (that would pick the object back up from next to the user).
        return {"ok": True, "handed_over": s.prompt, "where": where, "verified": ok,
                **({} if ok else {"note": reason})}

    # ------------------------------------------------------------------ one request
    def handle(self, text: str) -> str:
        t0 = time.time()
        bench_photo = self.bench.snapshot()
        spots = ", ".join(load_spots()) or "none saved"
        holding = self.bench.holding.prompt if self.bench.holding else "nothing"
        user = {"role": "user", "content": [
            {"type": "text", "text": f"{text}\n\n(state: holding {holding}; saved spots: {spots}; "
                                     f"you remember: {self.memory.summary()})"},
            {"type": "image_url", "image_url": {"url": _jpeg_data_url(bench_photo, max_side=1280)}},
        ]}
        # Keep earlier turns as text only (photos are big and stale).
        messages = [{"role": "system", "content": SYSTEM}, *self.history, user]
        client = get_client()
        reply = ""
        tidied = False
        announced = False
        for _ in range(self.max_steps):
            events.publish("status", text="Muse Spark is thinking...")
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
            spoke = bool(msg.content and msg.content.strip())
            if spoke and self.say and not announced:  # once per request: tidying shouldn't chatter
                self.say(msg.content.strip())
                announced = True
            results = []
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                self.log(f"  [tool] {tc.function.name}({', '.join(f'{k}={v!r}' for k, v in args.items())})")
                events.publish("tool", name=tc.function.name, args=args)
                result = self._tool(tc.function.name, args)
                self.log(f"         -> {result}")
                results.append((tc.function.name, args, result))
                tidied |= tc.function.name in ("put_away", "tidy_up")
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
            # A successful fetch is the whole job: answer right away instead of a second model call.
            if len(results) == 1 and results[0][0] == "fetch" and results[0][2].get("ok"):
                obj = results[0][1]["object"]
                if results[0][2].get("where") == "hand":
                    reply = random.choice(TAKEN).format(obj=obj)
                elif results[0][2].get("verified", True):
                    reply = random.choice(HANDED).format(obj=obj)
                else:
                    reply = f"I've put the {obj} down, but I'm not sure it landed in the right spot."
                break
            if len(results) == 1 and results[0][0] == "tidy_up":
                r = results[0][2]
                done, bad = r.get("put_away", []), r.get("couldnt", {})
                if done and not bad:
                    reply = random.choice(["All tidy!", "Done, everything's in the tray.", "Bench is clear!"])
                elif done:
                    reply = (f"Put away the {' and the '.join(done)}, but I couldn't manage the {', '.join(bad)}.")
                else:
                    reply = f"Sorry, I couldn't put any of them away: {next(iter(bad.values()), 'no idea why')}."
                break
            # Gestures only, and we've already said something: nothing more to add.
            if spoke and all(r[0] == "gesture" and r[2].get("ok") for r in results):
                break
        else:
            reply = "Sorry, that took too many steps; I've stopped."
        if tidied:  # put_away leaves the arm over the container, ready for the next one
            try:
                self._rest()
            except SkillError:
                pass
        self.history += [{"role": "user", "content": text}, {"role": "assistant", "content": reply}]
        self.history = self.history[-12:]
        self.log(f"  [{time.time() - t0:.1f}s]")
        return reply
