"""Wiggle's character: a name to answer to, a wake-up and a bedtime, and a few gestures.

Wiggle sleeps folded at home. Call its name and it wakes up: a sleepy twitch, a stretch and a
yawn (gripper wide open), a look around, and a happy double chirp of the gripper. While awake it
sits in its "awake" pose, looking at you, and goes back to sleep after a while with nothing to do.

The gripper is its mouth and the wrist is its head:
    wave        head waves side to side, mouth open
    nod         yes
    shake_head  no
    tilt_head   puzzled
    dance       a little celebration
    wake/sleep  the animations above

Gestures are small oscillations around the awake pose (signs don't matter, so they work on any
SO-101 calibration). The awake pose itself is saved by hand: sag-wiggle --save-awake.
"""
from __future__ import annotations

import json
import math
import random
import re
import time

import numpy as np

from .arm import load_clear, load_home
from .paths import CONFIG_DIR

NAME = "Wiggle"
AWAKE_FILE = CONFIG_DIR / "awake.json"
GESTURES = ["wave", "nod", "shake_head", "tilt_head", "dance", "wake", "sleep"]

PAN, LIFT, ELBOW, WFLEX, ROLL = range(5)
MOUTH_REST = 20.0  # gripper opening while awake

# How the name may come back from speech-to-text: Wiggle, Wiggles, Wiggy, Wigle, Wiggel...
_NAME = r"(?:wig{1,2}(?:le|les|el|al|ly|y|gle)|wriggle|wiggle)"
_CALL = re.compile(rf"\b(?:(?:hey|hi|hello|yo|ok|okay)[\s,]+)?{_NAME}\b[\s,.!?]*", re.I)
# Sloppier hearings ("Wego", "we go", "Wiggo", "Vega") only count at the very start of what was
# said, so "here we go" in conversation doesn't wake it.
_LOOSE = r"(?:we[\s-]?go+|wiggo|wigo|weego|wego|vega|wigga|weegle|wigl)"
_CALL_START = re.compile(rf"^\W*(?:(?:hey|hi|hello|yo|ok|okay)[\s,]+)?{_LOOSE}\b[\s,.!?]*", re.I)
_JUST_CALLING = re.compile(
    r"^(?:wake up|are you (?:there|awake|up)|you there|hello|hi|hey|good morning|morning)?\W*$", re.I)

GREETINGS = [
    "Hey! I'm up. What do you need?",
    "Ahh, good nap. What can I get you?",
    "I'm here, I'm here! What are we building?",
    "Yes boss? Ready when you are.",
]
GOODNIGHTS = ["Okay, nap time. Call me if you need me.", "Alright, I'll be right here. Goodnight!",
              "Going to sleep. Just say my name."]


def heard_name(text: str) -> tuple[bool, str]:
    """Was Wiggle called? Returns (called, what's left of the request).

    "Wiggle, pass me the pliers" -> (True, "pass me the pliers"); "hey wiggle!" -> (True, "")."""
    if _CALL.search(text):
        rest = _CALL.sub(" ", text).strip(" ,.!?")
    elif _CALL_START.search(text):
        rest = _CALL_START.sub(" ", text).strip(" ,.!?")
    else:
        return False, text.strip()
    if _JUST_CALLING.match(rest):
        rest = ""
    return True, rest


def load_awake() -> np.ndarray:
    if AWAKE_FILE.exists():
        return np.array(json.loads(AWAKE_FILE.read_text())["joints"], dtype=float)
    # Fallback: home with the arm raised a little and the head (wrist) up. Save your own.
    return load_home() + np.array([0.0, 30.0, -30.0, -35.0, 0.0])


def save_awake(joints: np.ndarray) -> None:
    AWAKE_FILE.parent.mkdir(parents=True, exist_ok=True)
    AWAKE_FILE.write_text(json.dumps({"joints": [round(float(v), 2) for v in joints]}, indent=2))


class Character:
    """Wiggle's body language on top of a say_and_grasp.skills.Bench."""

    def __init__(self, bench, say=print, log=print, sleep_after_s: float = 60.0):
        self.bench, self.say, self.log = bench, say, log
        self.awake = False
        self.awake_pose = load_awake()
        self.sleep_after_s = sleep_after_s
        self.last_active = time.time()
        if not AWAKE_FILE.exists():
            self.log("[wiggle] no awake pose saved; using a default (sag-wiggle --save-awake)")

    @property
    def arm(self):
        return self.bench.arm

    def touch(self) -> None:
        self.last_active = time.time()

    def sleepy(self) -> bool:
        return self.awake and time.time() - self.last_active > self.sleep_after_s

    # ------------------------------------------------------------------ motion primitives
    def _goto(self, q, gripper: float | None = None, speed: float | None = None, settle: float = 0.1) -> None:
        if not self.arm:
            return
        old = self.arm.speed
        if speed:
            self.arm.speed = speed
        try:
            self.arm.move(np.asarray(q, dtype=float), gripper=gripper, settle_s=settle)
        finally:
            self.arm.speed = old

    def _wiggle(self, seconds: float, joints: dict[int, tuple[float, float, float]] | None = None,
                mouth: tuple[float, float] | None = None, hz: float = 30.0) -> None:
        """Oscillate joints around the current pose: {joint: (amplitude deg, frequency Hz, phase)}.

        mouth=(amplitude, Hz) chatters the gripper. The motion fades in and out, so it starts and
        ends exactly at the pose it began from."""
        if not self.arm:
            time.sleep(seconds)
            return
        arm = self.arm
        base = arm.q_cmd.copy() if arm.q_cmd is not None else arm.joints()
        g0 = arm.gripper
        n = max(int(seconds * hz), 2)
        fade = min(0.3, seconds / 3)
        for k in range(n + 1):
            t = k / hz
            env = max(0.0, min(1.0, t / fade, (seconds - t) / fade))
            env = math.sin(env * math.pi / 2) ** 2
            q = base.copy()
            for j, (amp, f, ph) in (joints or {}).items():
                q[j] += env * amp * math.sin(2 * math.pi * f * t + ph)
            g = g0
            if mouth:
                g = float(np.clip(g0 + env * mouth[0] * math.sin(2 * math.pi * mouth[1] * t), 0, 100))
            arm._send(q + arm.corr, g)
            time.sleep(1.0 / hz)
        arm.q_cmd, arm.gripper = base, g0

    def _at_awake(self) -> None:
        """Make sure we're up and in the awake pose before a gesture."""
        if not self.awake:
            self.wake(quick=True)
        elif self.arm and np.max(np.abs(self.arm.q_cmd - self.awake_pose)) > 3:
            self._goto(self.awake_pose, gripper=MOUTH_REST, speed=60)

    # ------------------------------------------------------------------ wake / sleep
    def wake(self, quick: bool = False, greet: bool = True) -> None:
        self.touch()
        if self.awake:
            self._at_awake()
            if greet:
                self.say(random.choice(["Still here!", "Yep, I'm listening.", "What's up?"]))
            return
        self.log("[wiggle] waking up" + (" (quick)" if quick else ""))
        a = self.awake_pose
        clear = load_clear()
        if quick:
            if clear is not None:
                self._goto(clear, speed=70, settle=0)
            self._goto(a, gripper=MOUTH_REST, speed=70)
            self.awake = True
            return
        # 1. a sleepy twitch: "huh?"
        self._wiggle(0.9, {WFLEX: (3.0, 1.1, 0.0)})
        time.sleep(0.3)
        # 2. lift the head, then stretch up tall with a big yawn (mouth wide open)
        if clear is not None:
            self._goto(clear, speed=25, settle=0)
        stretch = a.copy()
        stretch[LIFT] += 8
        stretch[WFLEX] -= 15
        self._goto(stretch, gripper=80, speed=30, settle=0.2)
        if greet:
            self.say(random.choice(GREETINGS))
        time.sleep(0.5)
        # 3. relax into the awake pose, mouth closes
        self._goto(a, gripper=MOUTH_REST, speed=45, settle=0)
        # 4. look around for you
        for dp, sp in ((-18, 55), (18, 65), (0, 65)):
            look = a.copy()
            look[PAN] += dp
            self._goto(look, speed=sp, settle=0)
            time.sleep(0.2)
        # 5. a happy double chirp
        self._wiggle(1.0, mouth=(18.0, 2.0))
        self.awake = True

    def sleep(self, goodnight: bool = True) -> None:
        if not self.awake:
            return
        self.log("[wiggle] going to sleep")
        if self.bench.holding:
            self.bench.put_back()
        if goodnight:
            self.say(random.choice(GOODNIGHTS))
        a = self.awake_pose
        self._goto(a, speed=45, settle=0)
        # a yawn ...
        yawn = a.copy()
        yawn[WFLEX] -= 10
        self._goto(yawn, gripper=70, speed=30, settle=0.3)
        # ... then droop down slowly, mouth closing
        clear = load_clear()
        if clear is not None:
            self._goto(clear, gripper=self.bench.g_closed + 10, speed=25, settle=0)
        self._goto(load_home(), gripper=self.bench.g_closed + 4, speed=20, settle=0.3)
        # a little snore
        self._wiggle(2.4, mouth=(4.0, 0.45))
        self.awake = False

    # ------------------------------------------------------------------ gestures
    def wave(self) -> None:
        self._at_awake()
        a = self.awake_pose
        up = a.copy()
        up[WFLEX] -= 15
        self._goto(up, gripper=70, speed=70, settle=0)
        self._wiggle(2.6, {WFLEX: (18.0, 1.2, 0.0), PAN: (8.0, 0.6, 0.0)})
        self._goto(a, gripper=MOUTH_REST, speed=70, settle=0)

    def nod(self) -> None:
        self._at_awake()
        self._wiggle(1.4, {WFLEX: (10.0, 1.5, 0.0)})

    def shake_head(self) -> None:
        self._at_awake()
        self._wiggle(1.6, {PAN: (12.0, 1.3, 0.0)})

    def tilt_head(self) -> None:
        self._at_awake()
        a = self.awake_pose
        tilt = a.copy()
        tilt[ROLL] += -25 if a[ROLL] > 0 else 25  # towards the middle of its range
        self._goto(tilt, speed=70, settle=0)
        time.sleep(0.7)
        self._goto(a, speed=60, settle=0)

    def dance(self, seconds: float = 4.0) -> None:
        self._at_awake()
        self._wiggle(seconds, {PAN: (14.0, 0.75, 0.0), WFLEX: (12.0, 1.5, 0.0),
                               ROLL: (12.0, 0.75, math.pi / 2)}, mouth=(16.0, 2.0))

    def gesture(self, name: str) -> None:
        name = name.strip().lower().replace(" ", "_")
        fn = {"wave": self.wave, "nod": self.nod, "shake_head": self.shake_head, "tilt_head": self.tilt_head,
              "dance": self.dance, "wake": lambda: self.wake(greet=False), "sleep": self.sleep}.get(name)
        if fn is None:
            raise KeyError(f"I don't know the gesture {name!r}; I can do {', '.join(GESTURES)}.")
        self.touch()
        fn()

    # ------------------------------------------------------------------ after a task
    def rest(self) -> None:
        """Where to go once a job is done: back to looking at you if awake, else fold up."""
        self.touch()
        if self.awake and self.arm:
            self._goto(self.awake_pose, gripper=MOUTH_REST if not self.bench.holding else None, speed=self.arm.speed)
        else:
            self.bench.go_home()
