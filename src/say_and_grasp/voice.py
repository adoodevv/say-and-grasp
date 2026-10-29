"""Voice in and out.

listen()      push-to-talk: Enter to start, Enter to stop -> WAV bytes (16 kHz mono PCM16)
listen_auto() hands-free: waits for speech, records until you pause -> WAV bytes (or None)
transcribe()  Muse Voice Transcribe via Meta Model API's file endpoint -> text
speak()       natural offline voice (Piper), espeak as a fallback
"""
from __future__ import annotations

import io
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import time
import wave

import numpy as np
from pathlib import Path

ASR_URL = "https://api.meta.ai/v1/asr/transcribe"
ASR_MODEL = "muse-voice-transcribe-1.0"
RATE = 16000


def listen(prompt: str = "Press Enter and speak; Enter again when done... ") -> bytes:
    """Record from the default microphone between two Enter presses."""
    import sounddevice as sd

    input(prompt)
    chunks: list[np.ndarray] = []
    stop = threading.Event()

    def cb(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", callback=cb):
        print("  [listening...] Enter to stop", flush=True)
        threading.Thread(target=lambda: (input(), stop.set()), daemon=True).start()
        stop.wait()
    audio = np.concatenate(chunks) if chunks else np.zeros((0, 1), dtype=np.int16)
    return _wav(audio)


def _wav(audio: np.ndarray) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.astype(np.int16).tobytes())
    return buf.getvalue()


def listen_auto(timeout: float | None = None, min_level: float = 300.0, max_level: float = 1500.0,
                pause_s: float = 0.8, max_s: float = 12.0, log=None) -> bytes | None:
    """Hands-free: wait for someone to speak, record until they pause, return the WAV.

    A simple energy detector with a noise floor that keeps adapting to the room (fans, the arm's
    servos), measured only while nobody, Wiggle included, is talking. Audio heard while Wiggle
    itself talks is thrown away, so it never answers its own voice. Returns None if nobody speaks
    within `timeout` seconds."""
    import queue

    import sounddevice as sd

    while speaking():  # let Wiggle finish its sentence before measuring the room
        time.sleep(0.05)
    block = int(RATE * 0.03)  # 30 ms
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

    t_end = None if timeout is None else time.time() + timeout
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=block, callback=cb):
        noise = float(np.percentile([rms(q.get()) for _ in range(10)], 20))
        if log:
            log(f"  [mic] room noise {noise:.0f}, speech above {min(max(min_level, 3 * noise), max_level):.0f}")
        while True:
            pre: list[np.ndarray] = []  # keep 0.3 s before speech starts, so the first syllable isn't cut
            rec: list[np.ndarray] = []
            quiet = 0.0
            while True:
                try:
                    x = q.get(timeout=0.5)
                except queue.Empty:
                    x = None
                if t_end is not None and not rec and time.time() > t_end:
                    return None
                if x is None:
                    continue
                if speaking():
                    pre.clear()
                    rec.clear()
                    continue
                level = rms(x)
                thresh = min(max(min_level, 3.0 * noise), max_level)
                loud = level > thresh
                if not rec:
                    if not loud:  # track the room: drop quickly when it gets quieter, rise slowly
                        noise = 0.7 * noise + 0.3 * level if level < noise else 0.99 * noise + 0.01 * level
                    pre = (pre + [x])[-10:]
                    if loud:
                        rec = pre[:]
                    continue
                rec.append(x)
                quiet = 0.0 if loud else quiet + 0.03
                if quiet >= pause_s or len(rec) * 0.03 >= max_s:
                    break
            if len(rec) * 0.03 >= 0.35 + pause_s:  # long enough to be words, not a click or a cough
                return _wav(np.concatenate(rec))


def transcribe(wav: bytes, keywords: list[str] | None = None) -> str:
    """Send a WAV to Muse Voice Transcribe (file endpoint) and return the transcript.

    The file endpoint takes audioEncoding "WAV"; keywords/PUSH_TO_TALK only exist on the realtime
    WebSocket, so `keywords` is accepted here but not sent.
    """
    import httpx

    attempts = [
        {"model": ASR_MODEL, "audioEncoding": "WAV", "mode": "ENDPOINTING"},
        {"model": ASR_MODEL, "audioEncoding": "WAV"},
    ]
    last = None
    for settings in attempts:
        r = httpx.post(
            ASR_URL,
            headers={"Authorization": f"Bearer {os.environ['META_API_KEY']}", "Accept": "application/json"},
            files={
                "request": ("request.json", json.dumps(settings), "application/json"),
                "audio": ("speech.wav", wav, "audio/wav"),
            },
            timeout=60,
        )
        if r.status_code < 400:
            return (r.json().get("transcript") or "").strip()
        last = f"({r.status_code}) {r.text[:300]} [settings {settings}]"
        if r.status_code != 400:
            break
    raise RuntimeError(f"Transcription failed {last}")


# ---------------------------------------------------------------------------- speech out
# Piper: a small neural voice that runs offline on the CPU and sounds natural. espeak is the
# robotic-sounding fallback if Piper or its voice isn't installed.
#   pip install piper-tts
#   python -m piper.download_voices en_US-lessac-medium --download-dir assets/voices
VOICE = os.environ.get("WIGGLE_VOICE", "en_US-lessac-medium")
VOICE_DIR = Path(__file__).resolve().parents[2] / "assets" / "voices"
SPEED = float(os.environ.get("WIGGLE_SPEED", "0.9"))  # < 1 talks faster

_espeak = shutil.which("espeak-ng") or shutil.which("espeak")
_aplay = shutil.which("aplay") or shutil.which("paplay")
_piper = None  # loaded voice (or False if unavailable)
_queue: "queue.Queue[str]" = queue.Queue()
_busy = threading.Event()
_until = 0.0
_worker: threading.Thread | None = None


def _load_piper():
    global _piper
    if _piper is None:
        try:
            from piper import PiperVoice

            model = VOICE_DIR / f"{VOICE}.onnx"
            _piper = PiperVoice.load(model) if model.exists() else False
            if not _piper:
                print(f"[voice] {model} not found; using espeak. Download it with:\n"
                      f"  python -m piper.download_voices {VOICE} --download-dir {VOICE_DIR}")
        except Exception as e:
            print(f"[voice] Piper unavailable ({e.__class__.__name__}); using espeak (pip install piper-tts)")
            _piper = False
    return _piper


def _piper_wav(text: str) -> bytes:
    from piper.config import SynthesisConfig

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        _piper.synthesize_wav(text, w, syn_config=SynthesisConfig(length_scale=SPEED))
    return buf.getvalue()


def _play_wav(data: bytes) -> None:
    """Play a WAV and return only once it has finished (so lines never overlap)."""
    with wave.open(io.BytesIO(data)) as w:
        duration = w.getnframes() / w.getframerate()
    t0 = time.time()
    _play_wav_raw(data)
    left = duration - (time.time() - t0)
    if left > 0:  # some players return while the sound is still draining
        time.sleep(left + 0.05)


def _play_wav_raw(data: bytes) -> None:
    if _aplay:
        subprocess.run([_aplay, "-q", "-"] if _aplay.endswith("aplay") else [_aplay],
                       input=data, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return
    import sounddevice as sd

    with wave.open(io.BytesIO(data)) as w:
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        sd.play(audio, w.getframerate())
        sd.wait()


def _say_now(text: str) -> None:
    if _load_piper():
        try:
            _play_wav(_piper_wav(text))  # the whole sentence at once: no gaps mid-sentence
            return
        except Exception as e:
            print(f"[voice] Piper failed ({e}); falling back to espeak")
    if _espeak:
        r = subprocess.run([_espeak, "-s", "165", "-p", "60", "--stdout", text], capture_output=True)
        if r.returncode == 0 and r.stdout:
            _play_wav(r.stdout)


def _run() -> None:
    global _until
    while True:
        text = _queue.get()
        _busy.set()
        try:
            _say_now(text)
        finally:
            _until = time.time() + 0.4
            if _queue.empty():
                _busy.clear()


_last_said = ("", 0.0)


def _same(a: str, b: str) -> bool:
    wa, wb = set(re.findall(r"[a-z']+", a.lower())), set(re.findall(r"[a-z']+", b.lower()))
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) >= 0.5


def speak(text: str) -> None:
    """Say it out loud without blocking (the arm keeps moving while Wiggle talks). Lines are
    spoken in order, one after another, and a line that repeats the one just said is skipped."""
    global _worker, _last_said
    text = (text or "").strip()
    if not text:
        return
    if _same(text, _last_said[0]) and time.time() - _last_said[1] < 30:
        return
    _last_said = (text, time.time())
    print(f"\n  🤖 Wiggle: {text}\n", flush=True)
    if _worker is None:
        _load_piper()  # load the voice once, up front
        _worker = threading.Thread(target=_run, daemon=True)
        _worker.start()
    _busy.set()
    _queue.put(text)


def warm_up() -> None:
    """Load the voice now so the first reply isn't delayed."""
    if _load_piper():
        try:
            _piper_wav("hi")
        except Exception:
            pass


def speaking() -> bool:
    """True while Wiggle is talking, and for a moment after (so the mic doesn't catch the echo)."""
    return _busy.is_set() or time.time() < _until


if __name__ == "__main__":
    # Voice and mic test: python -m say_and_grasp.voice
    speak("Hey! I'm Wiggle. Say something and I'll tell you what I heard.")
    while speaking():
        time.sleep(0.1)
    wav = listen()
    print(f"recorded {len(wav) / (2 * RATE):.1f} s")
    print("transcript:", repr(transcribe(wav)))
