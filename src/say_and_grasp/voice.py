"""Voice in and out.

listen()      push-to-talk: Enter to start, Enter to stop -> WAV bytes (16 kHz mono PCM16)
listen_auto() hands-free: waits for speech, records until you pause -> WAV bytes (or None)
transcribe()  Muse Voice Transcribe via Meta Model API's file endpoint -> text
speak()       local text-to-speech (espeak-ng / spd-say if installed), else just prints
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import threading
import time
import wave

import numpy as np

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


def listen_auto(timeout: float | None = None, min_level: float = 400.0, pause_s: float = 0.8,
                max_s: float = 12.0) -> bytes | None:
    """Hands-free: wait for someone to speak, record until they pause, return the WAV.

    A simple energy detector: the first half second measures the room's noise floor; speech is
    anything well above it. Audio heard while Kojo itself is talking is ignored, so it doesn't
    answer its own voice. Returns None if nobody speaks within `timeout` seconds."""
    import queue

    import sounddevice as sd

    block = int(RATE * 0.03)  # 30 ms
    q: queue.Queue = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    def rms(x):
        return float(np.sqrt(np.mean(x.astype(np.float32) ** 2)))

    start = time.time()
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=block, callback=cb):
        noise = [rms(q.get()) for _ in range(15)]
        thresh = max(min_level, 3.0 * float(np.median(noise)))
        pre: list[np.ndarray] = []  # keep 0.3 s before speech starts, so the first syllable isn't cut
        rec: list[np.ndarray] = []
        quiet = 0.0
        while True:
            x = q.get()
            if speaking():
                pre.clear()
                rec.clear()
                start = time.time() if timeout else start
                continue
            loud = rms(x) > thresh
            if not rec:
                pre = (pre + [x])[-10:]
                if loud:
                    rec = pre[:]
                elif timeout is not None and time.time() - start > timeout:
                    return None
                continue
            rec.append(x)
            quiet = 0.0 if loud else quiet + 0.03
            if quiet >= pause_s or len(rec) * 0.03 >= max_s:
                break
    if len(rec) * 0.03 < 0.35 + pause_s:  # a click or a cough
        return listen_auto(timeout=None if timeout is None else max(0.1, timeout - (time.time() - start)),
                           min_level=min_level, pause_s=pause_s, max_s=max_s)
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


_tts = shutil.which("espeak-ng") or shutil.which("espeak") or shutil.which("spd-say")
_tts_proc: subprocess.Popen | None = None
_tts_until = 0.0


def speak(text: str) -> None:
    """Say it out loud without blocking (the arm keeps moving while Kojo talks)."""
    global _tts_proc
    if not text:
        return
    print(f"\n  🤖 Kojo: {text}\n", flush=True)
    if _tts:
        args = [_tts, text]
        if "espeak" in _tts:
            args = [_tts, "-s", "165", "-p", "60", text]  # a little quicker and higher: friendlier
        _tts_proc = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def speaking() -> bool:
    """True while Kojo is talking, and for a moment after (so the mic doesn't catch the echo)."""
    global _tts_until
    if _tts_proc is not None and _tts_proc.poll() is None:
        _tts_until = time.time() + 0.4
        return True
    return time.time() < _tts_until


if __name__ == "__main__":
    # Quick mic test: python -m say_and_grasp.voice
    wav = listen()
    print(f"recorded {len(wav) / (2 * RATE):.1f} s")
    print("transcript:", repr(transcribe(wav)))
