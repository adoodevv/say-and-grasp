"""Voice in and out.

listen()      push-to-talk: Enter to start, Enter to stop -> WAV bytes (16 kHz mono PCM16)
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
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


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


def speak(text: str) -> None:
    print(f"\n  🤖 {text}\n", flush=True)
    if _tts:
        subprocess.Popen([_tts, text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    # Quick mic test: python -m say_and_grasp.voice
    wav = listen()
    print(f"recorded {len(wav) / (2 * RATE):.1f} s")
    print("transcript:", repr(transcribe(wav)))
