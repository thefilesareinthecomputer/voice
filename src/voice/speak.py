"""`voice speak`: read text aloud, locally. A new call stops the one already playing,
so the latest reply always wins."""

from __future__ import annotations

import os
import signal
import subprocess
from pathlib import Path

from voice import models

STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "claude-voice"
PIDFILE = STATE / "speak.pid"


def stop_previous() -> None:
    """Stop the speaker that is playing, if the recorded pid is still a speaker."""
    try:
        pid = int(PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return
    if pid == os.getpid():
        return
    # Only signal the pid if it is still a speaker, never a reused pid.
    ps = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    if "voice" in ps.stdout and "speak" in ps.stdout:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def play(text: str, force: bool = False) -> None:
    import sounddevice as sd
    from mlx_audio.tts.utils import load_model

    voice = os.environ.get("VOICE_NAME", "af_heart")
    speed = float(os.environ.get("VOICE_SPEED", "1.0"))
    with models.loaded(load_model, models.tts_model(), force) as model:
        for result in model.generate(text, voice=voice, speed=speed, lang_code="a"):
            sd.play(result.audio, int(result.sample_rate))
            sd.wait()


def run(text: str | None, stop: bool = False, force: bool = False) -> int:
    STATE.mkdir(parents=True, exist_ok=True)
    stop_previous()
    if stop:
        PIDFILE.unlink(missing_ok=True)
        return 0
    text = (text or "").strip()
    if not text:
        return 0
    PIDFILE.write_text(str(os.getpid()))
    signal.signal(signal.SIGTERM, lambda *_: os._exit(0))
    try:
        play(text, force)
    finally:
        try:
            if PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except OSError:
            pass
    return 0
