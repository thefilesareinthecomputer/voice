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


def settings(voice: str | None = None, speed: float | None = None) -> tuple[str, float, str]:
    """The voice, the speed and the pronunciation code. A flag beats the environment, which beats the default.
    Kokoro's voice names start with their accent code: `a` American, `b` British."""
    voice = voice or os.environ.get("VOICE_NAME", "af_heart")
    speed = speed if speed is not None else float(os.environ.get("VOICE_SPEED", "1.0"))
    return voice, speed, voice[:1] or "a"


def voices() -> list[str]:
    """The voices of the default speech model in the local cache, or nothing before `voice setup`."""
    from huggingface_hub import snapshot_download

    try:
        root = Path(snapshot_download(models.tts_model(), local_files_only=True))
    except Exception:
        return []
    return sorted({p.stem for p in (root / "voices").glob("*.safetensors")})


def play(text: str, force: bool = False, voice: str | None = None, speed: float | None = None) -> None:
    import sounddevice as sd
    from mlx_audio.tts.utils import load_model

    voice, speed, lang = settings(voice, speed)
    with models.loaded(load_model, models.tts_model(), force) as model:
        for result in model.generate(text, voice=voice, speed=speed, lang_code=lang):
            sd.play(result.audio, int(result.sample_rate))
            sd.wait()


def run(text: str | None, stop: bool = False, force: bool = False,
        voice: str | None = None, speed: float | None = None) -> int:
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
        play(text, force, voice, speed)
    finally:
        try:
            if PIDFILE.read_text().strip() == str(os.getpid()):
                PIDFILE.unlink()
        except OSError:
            pass
    return 0
