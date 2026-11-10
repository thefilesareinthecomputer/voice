"""`voice speak`: read text aloud, locally. A new call stops the one already playing,
so the latest reply always wins.

When the resident speaker (`voice serve`) is running, the text goes to it and plays in
about 0.3 s. Otherwise this process loads the model, plays the text and exits."""

from __future__ import annotations

import contextlib
import math
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

from voice import models, state

PIDFILE = state.STATE / "speak.pid"
# The newest request's ticket. A request that waited for the speaker lock plays only if its
# ticket is still the newest, so speech never plays out of order and a stop clears the queue.
TICKET = state.STATE / "speak.latest"
SPEED_RANGE = (0.5, 2.0)


def stop_previous() -> None:
    """Stop the speaker that is playing. Only a live speaker holds the lock, so the pid of one
    that has exited is never signalled."""
    pid = state.holder(PIDFILE)
    if pid and pid != os.getpid():
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGTERM)


def take_ticket(value: str) -> None:
    fd = os.open(TICKET, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(value)


def newest() -> str:
    try:
        return TICKET.read_text()
    except OSError:
        return ""


def settings(voice: str | None = None, speed: float | None = None) -> tuple[str, float, str]:
    """The voice, the speed and the pronunciation code. A flag beats the environment, which beats the default.
    Kokoro's voice names start with their accent code: `a` American, `b` British."""
    voice = voice or os.environ.get("VOICE_NAME", "af_heart")
    if speed is None:
        low, high = SPEED_RANGE
        try:
            speed = float(os.environ.get("VOICE_SPEED", "1.0"))
        except ValueError:
            speed = math.nan
        if not (math.isfinite(speed) and low <= speed <= high):
            raise SystemExit(f"voice: VOICE_SPEED must be a number from {low} to {high}, such as 1.0")
    return voice, speed, voice[:1] or "a"


def voice_folder() -> Path | None:
    """The voices folder of the configured speech model, or None before `voice setup`."""
    root = models.folder(models.tts_model())
    return root / "voices" if root else None


def voices(folder: Path | None = None) -> list[str]:
    """The voices of the configured speech model, or nothing before `voice setup`."""
    folder = folder or voice_folder()
    return sorted({p.stem for p in folder.glob("*.safetensors")}) if folder else []


def voice_file(name: str) -> str:
    """The file of a listed voice. Handing mlx-audio a path keeps it from looking up the name in
    its own default repo, or downloading it, and the listing check keeps the name inside the folder."""
    folder = voice_folder()
    if folder is None:
        raise models.ModelMissing(f"{models.tts_model()} is not on this machine; run `voice setup` first")
    if name not in voices(folder):
        raise SystemExit(f"voice: no voice named {name!r}; `voice voices` lists them")
    return str(folder / f"{name}.safetensors")


def render(model: Any, text: str, voice_path: str, speed: float, lang: str, audio: Any,
           cancel: threading.Event | None = None, guard: Any = None) -> None:
    """Play `text` through `audio` (sounddevice), making each sentence while the one before it
    plays, so there is no gap between them. A set `cancel` ends it before the next sentence;
    `guard` is held while a sentence starts, so a stop taken under it cannot be missed."""
    for result in model.generate(text, voice=voice_path, speed=speed, lang_code=lang):
        audio.wait()
        with guard or contextlib.nullcontext():
            if cancel is not None and cancel.is_set():
                return
            audio.play(result.audio, int(result.sample_rate))
    audio.wait()


def play(text: str, force: bool = False, voice: str | None = None, speed: float | None = None) -> None:
    """Load the model, speak `text`, release the model: the one-shot path, when no server runs."""
    import sounddevice as sd

    voice, speed, lang = settings(voice, speed)
    path = voice_file(voice)
    with models.loaded(models.load_tts, models.tts_model(), force) as model:
        render(model, text, path, speed, lang, sd)


def run(text: str | None, stop: bool = False, force: bool = False, voice: str | None = None,
        speed: float | None = None, warm: bool = False, release: bool = False) -> int:
    from voice import serve

    state.folder()
    if stop or release:
        take_ticket("stop")
        stop_previous()
        serve.request({"op": "quit" if release else "stop"})
        return 0
    if warm:
        serve.start(force)
    text = (text or "").strip()
    if not text:
        return 0
    ticket = f"{os.getpid()}:{time.time_ns()}"
    take_ticket(ticket)
    stop_previous()
    name, rate, _ = settings(voice, speed)
    reply = serve.say(text, name, rate)
    if reply is not None:
        if reply.startswith("error"):
            raise SystemExit(f"voice: {reply.removeprefix('error: ')}")
        return 0
    # SIGTERM keeps its default, so a stop ends playback at once; a Python handler would wait for
    # the audio call to return. The kernel drops the lock with the process either way.
    lock = state.hold(PIDFILE, wait=True)  # waits the moment the stopped speaker takes to exit
    try:
        if newest() != ticket:
            return 0  # a newer request, or a stop, came in while this one waited
        state.write_pid(lock, os.getpid())
        play(text, force, voice, speed)
    finally:
        os.close(lock)
    return 0
