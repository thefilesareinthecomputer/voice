"""`voice listen`: record from the microphone, transcribe on device, print the text and
copy it to the clipboard.

Two ways to run it. Plain `voice listen` records until you pause and suits a terminal.
`voice listen --toggle` suits a keyboard shortcut: the first press starts a background
recording and returns at once, the next press stops it, and a sound plus a notification
mark each step, so you can walk away from the keyboard while it listens.
"""

from __future__ import annotations

import contextlib
import inspect
import os
import signal
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path

import numpy as np

from voice import models

RATE = 16_000          # what every speech-to-text model here expects
FRAME = 0.1            # seconds per level reading
SPEECH_LEVEL = 0.01    # RMS above which a frame counts as speech
CHUNK_SECONDS = 60.0   # long recordings are transcribed in chunks of this length

STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "claude-voice"
PIDFILE = STATE / "listen.pid"
LOG = STATE / "listen.log"
SOUNDS = {"start": "/System/Library/Sounds/Tink.aiff", "stop": "/System/Library/Sounds/Pop.aiff",
          "error": "/System/Library/Sounds/Basso.aiff"}


def end_of_speech(levels: list[float], silence: float, threshold: float = SPEECH_LEVEL) -> bool:
    """True once speech has started and the last `silence` seconds were all quiet."""
    quiet = int(round(silence / FRAME))
    if quiet < 1 or len(levels) <= quiet or not any(lvl > threshold for lvl in levels):
        return False
    return all(lvl <= threshold for lvl in levels[-quiet:])


def record(max_seconds: float, silence: float | None, stop: Callable[[], bool] | None = None) -> np.ndarray:
    """Record mono 16 kHz audio until `silence` seconds of quiet after speech (never, when None),
    until `stop()` turns true, or until `max_seconds`."""
    import sounddevice as sd

    frames: list[np.ndarray] = []
    levels: list[float] = []
    block = int(RATE * FRAME)
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=block) as stream:
        while len(frames) * FRAME < max_seconds and not (stop and stop()):
            chunk, _ = stream.read(block)
            frames.append(chunk[:, 0].copy())
            levels.append(float(np.sqrt(np.mean(chunk ** 2))))
            if silence is not None and end_of_speech(levels, silence):
                break
    return np.concatenate(frames) if frames else np.zeros(0, dtype="float32")


def transcribe_with(model, audio: np.ndarray) -> str:
    import mlx.core as mx

    if audio.size == 0:
        return ""
    options = {}
    # A long recording goes through in chunks where the model supports it, which bounds memory.
    if len(audio) > CHUNK_SECONDS * RATE and "chunk_duration" in inspect.signature(model.generate).parameters:
        options["chunk_duration"] = CHUNK_SECONDS
    result = model.generate(mx.array(audio), **options)
    return str(getattr(result, "text", "")).strip()


def transcribe(audio: np.ndarray, force: bool = False) -> str:
    """Load the model, transcribe one recording, release the model."""
    if audio.size == 0:
        return ""
    with models.loaded(_load, models.stt_model(), force) as model:
        return transcribe_with(model, audio)


def copy(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True, check=False)


def paste() -> bool:
    """Press Command-V in the frontmost app. macOS allows it only once the app that runs voice
    (Shortcuts, or your terminal) has Accessibility permission; returns whether it went through."""
    script = 'tell application "System Events" to keystroke "v" using command down'
    try:
        result = subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def cue(kind: str) -> None:
    """Play a short system sound without waiting for it."""
    with contextlib.suppress(OSError):
        subprocess.Popen(["afplay", SOUNDS[kind]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def notify(title: str, message: str) -> None:
    """Show a macOS notification. The text is passed as arguments, never spliced into the script."""
    script = "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run"
    with contextlib.suppress(OSError):
        subprocess.run(["osascript", "-e", script, title, message], capture_output=True, timeout=10)


def _load(repo: str):
    from mlx_audio.stt.utils import load_model

    return load_model(repo)


def run(max_seconds: float = 60.0, silence: float = 1.5, clipboard: bool = True, force: bool = False) -> int:
    """Record until a pause, in the foreground. The model loads while the speaker talks,
    so the wait after the pause is only the transcription."""
    with models.loading(_load, models.stt_model(), force) as pending:
        print("listening... (stops after a pause)", file=sys.stderr)
        audio = record(max_seconds, silence)
        text = transcribe_with(pending.result(), audio)
    if text:
        print(text)
        if clipboard:
            copy(text)
    return 0


def running_pid() -> int | None:
    """The pid of a background recording that is still running, if any."""
    try:
        pid = int(PIDFILE.read_text().strip())
    except (OSError, ValueError):
        return None
    ps = subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True)
    # Only a live voice recorder counts, never a reused pid.
    return pid if "voice" in ps.stdout and "--until-stopped" in ps.stdout else None


def toggle(max_seconds: float, clipboard: bool, force: bool, paste_too: bool = False) -> int:
    """Stop the background recording if one is running; otherwise start one and return at once."""
    pid = running_pid()
    if pid:
        os.kill(pid, signal.SIGUSR1)
        return 0
    models.check_pressure(force)
    STATE.mkdir(parents=True, exist_ok=True)
    argv = [sys.executable, "-m", "voice.cli", "listen", "--until-stopped", "--max-seconds", str(max_seconds)]
    argv += [] if clipboard else ["--no-copy"]
    argv += ["--paste"] if paste_too else []
    argv += ["--force"] if force else []
    with open(LOG, "a") as log:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True)
    return 0


def run_until_stopped(max_seconds: float, clipboard: bool, force: bool, paste_too: bool = False) -> int:
    """The background half of --toggle: record until signalled or `max_seconds`, then transcribe.
    With `paste_too`, the text also goes to the cursor, through the clipboard."""
    stopped = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: stopped.set())
    STATE.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    try:
        cue("start")
        notify("voice: listening", "Press your shortcut again to stop.")
        try:
            audio = record(max_seconds, None, stopped.is_set)
        finally:
            PIDFILE.unlink(missing_ok=True)  # a press from here on starts a new recording
        cue("stop")
        text = transcribe(audio, force)
    except Exception as exc:
        cue("error")
        notify("voice: failed", str(exc)[:200])
        raise
    if not text:
        notify("voice: nothing heard", "No speech was recognized.")
        return 0
    # Never print the text here: this process's output goes to the log file, and a
    # transcript must not reach the disk.
    if paste_too:
        copy(text)  # the paste goes through the clipboard, which keeps the text afterwards
        if paste():
            notify("voice: pasted", text[:200])
        else:
            notify("voice: copied, paste blocked",
                   "Allow Shortcuts (or your terminal) in System Settings > Privacy & Security > Accessibility.")
        return 0
    if clipboard:
        copy(text)
    notify("voice: copied" if clipboard else "voice: done", text[:200])
    return 0
