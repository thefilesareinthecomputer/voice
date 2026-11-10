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
import time
import traceback
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from voice import models, state

if TYPE_CHECKING:  # numpy loads with the recording, not with every key press
    import numpy as np

RATE = 16_000          # what every speech-to-text model here expects
FRAME = 0.1            # seconds per level reading
SPEECH_LEVEL = 0.01    # RMS above which a frame counts as speech
CUE_SECONDS = 0.6      # the start sound lasts 0.56 s; afplay itself stays up about a second longer
CHUNK_SECONDS = 60.0   # long recordings are transcribed in chunks of this length
PRESSURE_WAIT = 60.0   # seconds a finished take waits for critical memory pressure to ease

PIDFILE = state.STATE / "listen.pid"
LOG = state.STATE / "listen.log"
TRANSCRIBING = state.STATE / "transcribe.lock"
SOUNDS = {"start": "/System/Library/Sounds/Tink.aiff", "stop": "/System/Library/Sounds/Pop.aiff",
          "error": "/System/Library/Sounds/Basso.aiff"}
# By absolute path, so a folder early on PATH cannot stand in for them.
PBCOPY, OSASCRIPT, AFPLAY = "/usr/bin/pbcopy", "/usr/bin/osascript", "/usr/bin/afplay"


def end_of_speech(levels: list[float], silence: float, threshold: float = SPEECH_LEVEL) -> bool:
    """True once speech has started and the last `silence` seconds were all quiet."""
    quiet = int(round(silence / FRAME))
    if quiet < 1 or len(levels) <= quiet or not any(lvl > threshold for lvl in levels):
        return False
    return all(lvl <= threshold for lvl in levels[-quiet:])


def record(max_seconds: float, silence: float | None, stop: Callable[[], bool] | None = None,
           on_open: Callable[[], subprocess.Popen | None] | None = None,
           patience: float | None = None) -> np.ndarray:
    """Record mono 16 kHz audio until `silence` seconds of quiet after speech (never, when None),
    until `stop()` turns true, until `patience` seconds pass with no speech at all, or until
    `max_seconds`. `on_open` runs once the microphone is open, so a start cue never plays
    before the first word can be heard. When it returns the cue's process, what the microphone
    hears while the cue plays is dropped: through built-in speakers the cue is loud enough to
    count as speech, which would start the end-of-turn countdown before the user says a word."""
    import numpy as np
    import sounddevice as sd

    frames: list[np.ndarray] = []
    levels: list[float] = []
    spoke = False
    block = int(RATE * FRAME)
    options = {"samplerate": RATE, "channels": 1, "dtype": "float32", "blocksize": block}
    with contextlib.ExitStack() as stack:
        try:
            stream = stack.enter_context(sd.InputStream(**options))
        except sd.PortAudioError:
            # PortAudio keeps the device list it found at startup, so in a long-running process
            # (voice chat) the microphone fails once the input changes or the Mac wakes from sleep.
            # Start it again and retry once.
            sd._terminate()
            sd._initialize()
            stream = stack.enter_context(sd.InputStream(**options))
        playing = on_open() if on_open else None
        deadline = time.monotonic() + CUE_SECONDS
        while (playing is not None and playing.poll() is None and time.monotonic() < deadline
               and not (stop and stop())):
            stream.read(block)
        while len(frames) * FRAME < max_seconds and not (stop and stop()):
            chunk, _ = stream.read(block)
            frames.append(chunk[:, 0].copy())
            levels.append(float(np.sqrt(np.mean(chunk ** 2))))
            spoke = spoke or levels[-1] > SPEECH_LEVEL
            if silence is not None and end_of_speech(levels, silence):
                break
            if patience is not None and not spoke and len(levels) * FRAME >= patience:
                break
    return np.concatenate(frames) if frames else np.zeros(0, dtype="float32")


def peak(audio: np.ndarray) -> float:
    import numpy as np

    return float(np.max(np.abs(audio))) if audio.size else 0.0


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
    with models.loaded(models.load_stt, models.stt_model(), force) as model:
        return transcribe_with(model, audio)


def warm_imports() -> None:
    """Import MLX and the Parakeet code while the speaker talks; the stop press then waits only
    for the weights. Nothing here allocates model memory."""
    with contextlib.suppress(Exception):
        import mlx.core  # noqa: F401
        import mlx_audio.stt.models.parakeet  # noqa: F401
        import mlx_audio.utils  # noqa: F401


# Puts stdin on the clipboard as plain text marked org.nspasteboard.ConcealedType, the
# convention (nspasteboard.org) clipboard-history apps follow to leave an item out of their
# history. The text arrives on stdin, never in argv, where ps would show it.
CONCEALED_COPY = """ObjC.import('AppKit');
function run() {
  var data = $.NSFileHandle.fileHandleWithStandardInput.readDataToEndOfFile;
  var text = $.NSString.alloc.initWithDataEncoding(data, $.NSUTF8StringEncoding);
  var pb = $.NSPasteboard.generalPasteboard;
  pb.clearContents;
  if (!pb.setStringForType(text, $.NSPasteboardTypeString)) { throw new Error('copy refused'); }
  pb.setStringForType($(''), 'org.nspasteboard.ConcealedType');
}"""


def copy(text: str) -> bool:
    """Put a transcript on the clipboard, marked so clipboard-history apps skip it; plain
    pbcopy when that fails, so the text still arrives. Returns whether either worked."""
    try:
        if subprocess.run([OSASCRIPT, "-l", "JavaScript", "-e", CONCEALED_COPY], input=text, text=True,
                          capture_output=True, timeout=10).returncode == 0:
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        return subprocess.run([PBCOPY], input=text, text=True, check=False).returncode == 0
    except OSError:
        return False


def paste() -> bool:
    """Press Command-V in the frontmost app through System Events; returns whether it went through.
    macOS asks the app in front, not voice, for permission to control System Events, so each app
    dictated into asks once, and a yes gives that app lasting control (see the README)."""
    script = 'tell application "System Events" to keystroke "v" using command down'
    try:
        result = subprocess.run([OSASCRIPT, "-e", script], capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


def cue(kind: str) -> subprocess.Popen | None:
    """Play a short system sound without waiting for it; returns its process, None if it could not start."""
    try:
        return subprocess.Popen([AFPLAY, SOUNDS[kind]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def notify(title: str, message: str) -> None:
    """Show a macOS notification without waiting for it. The text is passed as arguments, never
    spliced into the script."""
    script = "on run argv\ndisplay notification (item 2 of argv) with title (item 1 of argv)\nend run"
    with contextlib.suppress(OSError):
        subprocess.Popen([OSASCRIPT, "-e", script, title, message],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def words(text: str) -> str:
    """What a notification says about a transcript. macOS keeps notification text in its own
    database, so the transcript itself never goes into one."""
    n = len(text.split())
    return f"{n} word" if n == 1 else f"{n} words"


def run(max_seconds: float = 60.0, silence: float = 1.5, clipboard: bool = True, force: bool = False) -> int:
    """Record until a pause, in the foreground. The model loads while the speaker talks,
    so the wait after the pause is only the transcription."""
    models.check_pressure(force)  # the cheap refusal first, whether or not a model is installed
    models.require(models.stt_model())  # before the recording, not after it
    with models.loading(models.load_stt, models.stt_model(), force) as pending:
        print("listening... (stops after a pause)", file=sys.stderr)
        audio = record(max_seconds, silence)
        text = transcribe_with(pending.result(), audio)
    if text:
        print(text)
        if clipboard:
            copy(text)
    return 0


def toggle(max_seconds: float, clipboard: bool, force: bool, paste_too: bool = False,
           preload: bool = False) -> int:
    """Stop the background recording if one is running; otherwise start one and return at once.
    A start that is refused says so with the error sound and a notification, since a keyboard
    shortcut has no terminal to print to."""
    pid = state.holder(PIDFILE)
    if pid:
        with contextlib.suppress(ProcessLookupError):  # it finished as the press landed
            os.kill(pid, signal.SIGUSR1)
        return 0
    try:
        models.check_pressure(force)
        models.require(models.stt_model())
    except models.ModelMissing as exc:
        cue("error")
        notify("voice: not started", f"{exc}.")
        return 4
    except models.MemoryPressure:
        cue("error")
        notify("voice: not started", "Memory pressure is critical. Close something, then press again.")
        return 3
    lock = state.hold(PIDFILE)
    if lock is None:
        return 0  # a recorder is starting: it holds the lock but has not written its pid yet
    try:
        # The recorder inherits the lock and keeps it while it records, so a press while it starts
        # finds the lock taken and does nothing. -I keeps the folder the shortcut ran in, and
        # PYTHONPATH, off its import path.
        argv = [sys.executable, "-I", "-m", "voice.cli", "listen", "--until-stopped", str(lock),
                "--max-seconds", str(max_seconds)]
        argv += [] if clipboard else ["--no-copy"]
        argv += ["--paste"] if paste_too else []
        argv += ["--preload"] if preload else []
        argv += ["--force"] if force else []
        fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with open(fd, "a") as log:
            subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=state.STATE,
                             pass_fds=(lock,), start_new_session=True)
    finally:
        os.close(lock)
    return 0


def check_lock(fd: int) -> None:
    """The descriptor `toggle` handed over must be the pid lock itself, so --until-stopped never
    writes a pid through, or closes, any other open file."""
    try:
        mine, expected = os.fstat(fd), os.stat(PIDFILE, follow_symlinks=False)
    except OSError:
        raise SystemExit("voice: --until-stopped needs the lock voice listen --toggle hands over") from None
    if (mine.st_dev, mine.st_ino) != (expected.st_dev, expected.st_ino):
        raise SystemExit("voice: --until-stopped needs the lock voice listen --toggle hands over")


@contextlib.contextmanager
def one_at_a_time() -> Iterator[None]:
    """Takes transcribe in turn, so a quick second take neither loads a second model beside the
    first nor overwrites a newer clipboard with an older transcript."""
    fd = state.hold(TRANSCRIBING, wait=True)
    try:
        yield
    finally:
        os.close(fd)


def transcribe_when_memory_allows(audio: np.ndarray, force: bool) -> str:
    """A finished take is worth keeping: under critical pressure it waits for the pressure to
    ease, up to PRESSURE_WAIT seconds, then loads anyway rather than discard the recording."""
    if not force and models.pressure_level() >= models.PRESSURE_CRITICAL:
        notify("voice: waiting for memory", "Memory pressure is critical; your recording is kept.")
        deadline = time.monotonic() + PRESSURE_WAIT
        while time.monotonic() < deadline and models.pressure_level() >= models.PRESSURE_CRITICAL:
            time.sleep(2)
    return transcribe(audio, force=True)


def deliver(text: str, clipboard: bool, paste_too: bool) -> None:
    """Hand the text on and say what happened. Never print it: this process's output goes to the
    log file, and a transcript must not reach the disk."""
    if not text:
        notify("voice: nothing heard", "No speech was recognized.")
        return
    if (clipboard or paste_too) and not copy(text):
        notify("voice: copy failed", "The text could not reach the clipboard.")
        return
    if paste_too:
        if paste():
            notify("voice: pasted", words(text))
        else:
            # No advice to grant anything: the grant macOS asks for goes to whichever app is in front.
            notify("voice: copied, paste blocked", "The text is on the clipboard. See README: --paste.")
        return
    notify("voice: copied" if clipboard else "voice: done", words(text))


def run_until_stopped(max_seconds: float, clipboard: bool, force: bool, paste_too: bool = False,
                      lock_fd: int | None = None, preload: bool = False) -> int:
    """The background half of --toggle: record until signalled or `max_seconds`, then transcribe.
    `lock_fd` is the pid lock `toggle` handed over; closing it lets the next press start a new
    recording. With `preload`, the model loads while the speaker talks, which shortens the wait
    after the stop press and holds about 1.7 GB for the whole recording."""
    if lock_fd is not None:
        check_lock(lock_fd)
    stopped = threading.Event()
    signal.signal(signal.SIGUSR1, lambda *_: stopped.set())
    if lock_fd is not None:
        state.write_pid(lock_fd, os.getpid())  # only now, when a press can stop it rather than kill it

    def opened() -> subprocess.Popen | None:
        playing = cue("start")
        notify("voice: listening", "Press your shortcut again to stop.")
        if not preload:
            threading.Thread(target=warm_imports, daemon=True).start()
        return playing

    try:
        loading = models.loading(models.load_stt, models.stt_model(), force) if preload else contextlib.nullcontext()
        with loading as pending:
            try:
                audio = record(max_seconds, None, stopped.is_set, on_open=opened)
            finally:
                if lock_fd is not None:
                    os.close(lock_fd)  # a press from here on starts a new recording
            cue("stop")
            loudest = peak(audio)
            if loudest <= SPEECH_LEVEL:
                hint = ("No sound reached the microphone. Check System Settings > Privacy & Security > Microphone."
                        if loudest == 0 else "No speech was recognized.")
                notify("voice: nothing heard", hint)
                return 0
            with one_at_a_time():
                if preload:
                    text = transcribe_with(pending.result(), audio)
                else:
                    text = transcribe_when_memory_allows(audio, force)
                deliver(text, clipboard, paste_too)
    except models.ModelMissing as exc:
        cue("error")
        notify("voice: model missing", f"{exc}.")
        return 4
    except Exception as exc:
        # The notification names the log, never the error: macOS keeps notification text, and
        # an error from transcription could quote what was said. The log gets the exception's
        # type and stack without its message, for the same reason.
        cue("error")
        notify("voice: failed", f"See {LOG} for what went wrong.")
        traceback.print_tb(exc.__traceback__)
        print(type(exc).__name__, file=sys.stderr)
        return 1
    return 0
