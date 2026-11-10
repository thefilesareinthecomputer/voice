"""`voice serve`: the resident speaker. It loads Kokoro once, builds its phoneme pipeline
with one silent sentence, then plays the text `voice speak` sends it over a socket in
the state folder. A cold `voice speak` takes about 4 s to first sound; through the
server it takes about 0.3 s.

It runs only while something asks for it: `voice speak --warm` starts it in the
background and `voice speak --release` stops it and frees the model. The Claude Code
hook ties both to `/voice on` and `/voice off`.

One request at a time plays. A newer one, or a stop, ends the one playing, and the
client of the ended request is told it is done, so `voice speak` keeps its old
behavior: it returns when its own text has finished or been cut off.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
from typing import Any

from voice import models, speak, state

SOCKET = state.STATE / "speak.sock"
PIDFILE = state.STATE / "serve.pid"
LOG = state.STATE / "serve.log"
MAX_REQUEST = 1_000_000   # bytes; `speak` sends at most 100,000 characters
STARTUP_WAIT = 30.0       # seconds a client waits for a server that is still loading
WARM_TEXT = "Ready."


class Speaker:
    """Plays one request at a time from the loaded model. `audio` is sounddevice, or a test's fake."""

    def __init__(self, model: Any, audio: Any) -> None:
        self.model, self.audio = model, audio
        self.current: tuple[threading.Thread, threading.Event] | None = None
        # Held while a chunk starts or a stop lands, so a stop can never fall between
        # the worker's check and its next play.
        self.guard = threading.Lock()

    def warm(self) -> None:
        """Make one sentence with the default voice and throw it away. Kokoro builds its phoneme
        pipeline on first use, and that build is most of a cold call's wait."""
        voice, speed, lang = speak.settings()
        for _ in self.model.generate(WARM_TEXT, voice=speak.voice_file(voice), speed=speed, lang_code=lang):
            pass

    def say(self, conn: Any, text: str, voice_path: str, speed: float, lang: str) -> None:
        self.stop()
        cancel = threading.Event()
        worker = threading.Thread(target=self._play, args=(conn, text, voice_path, speed, lang, cancel), daemon=True)
        self.current = (worker, cancel)
        worker.start()

    def playing(self) -> bool:
        return self.current is not None and self.current[0].is_alive()

    def stop(self) -> None:
        """End the request playing, if any, and wait for its worker, so two never use the model at once."""
        if self.current is None:
            return
        worker, cancel = self.current
        with self.guard:
            cancel.set()
            self.audio.stop()
        worker.join()
        self.current = None

    def _play(self, conn: Any, text: str, voice_path: str, speed: float, lang: str, cancel: threading.Event) -> None:
        try:
            speak.render(self.model, text, voice_path, speed, lang, self.audio, cancel, self.guard)
        except Exception as exc:  # the log gets the type and stack, never the text
            traceback.print_tb(exc.__traceback__)
            print(type(exc).__name__, file=sys.stderr)
        finally:
            answer(conn, "done")


def answer(conn: Any, reply: str) -> None:
    with contextlib.suppress(OSError):
        conn.sendall(reply.encode() + b"\n")
    with contextlib.suppress(OSError):
        conn.close()


def handle(speaker: Speaker, conn: Any, message: Any) -> bool:
    """Act on one request. Returns False when the server should exit. A `say` hands the
    connection to the speaker, which answers when playback ends; every other request is
    answered here."""
    op = message.get("op") if isinstance(message, dict) else None
    if op == "ping":
        answer(conn, "ok")
    elif op == "playing":
        answer(conn, "yes" if speaker.playing() else "no")
    elif op == "stop":
        speaker.stop()
        answer(conn, "ok")
    elif op == "quit":
        speaker.stop()
        answer(conn, "ok")
        return False
    elif op == "say":
        text, voice, speed = message.get("text"), message.get("voice"), message.get("speed")
        low, high = speak.SPEED_RANGE
        if not (isinstance(text, str) and isinstance(voice, str) and isinstance(speed, int | float)
                and math.isfinite(speed) and low <= speed <= high):
            answer(conn, "error: a say request needs text, a voice and a speed")
            return True
        try:
            path = speak.voice_file(voice)
        except (SystemExit, models.ModelMissing) as exc:
            answer(conn, f"error: {exc}")
            return True
        speaker.say(conn, text, path, float(speed), voice[:1] or "a")
    else:
        answer(conn, "error: unknown request")
    return True


def read_line(conn: socket.socket) -> bytes:
    """One request: everything up to the first newline, or up to the client's end of input."""
    data = b""
    while b"\n" not in data and len(data) <= MAX_REQUEST:
        chunk = conn.recv(65536)
        if not chunk:
            break
        data += chunk
    return data.split(b"\n", 1)[0]


def bind() -> socket.socket:
    """Listen on the socket, private to this user. Only the holder of the pid lock calls this,
    so a socket file already there is a dead server's and safe to replace."""
    with contextlib.suppress(FileNotFoundError):
        SOCKET.unlink()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    old = os.umask(0o177)
    try:
        sock.bind(str(SOCKET))
    finally:
        os.umask(old)
    sock.listen(8)
    return sock


def serve_forever(sock: socket.socket, speaker: Speaker) -> None:
    try:
        while True:
            conn, _ = sock.accept()
            conn.settimeout(5)  # for the request; a say then waits as long as playback takes
            try:
                message = json.loads(read_line(conn))
            except (OSError, ValueError):
                answer(conn, "error: unreadable request")
                continue
            conn.settimeout(None)
            if not handle(speaker, conn, message):
                return
    finally:
        sock.close()
        with contextlib.suppress(FileNotFoundError):
            SOCKET.unlink()


def run(force: bool = False) -> int:
    """`voice serve`: load, warm, then serve until a quit request or a signal."""
    lock = state.hold(PIDFILE)
    if lock is None:
        print("voice serve: already running", file=sys.stderr)
        return 0
    try:
        models.check_pressure(force)
        import sounddevice as sd

        model = models.load_tts(models.tts_model())
        speaker = Speaker(model, sd)
        try:
            speaker.warm()
        except (Exception, SystemExit) as exc:  # a bad VOICE_NAME shows up on the first request instead
            print(f"voice serve: warm-up skipped ({type(exc).__name__})", file=sys.stderr)
        sock = bind()
        state.write_pid(lock, os.getpid())  # ready: clients stop waiting and connect
        serve_forever(sock, speaker)
    finally:
        os.close(lock)
        models.release()
    return 0


def request(message: dict, timeout: float | None = 5.0) -> str | None:
    """Send one request to the running server and return its reply, or None when no server answers.
    `timeout` bounds the wait for the reply; None waits as long as playback takes."""
    if state.holder(PIDFILE) is None:
        return None
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.settimeout(5)
        sock.connect(str(SOCKET))
        sock.sendall(json.dumps(message).encode() + b"\n")
        sock.settimeout(timeout)
        return read_line(sock).decode(errors="replace")
    except OSError:
        return None
    finally:
        sock.close()


def ready(wait: float = STARTUP_WAIT) -> bool:
    """True once a server is ready. A server that holds the lock but has not written its pid is
    still loading; this waits for it, and gives up when it exits or after `wait` seconds."""
    deadline = time.monotonic() + wait
    while state.holder(PIDFILE) is None:
        if not state.locked(PIDFILE) or time.monotonic() > deadline:
            return False
        time.sleep(0.1)
    return True


def say(text: str, voice: str, speed: float) -> str | None:
    """Speak through the server and wait until playback ends. None when no server is there, so
    the caller plays it itself. A Control-C stops the playback too."""
    if not ready():
        return None
    try:
        return request({"op": "say", "text": text, "voice": voice, "speed": speed}, timeout=None)
    except KeyboardInterrupt:
        request({"op": "stop"})
        raise


def start(force: bool = False) -> bool:
    """Start a server in the background unless one is running or starting. Returns whether it started one."""
    if state.locked(PIDFILE):
        return False
    state.folder()
    argv = [sys.executable, "-I", "-m", "voice.cli", "serve"] + (["--force"] if force else [])
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with open(fd, "a") as log:
        subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=log, stderr=log, cwd=state.STATE,
                         start_new_session=True)
    return True
