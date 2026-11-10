"""`voice chat`: a hands-free conversation with Claude Code in this terminal.

`voice chat` runs `claude` inside a pseudo-terminal it owns and passes the keyboard and
screen through untouched. When a reply ends, its closing summary line is spoken; once
playback ends the microphone opens, a pause ends your turn, and the text is typed into
Claude's prompt and submitted. Owning the terminal is what lets it type there without
an Accessibility or Automation grant.

    Control+Q         turn the loop off, or on again (it starts on)
    any other key     stop the speech or the listening in progress; the key goes to Claude
    "voice chat off"  said on its own, turns the loop off; so does 30 seconds of silence

Claude Code learns about the loop from two flags on this one session: `--settings` adds
hooks that report each prompt and reply over a private socket, and
`--append-system-prompt` asks for the summary line. Nothing in ~/.claude changes.

Parakeet stays loaded while the loop is on and is released when it is off. Speech goes
through the resident speaker, started for the session if it was not running.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pty
import queue
import re
import select
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import termios
import threading
import time
import traceback
import tty
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from voice import listen, models, serve, speak, state

LOG = state.STATE / "chat.log"
SILENCE = 3.0          # seconds of quiet that end your turn, unless --silence says otherwise
PATIENCE = 30.0        # seconds with no speech before the loop turns itself off
MAX_TURN = 600.0       # the longest single turn, in seconds
SUBMIT_DELAY = 0.15    # between the pasted text and Enter, so Claude Code reads Enter as a submit
TOGGLE_KEYS = (b"\x11", b"\x1b[113;5u")  # Control+Q, plain and in the kitty keyboard protocol
INTERRUPT_KEYS = (b"\x1b", b"\x1b[27u", b"\x03", b"\x1b[99;5u")  # Escape, Control+C, the same two ways
ENTER_KEYS = (b"\r", b"\x1b[13u")
RESUME_SECONDS = 2.0   # after a key that ends a turn with no event, before the microphone reopens;
                       # long enough for a real prompt's hook to arrive first and cancel it
# What the terminal sends on its own rather than for a key: focus changes, the mouse (Claude
# Code turns on reporting, so even moving the pointer over the window sends these), and answers
# to Claude Code's queries (keyboard protocol, device attributes, cursor position, status, color
# scheme, modes, window size, colors, version, graphics). These pass through without counting
# as typing.
REPORTS = re.compile(rb"\x1b\[[IO]"                       # focus in, out
                     rb"|\x1b\[<\d+;\d+;\d+[Mm]"          # mouse, SGR encoding
                     rb"|\x1b\[M[\x20-\xff]{3}"           # mouse, legacy encoding
                     rb"|\x1b\[\d+;\d+;\d+[Mt]"           # mouse in urxvt encoding; window size
                     rb"|\x1b\[[?>=][\d;]*[uc]"           # keyboard protocol; device attributes
                     rb"|\x1b\[\??[\d;]+[Rn]"             # cursor position; status, color scheme
                     rb"|\x1b\[\??[\d;]+\$y"              # mode
                     rb"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)"  # colors and other OSC answers
                     rb"|\x1b[P_][^\x1b]*\x1b\\")         # version (DCS); graphics (APC)
OFF_PHRASES = {"voice chat off", "stop listening"}
SOUNDS = {"on": "/System/Library/Sounds/Glass.aiff", "off": "/System/Library/Sounds/Bottle.aiff"}
SOUND_SECONDS = 1.7    # the on sound lasts 1.65 s; afplay itself stays up about a second longer

INSTRUCTION = (
    "The user is talking with you by voice through `voice chat`: they hear a spoken summary "
    "of each reply instead of reading it, and their messages are speech transcripts, so read "
    "past small transcription errors. End every reply with one line <speak>...</speak>: one "
    "to three plain spoken sentences saying what you did or found and what you need from the "
    "user next. If you need a decision, ask for it there. No file paths, code, commands, URLs "
    "or symbols in it; say them in words or say they are on screen."
)

# The last <speak> line, and only when nothing but whitespace follows it.
SPEAK = re.compile(r"<speak>((?:(?!<speak>).)*?)</speak>\s*\Z", re.DOTALL | re.IGNORECASE)
MARKS = re.compile(r"[*`#>]+|(?<!\w)_+|_+(?!\w)")
CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MAX_SPOKEN = 400


def spoken(text: str) -> str:
    """The summary line that ends a reply, cleaned for speech, or "" when there is none."""
    found = SPEAK.search(text)
    if not found:
        return ""
    return " ".join(CONTROL.sub(" ", MARKS.sub("", found.group(1))).split())[:MAX_SPOKEN]


def off_phrase(text: str) -> bool:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split()) in OFF_PHRASES


class Session:
    """The loop's decisions, kept apart from the devices: `fx` does the work. Every method runs
    on the main thread. A listen or a playback reports back with the generation it started
    under, and a result from a generation that has since been cancelled is dropped."""

    def __init__(self, fx: Any) -> None:
        self.fx = fx
        self.on = False
        self.busy = False       # Claude is working on a prompt
        self.phase = "idle"     # or "speaking" or "listening"
        self.generation = 0

    def start(self) -> None:
        if self.fx.load():
            self.on = True
            self.fx.claim()
            self._listen()

    def handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "toggle":
            self.toggle()
        elif kind == "off":  # another session's loop came on: one microphone loop at a time
            if self.on:
                self._turn_off()
                self.fx.notify("voice chat: moved", "Another voice chat is listening now. "
                               "This one is plain typing; press Control+Q here to take it back.")
        elif kind == "prompt":
            self.busy = True
            self._cancel()
        elif kind == "reply":
            # No prompt event needs to come first: Claude starts some turns itself (a background
            # task or another agent finishing), and their replies are spoken too.
            self.busy = False
            if self.on and self.phase != "listening":  # never talk over the user
                self._reply(event[1])
        elif kind == "notice":
            # A permission request mid-turn. While idle the microphone may be open, so stay quiet.
            if self.on and self.busy and self.phase == "idle":
                self._say(event[1])
        elif kind == "spoken":
            if event[1] == self.generation and self.phase == "speaking":
                self.phase = "idle"
                if self.on and not self.busy:
                    self._listen()
        elif kind == "heard" and event[1] == self.generation and self.phase == "listening":
            self.phase = "idle"
            self._heard(event[2])
        elif kind == "resume" and event[1] == self.generation and self.on and not self.busy and self.phase == "idle":
            self._listen()

    def keys(self, data: bytes) -> bytes:
        """Keyboard input: returns what goes on to Claude. Control+Q toggles the loop and is kept
        back; any other key stops the speech or listening in progress and goes through.

        Two key presses end a turn that Claude Code reports nothing for, so the microphone
        reopens on its own shortly after: Escape or Control+C interrupting Claude (no Stop event
        follows), and Enter on a line that starts no turn, such as /clear (no prompt event)."""
        toggles = 0
        for key in TOGGLE_KEYS:
            toggles += data.count(key)
            data = data.replace(key, b"")
        typed = REPORTS.sub(b"", data)
        if self.phase != "idle" and typed:
            self._cancel()
        if self.on and self.busy and typed in INTERRUPT_KEYS:
            self.busy = False
            self.fx.later(RESUME_SECONDS, ("resume", self.generation))
        elif self.on and not self.busy and typed.endswith(ENTER_KEYS):
            self.fx.later(RESUME_SECONDS, ("resume", self.generation))
        for _ in range(toggles):
            self.toggle()
        return data

    def toggle(self) -> None:
        if self.on:
            self._turn_off()
        elif self.fx.load():
            self.on = True
            self.fx.claim()
            self.fx.cue("on")
            self._listen()

    def _reply(self, text: str) -> None:
        line = spoken(text)
        if line:
            self._say(line)
        else:
            self._listen()

    def _say(self, line: str) -> None:
        self.generation += 1
        self.phase = "speaking"
        self.fx.say(line, self.generation)

    def _listen(self) -> None:
        self.generation += 1
        self.phase = "listening"
        self.fx.listen(self.generation)

    def _heard(self, text: str | None) -> None:
        if text is None:  # the microphone or the model failed; the error sound has played
            self._turn_off()
            self.fx.notify("voice chat: off", "The microphone or the speech model failed; the error is in "
                           "~/.local/state/claude-voice/chat.log. Press Control+Q to try again.")
        elif not text.strip():
            self._turn_off(paused=True)
        elif off_phrase(text):
            self._turn_off()
        else:
            self.fx.submit(text.strip())

    def _cancel(self) -> None:
        if self.phase == "speaking":
            self.fx.stop_speech()
        elif self.phase == "listening":
            self.fx.stop_listening()
        self.phase = "idle"
        self.generation += 1

    def _turn_off(self, paused: bool = False) -> None:
        self._cancel()
        self.on = False
        self.fx.unload()
        self.fx.cue("off")
        if paused:
            self.fx.notify("voice chat: paused", "Nothing was said. Press Control+Q to talk again.")


class Devices:
    """The real effects: Parakeet, the microphone, the speaker and Claude's terminal. Work that
    takes time runs on a thread and reports back through `post`."""

    def __init__(self, post: Callable[[tuple], None], force: bool, silence: float = SILENCE) -> None:
        self.post, self.force, self.silence = post, force, silence
        self.terminal: Terminal | None = None
        self.pool: ThreadPoolExecutor | None = None
        self.model: Future | None = None
        self.stopped = threading.Event()
        self.sound: subprocess.Popen | None = None  # the last on or off sound
        self.own: Path | None = None  # this session's event socket
        self.started_speaker = False  # restarted the speaker mid-session, so release it at the end
        self.closing = False
        self.stopping: threading.Thread | None = None  # the last stop sent to the speaker

    def load(self) -> bool:
        if self.model is not None:
            return True
        try:
            models.check_pressure(self.force)
        except models.MemoryPressure:
            listen.cue("error")
            listen.notify("voice chat: off", "Memory pressure is critical. Close something, then press Control+Q.")
            return False
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.model = self.pool.submit(models.load_stt, models.stt_model())
        return True

    def unload(self) -> None:
        pool, self.pool, self.model = self.pool, None, None
        if pool is not None:
            # Off the main thread: a load in progress finishes first, then the buffers go.
            threading.Thread(target=lambda: (pool.shutdown(wait=True), models.release()), daemon=True).start()

    def say(self, line: str, generation: int) -> None:
        stopping = self.stopping

        def work() -> None:
            try:
                if stopping is not None:  # a stop still on its way must not cut this summary off
                    stopping.join(timeout=5)
                # A speaker that died mid-session comes back, rather than every reply loading
                # the voice model in this process; never once the session is closing.
                if not self.closing and serve.start(self.force):
                    self.started_speaker = True
                speak.run(line)
            except (Exception, SystemExit) as exc:
                log(exc)
            self.post(("spoken", generation))
        threading.Thread(target=work, daemon=True).start()

    def later(self, seconds: float, event: tuple) -> None:
        timer = threading.Timer(seconds, self.post, args=(event,))
        timer.daemon = True
        timer.start()

    def claim(self) -> None:
        threading.Thread(target=release_others, args=(self.own,), daemon=True).start()

    def stop_speech(self) -> None:
        # Off the main thread: a speaker that is slow to answer must not freeze the terminal.
        def work() -> None:
            with contextlib.suppress(Exception, SystemExit):
                speak.run(None, stop=True)
        self.stopping = threading.Thread(target=work, daemon=True)
        self.stopping.start()

    def listen(self, generation: int) -> None:
        self.stopped = stopped = threading.Event()
        model, sound = self.model, self.sound

        def work() -> None:
            text: str | None = ""
            try:
                quiet(stopped)  # never record the speaker
                if sound is not None:  # nor the on sound, which would count as speech
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        sound.wait(timeout=SOUND_SECONDS)
                if stopped.is_set():
                    return
                # A click, a cough or room noise followed by a pause ends a recording with no words
                # in it. That is not the user's turn: listen on, quietly, until PATIENCE runs out.
                started = time.monotonic()
                on_open: Callable[[], Any] | None = lambda: listen.cue("start")  # noqa: E731
                while True:
                    left = max(0.0, PATIENCE - (time.monotonic() - started))
                    audio = listen.record(MAX_TURN, self.silence, stop=stopped.is_set, on_open=on_open, patience=left)
                    if stopped.is_set():
                        return
                    if listen.peak(audio) > listen.SPEECH_LEVEL and model is not None:
                        text = listen.transcribe_with(model.result(), audio)
                    if text.strip() or model is None or time.monotonic() - started >= PATIENCE:
                        break
                    on_open = None
                if text.strip():
                    listen.cue("stop")
            except Exception as exc:
                listen.cue("error")
                log(exc)
                text = None  # not silence: the session says what failed
            self.post(("heard", generation, text))
        threading.Thread(target=work, daemon=True).start()

    def stop_listening(self) -> None:
        self.stopped.set()

    def submit(self, text: str) -> None:
        if self.terminal is not None:
            self.terminal.submit(text)

    def cue(self, kind: str) -> None:
        with contextlib.suppress(OSError):
            self.sound = subprocess.Popen([listen.AFPLAY, SOUNDS[kind]],
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def notify(self, title: str, message: str) -> None:
        listen.notify(title, message)


def quiet(stopped: threading.Event, wait: float = 60.0) -> None:
    """Wait while the resident speaker is playing anything, such as a reply another hook started."""
    deadline = time.monotonic() + wait
    while not stopped.is_set() and time.monotonic() < deadline and serve.request({"op": "playing"}) == "yes":
        time.sleep(0.1)


def log(exc: BaseException) -> None:
    """The error's type and stack, never its message, which could quote what was said."""
    traceback.print_tb(exc.__traceback__, file=sys.stderr)
    print(type(exc).__name__, file=sys.stderr)


def paste_bytes(text: str, bracketed: bool) -> bytes:
    """The transcript as terminal input, with no control characters, so it can never press a key,
    and no leading /, ! or #, which Claude Code reads as a command or a mode switch even when
    pasted. Bracketed, when the app asked for it."""
    clean = " ".join(CONTROL.sub(" ", text).split()).lstrip("/!# ").encode()
    return b"\x1b[200~" + clean + b"\x1b[201~" if bracketed else clean


class Terminal:
    """A program running in a pseudo-terminal that this process owns."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        pid, fd = pty.fork()
        if pid == 0:  # the child: become the program, or exit as a shell would for a missing one
            try:
                os.execve(argv[0], argv, env)
            finally:
                os._exit(127)
        self.pid, self.fd = pid, fd
        self.bracketed = False
        self.eof = False

    def resize(self, source: int) -> None:
        with contextlib.suppress(OSError):
            size = fcntl.ioctl(source, termios.TIOCGWINSZ, b"\0" * 8)
            fcntl.ioctl(self.fd, termios.TIOCSWINSZ, size)

    def write(self, data: bytes) -> None:
        while data:
            data = data[os.write(self.fd, data):]

    def read(self, timeout: float | None = None) -> bytes:
        """What the program wrote, or b"" after `timeout` or once it has exited (then `eof` is set)."""
        if not select.select([self.fd], [], [], timeout)[0]:
            return b""
        try:
            data = os.read(self.fd, 65536)
        except OSError:  # EIO: the program has exited and closed its side
            data = b""
        if not data:
            self.eof = True
            return b""
        on, off = data.rfind(b"\x1b[?2004h"), data.rfind(b"\x1b[?2004l")
        if on != off:
            self.bracketed = on > off
        return data

    def submit(self, text: str) -> None:
        self.write(paste_bytes(text, self.bracketed))
        time.sleep(SUBMIT_DELAY)
        self.write(b"\r")

    def wait(self) -> int:
        _, status = os.waitpid(self.pid, 0)
        self.pid = 0
        return os.waitstatus_to_exitcode(status)

    def close(self, grace: float = 3.0) -> None:
        """End the program if it is still running. Closing this side first is the hangup a closed
        terminal sends, and it makes the program's pending writes fail rather than block, so it
        can exit; a kill follows if it is still there after `grace` seconds."""
        if self.fd >= 0:
            with contextlib.suppress(OSError):
                os.close(self.fd)
            self.fd = -1  # a second close must never hit a descriptor the number was reused for
        if self.pid:
            with contextlib.suppress(OSError):
                os.kill(self.pid, signal.SIGHUP)
            deadline = time.monotonic() + grace
            with contextlib.suppress(ChildProcessError):
                while os.waitpid(self.pid, os.WNOHANG) == (0, 0):
                    if time.monotonic() > deadline:
                        os.kill(self.pid, signal.SIGKILL)
                        os.waitpid(self.pid, 0)
                        break
                    time.sleep(0.05)
            self.pid = 0


# The hook side: Claude Code runs `voice chat --hook` for each event, and it forwards the
# event to the session over a socket whose path the session put in the environment.

def event_socket_path() -> Path:
    return state.STATE / f"chat-{os.getpid()}.sock"


def event(payload: Any) -> tuple | None:
    """The session event for a hook payload, or None for one the loop ignores."""
    if not isinstance(payload, dict):
        return None
    name = payload.get("hook_event_name")
    if name == "Toggle":  # from `voice chat --toggle`, not from Claude Code
        return ("toggle",)
    if name == "Off":  # from another session whose loop came on
        return ("off",)
    if name == "UserPromptSubmit":
        return ("prompt",)
    text = payload.get("last_assistant_message" if name == "Stop" else "message")
    text = text if isinstance(text, str) else ""
    if name == "Stop":
        return ("reply", text)
    if name == "Notification":
        return ("notice", " ".join(CONTROL.sub(" ", text).split())[:MAX_SPOKEN])
    return None


def hook() -> int:
    """`voice chat --hook`: forward one hook payload to the session. Never blocks or fails Claude Code."""
    with contextlib.suppress(Exception):
        path = Path(os.environ.get("VOICE_CHAT_SOCKET", ""))
        if path.parent != state.STATE or not path.name.startswith("chat-"):
            return 0  # only a session's own socket, in the private state folder
        payload = json.loads(sys.stdin.read(serve.MAX_REQUEST))
        if not isinstance(payload, dict):
            return 0
        keep = ("hook_event_name", "last_assistant_message", "message")
        message = json.dumps({k: payload[k] for k in keep if k in payload}).encode() + b"\n"
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            sock.connect(str(path))
            sock.sendall(message)
    return 0


def toggle_running() -> int:
    """`voice chat --toggle`: turn the loop of the newest running session off or on. For a global
    keyboard shortcut, which a terminal never passes on when it holds Command."""
    sockets = sorted(state.STATE.glob("chat-*.sock"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in sockets:  # a session that crashed leaves a socket nothing answers
        with contextlib.suppress(OSError), socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            sock.connect(str(path))
            sock.sendall(b'{"hook_event_name": "Toggle"}\n')
            return 0
    listen.cue("error")
    listen.notify("voice chat: not running", "Start voice chat in a terminal first.")
    return 1


def release_others(own: Path | None) -> None:
    """Tell every other running session to turn its loop off: two loops would both hear the user
    and both type what was said. A socket nothing answers, left by a crash, is skipped."""
    for path in state.STATE.glob("chat-*.sock"):
        if path == own:
            continue
        with contextlib.suppress(OSError), socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(2)
            sock.connect(str(path))
            sock.sendall(b'{"hook_event_name": "Off"}\n')


class EventServer:
    """Accepts hook connections on a thread and posts each event."""

    def __init__(self, path: Path, post: Callable[[tuple], None]) -> None:
        self.path = path
        state.folder()
        with contextlib.suppress(FileNotFoundError):
            path.unlink()
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o177)
        try:
            self.sock.bind(str(path))
        finally:
            os.umask(old)
        self.sock.listen(8)
        threading.Thread(target=self._accept, args=(post,), daemon=True).start()

    def _accept(self, post: Callable[[tuple], None]) -> None:
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return  # closed
            with conn:
                conn.settimeout(2)
                with contextlib.suppress(OSError, ValueError):
                    found = event(json.loads(serve.read_line(conn)))
                    if found is not None:
                        post(found)

    def close(self) -> None:
        self.sock.close()
        with contextlib.suppress(FileNotFoundError):
            self.path.unlink()


def listen_for_events(path: Path, post: Callable[[tuple], None]) -> EventServer:
    return EventServer(path, post)


def settings() -> dict:
    """The hooks this session adds to Claude Code through --settings."""
    command = " ".join(shlex.quote(part) for part in [sys.executable, "-I", "-m", "voice.cli", "chat", "--hook"])
    entry = [{"hooks": [{"type": "command", "command": command, "timeout": 5}]}]
    return {"hooks": {"UserPromptSubmit": entry, "Stop": entry, "Notification": entry}}


def run(claude_args: list[str], force: bool = False, silence: float = SILENCE) -> int:
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        print("voice chat: run it in a terminal; it shares the terminal with Claude Code", file=sys.stderr)
        return 2
    claude = shutil.which("claude")
    if claude is None:
        print("voice chat: claude is not on PATH; install Claude Code first", file=sys.stderr)
        return 2
    models.check_pressure(force)
    models.require(models.stt_model())
    models.require(models.tts_model())

    events: queue.Queue[tuple] = queue.Queue()
    wake_r, wake_w = os.pipe()

    def post(item: tuple) -> None:
        events.put(item)
        with contextlib.suppress(OSError):
            os.write(wake_w, b"e")

    path = event_socket_path()
    env = dict(os.environ, VOICE_CHAT_SOCKET=str(path))
    argv = [claude, "--settings", json.dumps(settings()), "--append-system-prompt", INSTRUCTION, *claude_args]
    stdin, stdout = sys.stdin.fileno(), sys.stdout.fileno()
    saved = termios.tcgetattr(stdin)
    resized = threading.Event()

    def on_resize(*_: Any) -> None:
        resized.set()
        with contextlib.suppress(OSError):
            os.write(wake_w, b"w")

    state.folder()  # a first run on a new machine has none yet
    log_fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    saved_err = os.dup(2)
    started_speaker = serve.start(force)
    fx = Devices(post, force, silence)
    fx.own = path
    server: EventServer | None = None
    # Fork before any thread starts. Claude Code runs the hooks only once a prompt arrives,
    # well after the event server below is listening.
    terminal = Terminal(argv, env)
    fx.terminal = terminal
    try:
        server = EventServer(path, post)
        session = Session(fx)
        signal.signal(signal.SIGWINCH, on_resize)
        os.dup2(log_fd, 2)  # errors from threads go to the log, never over Claude's screen
        tty.setraw(stdin)
        terminal.resize(stdin)
        session.start()
        while not terminal.eof:
            ready = select.select([terminal.fd, stdin, wake_r], [], [])[0]
            if terminal.fd in ready:
                data = terminal.read(0)
                while data:
                    data = data[os.write(stdout, data):]
            if stdin in ready:
                data = session.keys(os.read(stdin, 4096))
                if data:
                    terminal.write(data)
            if wake_r in ready:
                os.read(wake_r, 4096)
                if resized.is_set():
                    resized.clear()
                    terminal.resize(stdin)
                while not events.empty():
                    session.handle(events.get())
        return terminal.wait()
    finally:
        fx.closing = True
        termios.tcsetattr(stdin, termios.TCSAFLUSH, saved)
        signal.signal(signal.SIGWINCH, signal.SIG_DFL)
        fx.stop_listening()
        fx.unload()
        if server is not None:
            server.close()
        terminal.close()
        if started_speaker or fx.started_speaker:
            speak.run(None, release=True)
        os.dup2(saved_err, 2)
        os.close(saved_err)
        os.close(log_fd)
