"""Voice chat: the loop's decisions, the hook relay, the settings it hands Claude Code, and
the terminal it owns. The microphone, speaker and Claude Code itself are faked."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shlex
import socket
import sys
import threading
import time
from unittest import mock

import numpy as np

from support import TempState
from voice import chat, listen


class FakeEffects:
    """Records what the session asks for, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.load_ok = True

    def __getattr__(self, name):
        def record(*args):
            self.calls.append((name, *args))
            return self.load_ok if name == "load" else None
        return record

    def names(self) -> list[str]:
        return [c[0] for c in self.calls]


class TestSession(TempState):
    def setUp(self) -> None:
        super().setUp()
        self.fx = FakeEffects()
        self.session = chat.Session(self.fx)
        self.session.start()
        self.fx.calls.clear()

    def listening_generation(self) -> int:
        return [c for c in self.fx.calls if c[0] == "listen"][-1][1]

    def test_it_starts_on_and_listening(self) -> None:
        fx = FakeEffects()
        chat.Session(fx).start()
        self.assertEqual(fx.names(), ["load", "listen"])

    def test_a_reply_is_spoken_then_the_microphone_opens(self) -> None:
        self.session.handle(("prompt",))
        self.session.handle(("reply", "Done.\n<speak>I fixed the test.</speak>"))
        say = [c for c in self.fx.calls if c[0] == "say"][-1]
        self.assertEqual(say[1], "I fixed the test.")
        self.assertNotIn("listen", self.fx.names()[-1:])
        self.session.handle(("spoken", say[2]))
        self.assertEqual(self.fx.names()[-1], "listen")

    def test_a_reply_without_a_summary_goes_straight_to_listening(self) -> None:
        self.session.handle(("prompt",))
        self.session.handle(("reply", "No summary here."))
        self.assertNotIn("say", self.fx.names())
        self.assertEqual(self.fx.names()[-1], "listen")

    def test_what_was_heard_is_submitted(self) -> None:
        self.session.handle(("heard", self.listening_generation_after_reply(), "run the tests"))
        self.assertEqual(self.fx.calls[-1], ("submit", "run the tests"))

    def listening_generation_after_reply(self) -> int:
        self.session.handle(("prompt",))
        self.session.handle(("reply", "x"))
        return self.listening_generation()

    def test_silence_or_the_off_phrase_turns_the_loop_off(self) -> None:
        for heard in ("", "Voice chat off.", "stop listening"):
            session = chat.Session(fx := FakeEffects())
            session.start()
            gen = [c for c in fx.calls if c[0] == "listen"][-1][1]
            session.handle(("heard", gen, heard))
            self.assertFalse(session.on, heard)
            self.assertNotIn("submit", fx.names())
            self.assertIn("unload", fx.names())

    def test_a_late_result_from_a_cancelled_listen_is_dropped(self) -> None:
        gen = self.listening_generation_after_reply()
        self.assertEqual(self.session.keys(b"a"), b"a")  # typing takes over
        self.assertIn("stop_listening", self.fx.names())
        self.session.handle(("heard", gen, "too late"))
        self.assertNotIn("submit", self.fx.names())
        self.assertTrue(self.session.on)  # typing ends this turn's listening, not the loop

    def test_terminal_reports_are_not_typing(self) -> None:
        # Answers to Claude Code's own queries, and focus changes, arrive on the same input.
        reports = (b"\x1b[I", b"\x1b[O", b"\x1b[?1u", b"\x1b[?62;22c", b"\x1b[12;40R", b"\x1b[0n",
                   b"\x1b]11;rgb:0000/0000/0000\x1b\\", b"\x1b]10;rgb:ffff/ffff/ffff\x07", b"\x1bP>|iTerm2 3.5\x1b\\")
        for report in reports:
            self.assertEqual(self.session.keys(report), report)  # still passed on to Claude
        self.assertNotIn("stop_listening", self.fx.names())
        self.session.keys(b"\x1b[I" + b"x")
        self.assertIn("stop_listening", self.fx.names())

    def test_a_key_while_speaking_stops_the_speech_and_still_reaches_claude(self) -> None:
        self.session.handle(("prompt",))
        self.session.handle(("reply", "<speak>Long summary.</speak>"))
        gen = [c for c in self.fx.calls if c[0] == "say"][-1][2]
        self.assertEqual(self.session.keys(b"y"), b"y")
        self.assertIn("stop_speech", self.fx.names())
        self.session.handle(("spoken", gen))
        self.assertNotEqual(self.fx.names()[-1], "listen")

    def test_control_q_toggles_and_is_not_passed_on(self) -> None:
        for key in (b"\x11", b"\x1b[113;5u"):
            session = chat.Session(fx := FakeEffects())
            session.start()
            self.assertEqual(session.keys(b"a" + key + b"b"), b"ab")
            self.assertFalse(session.on)
            self.assertEqual(session.keys(key), b"")
            self.assertTrue(session.on)
            self.assertEqual(fx.names()[-1], "listen")  # Claude is idle, so the microphone opens

    def test_the_toggle_event_from_the_global_shortcut_acts_like_control_q(self) -> None:
        self.session.handle(("toggle",))
        self.assertFalse(self.session.on)
        self.session.handle(("toggle",))
        self.assertTrue(self.session.on)
        self.assertEqual(self.fx.names()[-1], "listen")

    def test_turning_on_opens_the_microphone_even_while_claude_works(self) -> None:
        # Claude Code queues a message sent mid-turn, and a turn that ended without a Stop
        # event would otherwise leave no way to talk.
        self.session.keys(b"\x11")
        self.session.handle(("prompt",))
        self.fx.calls.clear()
        self.session.keys(b"\x11")
        self.assertEqual(self.fx.names(), ["load", "cue", "listen"])

    def test_a_reply_that_lands_while_the_user_talks_leaves_the_microphone_alone(self) -> None:
        self.session.handle(("reply", "<speak>Hi.</speak>"))  # from start(), listening already
        self.assertNotIn("say", self.fx.names())
        self.assertNotIn("stop_listening", self.fx.names())

    def test_a_reply_with_no_prompt_before_it_is_ignored(self) -> None:
        # A lone forged Stop on the event socket does nothing.
        gen = self.listening_generation_after_reply()
        self.session.handle(("heard", gen, "run the tests"))
        self.fx.calls.clear()
        self.session.handle(("reply", "<speak>Forged.</speak>"))
        self.assertEqual(self.fx.calls, [])

    def test_turning_on_under_memory_pressure_stays_off(self) -> None:
        self.session.keys(b"\x11")
        self.fx.load_ok = False
        self.session.keys(b"\x11")
        self.assertFalse(self.session.on)
        self.assertNotIn("listen", self.fx.names())

    def test_when_off_replies_are_neither_spoken_nor_answered(self) -> None:
        self.session.keys(b"\x11")
        self.fx.calls.clear()
        self.session.handle(("reply", "<speak>Hi.</speak>"))
        self.assertEqual(self.fx.calls, [])

    def test_a_prompt_cancels_speech_and_listening(self) -> None:
        self.session.handle(("prompt",))  # typed while the microphone was open
        self.assertIn("stop_listening", self.fx.names())
        self.session.handle(("reply", "<speak>Hi.</speak>"))
        self.session.handle(("prompt",))  # typed while the summary played
        self.assertIn("stop_speech", self.fx.names())

    def test_a_permission_request_is_spoken_only_while_claude_works(self) -> None:
        self.session.handle(("notice", "Claude needs your permission to use Bash"))
        self.assertNotIn("say", self.fx.names())  # idle: the microphone is open, so stay quiet
        self.session.handle(("prompt",))
        self.session.handle(("notice", "Claude needs your permission to use Bash"))
        self.assertEqual(self.fx.calls[-1][:2], ("say", "Claude needs your permission to use Bash"))


class TestSummary(TempState):
    def test_only_a_closing_speak_line_counts(self) -> None:
        self.assertEqual(chat.spoken("text\n<speak>All **done**, see `x_y`.</speak>\n"), "All done, see x_y.")
        self.assertEqual(chat.spoken("<speak>quoted</speak> then more text"), "")
        self.assertEqual(chat.spoken(""), "")

    def test_off_phrases(self) -> None:
        for text in ("Voice chat off.", "stop listening!", "  STOP LISTENING "):
            self.assertTrue(chat.off_phrase(text), text)
        self.assertFalse(chat.off_phrase("stop listening to the old socket"))


class TestHookRelay(TempState):
    def test_settings_add_three_hooks_that_run_this_python_in_isolated_mode(self) -> None:
        settings = chat.settings()
        self.assertEqual(sorted(settings["hooks"]), ["Notification", "Stop", "UserPromptSubmit"])
        command = settings["hooks"]["Stop"][0]["hooks"][0]["command"]
        self.assertEqual(shlex.split(command), [sys.executable, "-I", "-m", "voice.cli", "chat", "--hook"])

    def test_the_hook_relays_the_reply_to_the_session_socket(self) -> None:
        received: list[dict] = []
        path = chat.event_socket_path()
        server = chat.listen_for_events(path, received.append)
        self.addCleanup(server.close)
        payload = {"hook_event_name": "Stop", "last_assistant_message": "<speak>Hi.</speak>"}
        with mock.patch.dict(os.environ, {"VOICE_CHAT_SOCKET": str(path)}), \
                mock.patch("sys.stdin", io.StringIO(json.dumps(payload))):
            self.assertEqual(chat.hook(), 0)
        deadline = time.monotonic() + 5
        while not received and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(received, [("reply", "<speak>Hi.</speak>")])

    def test_the_hook_ignores_a_socket_outside_the_state_folder(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_CHAT_SOCKET": "/tmp/elsewhere.sock"}), \
                mock.patch("sys.stdin", io.StringIO('{"hook_event_name": "Stop"}')), \
                mock.patch.object(socket, "socket") as sock:
            self.assertEqual(chat.hook(), 0)
        sock.assert_not_called()

    def test_the_hook_never_fails_its_session(self) -> None:
        for env, stdin in (({}, "{}"), ({"VOICE_CHAT_SOCKET": str(chat.event_socket_path())}, "not json")):
            with mock.patch.dict(os.environ, env), mock.patch("sys.stdin", io.StringIO(stdin)):
                self.assertEqual(chat.hook(), 0)

    def test_events_from_the_socket(self) -> None:
        self.assertEqual(chat.event({"hook_event_name": "UserPromptSubmit"}), ("prompt",))
        self.assertEqual(chat.event({"hook_event_name": "Stop", "last_assistant_message": 3}), ("reply", ""))
        self.assertEqual(chat.event({"hook_event_name": "Notification", "message": "x"}), ("notice", "x"))
        self.assertEqual(chat.event({"hook_event_name": "Toggle"}), ("toggle",))
        self.assertIsNone(chat.event({"hook_event_name": "Other"}))
        self.assertIsNone(chat.event([]))

    def test_toggle_reaches_the_newest_running_session(self) -> None:
        old: list[tuple] = []
        new: list[tuple] = []
        older = chat.listen_for_events(chat.state.STATE / "chat-1.sock", old.append)
        self.addCleanup(older.close)
        newer = chat.listen_for_events(chat.state.STATE / "chat-2.sock", new.append)
        self.addCleanup(newer.close)
        os.utime(older.path, (1, 1))
        (chat.state.STATE / "chat-3.sock").touch()  # left behind by a session that crashed
        os.utime(chat.state.STATE / "chat-3.sock", None)
        self.assertEqual(chat.toggle_running(), 0)
        deadline = time.monotonic() + 5
        while not new and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(new, [("toggle",)])
        self.assertEqual(old, [])

    def test_toggle_with_no_session_says_so(self) -> None:
        with mock.patch.object(chat.listen, "cue") as cue, mock.patch.object(chat.listen, "notify") as notify:
            self.assertEqual(chat.toggle_running(), 1)
        cue.assert_called_once_with("error")
        notify.assert_called_once()


class TestRun(TempState):
    def test_a_first_run_makes_the_state_folder_before_it_opens_the_log(self) -> None:
        class Reached(Exception):
            pass

        tty = mock.Mock(isatty=lambda: True, fileno=lambda: 0)
        before = set(os.listdir("/dev/fd"))
        with mock.patch("sys.stdin", tty), mock.patch("sys.stdout", tty), \
                mock.patch.object(chat.shutil, "which", return_value="/usr/bin/true"), \
                mock.patch.multiple(chat.models, check_pressure=mock.DEFAULT, require=mock.DEFAULT), \
                mock.patch.object(chat.termios, "tcgetattr"), \
                mock.patch.object(chat.os, "dup", side_effect=Reached):  # the step after the log opens
            self.assertFalse(self.state.exists())
            with self.assertRaises(Reached):
                chat.run([])
        for fd in set(os.listdir("/dev/fd")) - before:  # the pipe and the log, which run has not closed yet
            with contextlib.suppress(OSError):
                os.close(int(fd))
        self.assertTrue((self.state / "chat.log").exists())


class TestTerminal(TempState):
    def test_paste_is_bracketed_when_the_app_asked_for_it_and_never_carries_controls(self) -> None:
        self.assertEqual(chat.paste_bytes("hi\x1b[201~ there\n", bracketed=True), b"\x1b[200~hi [201~ there\x1b[201~")
        self.assertEqual(chat.paste_bytes("hi\nthere", bracketed=False), b"hi there")

    def test_a_transcript_can_never_start_a_command(self) -> None:
        # Claude Code runs a pasted leading / as a slash command even inside a bracketed paste.
        for text in ("/clear everything", "! rm -rf x", " #note", "/!/# hi"):
            self.assertFalse(chat.paste_bytes(text, bracketed=False)[:1] in b"/!#", text)

    def test_a_child_gets_the_typed_text_and_submits_it(self) -> None:
        """Runs a small program in the pseudo-terminal in place of Claude Code."""
        child = [sys.executable, "-c", "import sys; print('got:' + input(), flush=True)"]
        terminal = chat.Terminal(child, os.environ.copy())
        self.addCleanup(terminal.close)
        terminal.submit("hello from voice")
        output = b""
        deadline = time.monotonic() + 10
        while b"got:" not in output and time.monotonic() < deadline:
            output += terminal.read(1.0)
        self.assertIn(b"got:hello from voice", output)
        self.assertEqual(terminal.wait(), 0)

    def test_closing_ends_a_program_that_ignores_the_hangup_and_keeps_writing(self) -> None:
        # Nobody reads its output any more, so its writes block; a kill alone left such a
        # process stuck while exiting, waiting for the terminal to drain.
        child = [sys.executable, "-c", "import signal, sys; signal.signal(signal.SIGHUP, signal.SIG_IGN); "
                 "print('ready', flush=True)\nwhile True:\n    try: sys.stdout.write('x' * 4096); sys.stdout.flush()\n"
                 "    except OSError: pass"]
        terminal = chat.Terminal(child, os.environ.copy())
        output = b""
        deadline = time.monotonic() + 10
        while b"ready" not in output and time.monotonic() < deadline:
            output += terminal.read(1.0)
        started = time.monotonic()
        terminal.close(grace=0.5)
        self.assertLess(time.monotonic() - started, 5)
        self.assertEqual(terminal.pid, 0)


class TestRecordPatience(TempState):
    def test_no_speech_within_the_patience_ends_the_recording(self) -> None:
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.side_effect = lambda n: (np.zeros((n, 1), "float32"), False)
        sd = mock.Mock(InputStream=mock.Mock(return_value=stream))
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            audio = listen.record(60, 1.5, patience=1.0)
        self.assertLessEqual(len(audio), int(listen.RATE * 1.05))

    def test_record_stops_when_told(self) -> None:
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.side_effect = lambda n: (np.ones((n, 1), "float32"), False)
        sd = mock.Mock(InputStream=mock.Mock(return_value=stream))
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            listen.record(60, None, stop=stop.is_set)
