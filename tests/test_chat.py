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

    def asked(self, session: chat.Session | None = None) -> None:
        """The user speaks a request: it is heard and submitted, and Claude Code reports the prompt."""
        session = session or self.session
        session.handle(("heard", session.generation, "do it"))
        session.handle(("prompt",))

    def test_it_starts_on_and_listening(self) -> None:
        fx = FakeEffects()
        chat.Session(fx).start()
        self.assertEqual(fx.names(), ["load", "claim", "listen"])  # claim: other loops turn off

    def test_a_reply_is_spoken_then_the_microphone_opens(self) -> None:
        self.asked()
        self.session.handle(("reply", "Done.\n<speak>I fixed the test.</speak>"))
        say = [c for c in self.fx.calls if c[0] == "say"][-1]
        self.assertEqual(say[1], "I fixed the test.")
        self.assertNotIn("listen", self.fx.names()[-1:])
        self.session.handle(("spoken", say[2]))
        self.assertEqual(self.fx.names()[-1], "listen")

    def test_a_reply_without_a_summary_goes_straight_to_listening(self) -> None:
        self.asked()
        self.session.handle(("reply", "No summary here."))
        self.assertNotIn("say", self.fx.names())
        self.assertEqual(self.fx.names()[-1], "listen")

    def test_what_was_heard_is_submitted(self) -> None:
        self.session.handle(("heard", self.listening_generation_after_reply(), "run the tests"))
        self.assertEqual(self.fx.calls[-1], ("submit", "run the tests"))

    def test_what_is_heard_while_claude_works_is_held_and_sent_when_the_turn_ends(self) -> None:
        """A turn another agent started may have a question dialog on screen, and text typed
        into a dialog is lost: nothing is typed until the turn ends."""
        self.session.handle(("prompt",))  # a turn the user did not start
        self.session.handle(("heard", self.session.generation, "a long message"))
        self.session.handle(("heard", self.session.generation, "ignored: the generation moved on"))
        self.assertNotIn("submit", self.fx.names())
        self.assertIn(("keep", "a long message"), self.fx.calls)
        self.assertIn("notify", self.fx.names())
        self.session.handle(("reply", "<speak>Done.</speak>"))
        self.assertEqual(self.fx.calls[-1], ("submit", "a long message"))
        self.assertNotIn("say", self.fx.names())  # the user's message goes first

    def test_a_held_message_goes_in_after_an_interrupted_turn(self) -> None:
        self.session.handle(("prompt",))
        self.session.handle(("heard", self.session.generation, "held words"))
        self.session.keys(b"\x1b")  # Escape: no reply event follows
        self.session.handle(self.fx.calls[-1][2])  # the resume timer fires
        self.assertEqual(self.fx.calls[-1], ("submit", "held words"))

    def test_the_copy_on_disk_is_kept_until_claude_code_reports_the_prompt(self) -> None:
        self.session.handle(("heard", self.session.generation, "do it"))
        self.assertEqual(self.fx.calls[-2:], [("keep", "do it"), ("submit", "do it")])
        self.assertNotIn("delivered", self.fx.names())
        self.session.handle(("prompt", "do it"))
        self.assertEqual(self.fx.calls[-1], ("delivered", "do it"))

    def test_a_prompt_that_is_not_the_typed_message_keeps_the_copy(self) -> None:
        """The text went into a dialog, and then another agent's message or the user's next
        typed prompt starts a turn: neither shows that Claude Code got what was said."""
        self.session.handle(("heard", self.session.generation, "what I said"))
        for other in (("prompt", "Another Claude session sent a message"), ("prompt", ""), ("prompt",),
                      ("prompt", "that is not what I said at all")):  # holds the words, is not the message
            self.session.handle(other)
        self.assertNotIn("delivered", self.fx.names())
        self.session.handle(("prompt", "  what I said\n"))
        self.assertEqual(self.fx.calls[-1], ("delivered", "what I said"))

    def test_a_question_or_permission_dialog_holds_what_is_said(self) -> None:
        for sign in (("dialog",), ("notice", "Claude needs your permission", True)):
            fx = FakeEffects()
            session = chat.Session(fx)
            session.start()
            session.handle(sign)  # no prompt event came first
            session.handle(("heard", session.generation, "more words"))
            self.assertNotIn("submit", fx.names())
            session.handle(("reply", "x"))
            self.assertEqual(fx.calls[-1], ("submit", "more words"))

    def test_a_notice_with_no_dialog_holds_nothing(self) -> None:
        self.session.handle(("notice", "Claude is waiting for your input", False))
        self.session.handle(("heard", self.session.generation, "hello"))
        self.assertEqual(self.fx.calls[-1], ("submit", "hello"))

    def test_the_unsent_file_is_private_and_loses_only_what_was_delivered(self) -> None:
        fx = chat.Devices(lambda event: None, force=True)
        chat.state.folder()
        for text in ("first", "second", "third"):
            fx.keep(text)
        path = chat.unsent_path()
        self.assertEqual(path.read_text(), "first\n\nsecond\n\nthird\n\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        fx.delivered("something else")
        self.assertEqual(path.read_text(), "first\n\nsecond\n\nthird\n\n")
        fx.delivered("second third")  # two messages held and sent as one
        self.assertEqual(path.read_text(), "first\n\n")
        fx.delivered("first")
        self.assertFalse(path.exists())
        fx.delivered("first")  # nothing to remove is not an error

    def listening_generation_after_reply(self) -> int:
        self.asked()
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
        self.fx.calls.clear()
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

    def test_the_mouse_and_other_reports_are_not_typing(self) -> None:
        # Claude Code turns on mouse reporting, so moving the pointer over the window sends input.
        reports = (b"\x1b[<35;10;5M", b"\x1b[<0;10;5M\x1b[<0;10;5m", b"\x1b[<64;3;4M",  # move, click, scroll
                   b"\x1b[M #!", b"\x1b[32;10;5M",                    # older mouse encodings
                   b"\x1b[>0;95;0c", b"\x1b[?2026;2$y", b"\x1b[?997;1n",  # attributes, mode, color scheme
                   b"\x1b[8;40;120t", b"\x1b_Gi=1;OK\x1b\\")          # window size, graphics
        for report in reports:
            self.assertEqual(self.session.keys(report), report)
        self.assertNotIn("stop_listening", self.fx.names())
        self.session.keys(b"\x1b[<35;11;5Mq")  # a key that arrives with a pointer move still counts
        self.assertIn("stop_listening", self.fx.names())

    def test_a_key_while_speaking_stops_the_speech_and_still_reaches_claude(self) -> None:
        self.asked()
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
        self.assertEqual(self.fx.names(), ["load", "claim", "cue", "listen"])

    def test_a_reply_that_lands_while_the_user_talks_leaves_the_microphone_alone(self) -> None:
        self.session.handle(("reply", "<speak>Hi.</speak>"))  # from start(), listening already
        self.assertNotIn("say", self.fx.names())
        self.assertNotIn("stop_listening", self.fx.names())

    def test_a_reply_to_a_turn_claude_started_itself_is_spoken(self) -> None:
        # A background task finishing starts a turn with no prompt event before its reply.
        gen = self.listening_generation_after_reply()
        self.session.handle(("heard", gen, "run the tests"))
        self.fx.calls.clear()
        self.session.handle(("reply", "<speak>The task finished.</speak>"))
        self.assertEqual(self.fx.calls[-1][:2], ("say", "The task finished."))

    def test_escape_mid_turn_reopens_the_microphone(self) -> None:
        # Claude Code sends no Stop for an interrupted turn.
        for key in (b"\x1b", b"\x03", b"\x1b[27u"):
            session = chat.Session(fx := FakeEffects())
            session.start()
            session.handle(("prompt",))
            fx.calls.clear()
            self.assertEqual(session.keys(key), key)  # Claude still gets the key
            self.assertFalse(session.busy)
            later = [c for c in fx.calls if c[0] == "later"]
            self.assertEqual(len(later), 1, key)
            session.handle(later[0][2])
            self.assertEqual(fx.names()[-1], "listen", key)

    def test_enter_on_a_line_that_starts_no_turn_reopens_the_microphone(self) -> None:
        self.session.keys(b"/")  # typing takes over from the listening
        self.session.keys(b"clear\r")
        resume = [c for c in self.fx.calls if c[0] == "later"][-1][2]
        self.session.handle(resume)  # /clear sends no prompt event
        self.assertEqual(self.fx.names()[-1], "listen")

    def test_enter_on_a_real_prompt_leaves_the_microphone_to_the_reply(self) -> None:
        self.session.keys(b"run the tests\r")
        resume = [c for c in self.fx.calls if c[0] == "later"][-1][2]
        self.session.handle(("prompt",))  # the prompt's hook arrives first
        self.fx.calls.clear()
        self.session.handle(resume)
        self.assertEqual(self.fx.calls, [])

    def test_a_microphone_or_model_failure_says_so(self) -> None:
        self.session.handle(("heard", self.session.generation, None))  # listening since start()
        self.assertFalse(self.session.on)
        notify = [c for c in self.fx.calls if c[0] == "notify"][-1]
        self.assertIn("failed", notify[2])

    def test_another_session_coming_on_turns_this_loop_off_and_says_so(self) -> None:
        self.session.handle(("off",))
        self.assertFalse(self.session.on)
        self.assertIn("stop_listening", self.fx.names())
        self.assertEqual(self.fx.calls[-1][0], "notify")
        self.fx.calls.clear()
        self.session.handle(("off",))  # already off: nothing more happens
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

    def test_a_turn_claude_starts_itself_cuts_off_neither_the_user_nor_the_summary(self) -> None:
        # A background task or another agent finishing starts a turn with a prompt event.
        self.session.handle(("prompt",))  # while the microphone is open
        self.assertNotIn("stop_listening", self.fx.names())
        self.asked()
        self.session.handle(("reply", "<speak>First.</speak>"))
        self.session.handle(("prompt",))  # while the summary plays
        self.assertNotIn("stop_speech", self.fx.names())

    def test_a_reply_that_lands_during_a_summary_is_spoken_after_it(self) -> None:
        self.asked()
        self.session.handle(("reply", "<speak>First.</speak>"))
        first = [c for c in self.fx.calls if c[0] == "say"][-1]
        self.session.handle(("reply", "<speak>Second.</speak>"))
        self.assertEqual([c[1] for c in self.fx.calls if c[0] == "say"], ["First."])  # not over it
        self.session.handle(("spoken", first[2]))
        self.assertEqual(self.fx.calls[-1][:2], ("say", "Second."))

    def test_control_q_on_a_loop_left_waiting_starts_listening(self) -> None:
        # An interruption can leave the loop on but neither speaking nor listening; turning it
        # off there would look like the press did nothing.
        self.asked()
        self.fx.calls.clear()
        self.session.keys(b"\x11")
        self.assertTrue(self.session.on)
        self.assertEqual(self.fx.names(), ["cue", "listen"])
        self.session.keys(b"\x11")  # now it is listening, so this press turns it off
        self.assertFalse(self.session.on)

    def test_a_permission_request_is_spoken_only_while_claude_works(self) -> None:
        self.session.handle(("notice", "Claude needs your permission to use Bash"))
        self.assertNotIn("say", self.fx.names())  # idle: the microphone is open, so stay quiet
        self.asked()
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
    def test_settings_add_hooks_that_run_this_python_in_isolated_mode(self) -> None:
        settings = chat.settings()
        self.assertEqual(sorted(settings["hooks"]), ["Notification", "PreToolUse", "Stop", "UserPromptSubmit"])
        self.assertEqual(settings["hooks"]["PreToolUse"][0]["matcher"], "AskUserQuestion|ExitPlanMode")
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

    def prompt_hook_output(self, loop_on: bool) -> str:
        path = chat.event_socket_path()
        server = chat.listen_for_events(path, lambda event: None, lambda: loop_on)
        self.addCleanup(server.close)
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"VOICE_CHAT_SOCKET": str(path)}), \
                mock.patch("sys.stdin", io.StringIO('{"hook_event_name": "UserPromptSubmit"}')), \
                mock.patch("sys.stdout", out):
            self.assertEqual(chat.hook(), 0)
        return out.getvalue()

    def test_while_the_loop_is_on_each_message_asks_for_the_spoken_line(self) -> None:
        note = json.loads(self.prompt_hook_output(loop_on=True))["hookSpecificOutput"]
        self.assertEqual(note["hookEventName"], "UserPromptSubmit")
        self.assertEqual(note["additionalContext"], chat.INSTRUCTION)

    def test_while_the_loop_is_off_claude_code_gets_nothing_extra(self) -> None:
        self.assertEqual(self.prompt_hook_output(loop_on=False), "")

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
        self.assertEqual(chat.event({"hook_event_name": "UserPromptSubmit"}), ("prompt", ""))
        self.assertEqual(chat.event({"hook_event_name": "UserPromptSubmit", "prompt": "hi"}), ("prompt", "hi"))
        self.assertEqual(chat.event({"hook_event_name": "Stop", "last_assistant_message": 3}), ("reply", ""))
        self.assertEqual(chat.event({"hook_event_name": "Notification", "message": "x"}), ("notice", "x", False))
        self.assertEqual(chat.event({"hook_event_name": "Notification", "message": "x",
                                     "notification_type": "permission_prompt"}), ("notice", "x", True))
        self.assertEqual(chat.event({"hook_event_name": "PreToolUse", "tool_name": "AskUserQuestion"}), ("dialog",))
        self.assertEqual(chat.event({"hook_event_name": "PreToolUse", "tool_name": "ExitPlanMode"}), ("dialog",))
        self.assertIsNone(chat.event({"hook_event_name": "PreToolUse", "tool_name": "Bash"}))
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

    def test_coming_on_turns_every_other_running_loop_off(self) -> None:
        own: list[tuple] = []
        other: list[tuple] = []
        mine = chat.listen_for_events(chat.state.STATE / "chat-1.sock", own.append)
        self.addCleanup(mine.close)
        theirs = chat.listen_for_events(chat.state.STATE / "chat-2.sock", other.append)
        self.addCleanup(theirs.close)
        (chat.state.STATE / "chat-3.sock").touch()  # left behind by a session that crashed
        chat.release_others(mine.path)
        deadline = time.monotonic() + 5
        while not other and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual(other, [("off",)])
        self.assertEqual(own, [])

    def test_toggle_with_no_session_says_so(self) -> None:
        with mock.patch.object(chat.listen, "cue") as cue, mock.patch.object(chat.listen, "notify") as notify:
            self.assertEqual(chat.toggle_running(), 1)
        cue.assert_called_once_with("error")
        notify.assert_called_once()


class TestDevices(TempState):
    def test_the_microphone_opens_only_after_the_on_sound_ends(self) -> None:
        order: list[str] = []
        heard = threading.Event()
        fx = chat.Devices(lambda event: heard.set(), force=False)
        fx.sound = mock.Mock(wait=lambda timeout: order.append("sound ended"))
        with mock.patch.object(chat, "quiet"), mock.patch.object(chat.listen, "cue"), \
                mock.patch.object(chat.listen, "record",
                                  side_effect=lambda *a, **k: order.append("record") or np.zeros(10, "float32")):
            fx.listen(1)
            self.assertTrue(heard.wait(5))
        self.assertEqual(order, ["sound ended", "record"])


    def test_a_speaker_that_died_is_started_again_and_released_at_the_end(self) -> None:
        spoken = threading.Event()
        fx = chat.Devices(lambda event: spoken.set(), force=False)
        with mock.patch.object(chat.serve, "start", return_value=True) as start, \
                mock.patch.object(chat.speak, "run") as run:
            fx.say("Hello.", 1)
            self.assertTrue(spoken.wait(5))
        start.assert_called_once()
        run.assert_called_once_with("Hello.")
        self.assertTrue(fx.started_speaker)

    def test_a_summary_waits_for_a_stop_still_on_its_way(self) -> None:
        order: list[str] = []
        spoken = threading.Event()
        fx = chat.Devices(lambda event: spoken.set(), force=False)
        release = threading.Event()

        def run(text, stop=False, **kwargs):
            if stop:
                release.wait(5)  # a speaker slow to answer the stop
            order.append("stop" if stop else "say")
        with mock.patch.object(chat.serve, "start", return_value=False), \
                mock.patch.object(chat.speak, "run", side_effect=run):
            fx.stop_speech()
            fx.say("Hello.", 1)
            release.set()
            self.assertTrue(spoken.wait(5))
        self.assertEqual(order, ["stop", "say"])

    def test_a_closing_session_never_starts_the_speaker(self) -> None:
        spoken = threading.Event()
        fx = chat.Devices(lambda event: spoken.set(), force=False)
        fx.closing = True
        with mock.patch.object(chat.serve, "start") as start, mock.patch.object(chat.speak, "run"):
            fx.say("Hello.", 1)
            self.assertTrue(spoken.wait(5))
        start.assert_not_called()

    def listen_with(self, transcripts: list[str]) -> tuple[list[tuple], list[dict]]:
        """Run one listen where each recording holds sound and transcribes to the next transcript."""
        posted: list[tuple] = []
        done = threading.Event()
        calls: list[dict] = []
        fx = chat.Devices(lambda event: (posted.append(event), done.set()), force=False)
        fx.model = mock.Mock(result=mock.Mock(return_value="model"))
        texts = iter(transcripts)

        def record(*args, **kwargs):
            calls.append(kwargs)
            return np.ones(10, "float32")
        with mock.patch.object(chat, "quiet"), mock.patch.object(chat.listen, "cue"), \
                mock.patch.object(chat.listen, "record", side_effect=record), \
                mock.patch.object(chat.listen, "transcribe_with", side_effect=lambda model, audio: next(texts)):
            fx.listen(1)
            self.assertTrue(done.wait(5))
        return posted, calls

    def test_a_pause_after_a_sound_with_no_words_keeps_listening(self) -> None:
        posted, calls = self.listen_with(["", "  ", "run the tests"])  # a click, a cough, then words
        self.assertEqual(posted, [("heard", 1, "run the tests")])
        self.assertEqual(len(calls), 3)
        self.assertIsNotNone(calls[0]["on_open"])
        self.assertIsNone(calls[1]["on_open"])  # no second start sound: the microphone stayed open
        self.assertLess(calls[1]["patience"], calls[0]["patience"] + 0.001)

    def test_no_words_before_the_patience_runs_out_ends_the_turn(self) -> None:
        with mock.patch.object(chat, "PATIENCE", 0.0):
            posted, calls = self.listen_with([""])
        self.assertEqual(posted, [("heard", 1, "")])
        self.assertEqual(len(calls), 1)


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

    def test_a_closed_window_or_a_kill_leaves_through_the_cleanup(self) -> None:
        """Without a handler either signal ends the process on the spot, and the speaker this
        session started stays loaded with nothing left to release it."""
        import signal

        self.assertEqual(set(chat.ENDING_SIGNALS), {signal.SIGHUP, signal.SIGTERM})
        with self.assertRaises(SystemExit) as raised:
            chat.ended(signal.SIGHUP, None)
        self.assertEqual(raised.exception.code, 128 + signal.SIGHUP)


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

    def test_what_the_microphone_hears_while_the_start_sound_plays_is_dropped(self) -> None:
        sound = iter([np.ones((1600, 1), "float32")] * 3)  # loud enough to count as speech
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.side_effect = lambda n: (next(sound, np.zeros((n, 1), "float32")), False)
        sd = mock.Mock(InputStream=mock.Mock(return_value=stream))
        playing = mock.Mock(poll=mock.Mock(side_effect=[None, None, None, 0]))
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            audio = listen.record(60, 1.5, on_open=lambda: playing, patience=1.0)
        self.assertEqual(listen.peak(audio), 0.0)
        self.assertLessEqual(len(audio), int(listen.RATE * 1.05))  # the patience ended it, not a pause

    def test_a_changed_input_device_restarts_portaudio_and_records(self) -> None:
        class PortAudioError(Exception):
            pass

        stream = mock.MagicMock()
        stream.__enter__.return_value.read.side_effect = lambda n: (np.zeros((n, 1), "float32"), False)
        opens = iter([PortAudioError("Error starting stream"), stream])  # stale device list, then fresh

        def input_stream(**options):
            opened = next(opens)
            if isinstance(opened, Exception):
                raise opened
            return opened
        sd = mock.Mock(InputStream=input_stream, PortAudioError=PortAudioError)
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            audio = listen.record(60, 1.5, patience=0.3)
        sd._initialize.assert_called_once()
        self.assertGreater(len(audio), 0)

    def test_record_stops_when_told(self) -> None:
        stream = mock.MagicMock()
        stream.__enter__.return_value.read.side_effect = lambda n: (np.ones((n, 1), "float32"), False)
        sd = mock.Mock(InputStream=mock.Mock(return_value=stream))
        stop = threading.Event()
        threading.Timer(0.05, stop.set).start()
        with mock.patch.dict(sys.modules, {"sounddevice": sd}):
            listen.record(60, None, stop=stop.is_set)
