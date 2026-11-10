"""The resident speaker: one loaded model, a newest-request-wins queue, and the client's fallback.
The model and the audio device are faked."""

from __future__ import annotations

import json
import os
import threading
import time
from types import SimpleNamespace
from unittest import mock

from support import TempState
from voice import models, serve, speak, state


class FakeAudio:
    """Stands in for sounddevice: play starts a chunk, wait blocks until stop or finish."""

    class PortAudioError(Exception):
        pass

    def __init__(self) -> None:
        self.played: list[str] = []
        self.stops = 0
        self.idle = threading.Event()
        self.idle.set()
        self.lost_device = False  # every play fails until PortAudio starts again
        self.restarts = 0

    def play(self, data, rate) -> None:
        if self.lost_device:
            raise self.PortAudioError("Error starting stream")
        self.played.append(data)
        self.idle.clear()

    def _terminate(self) -> None:
        pass

    def _initialize(self) -> None:
        self.restarts += 1
        self.lost_device = False

    def wait(self) -> None:
        self.idle.wait(5)

    def stop(self) -> None:
        self.stops += 1
        self.idle.set()

    def finish(self) -> None:
        self.idle.set()


def fake_model(*chunks_per_call: list[str]):
    model = mock.Mock()
    model.generate.side_effect = [iter(SimpleNamespace(audio=c, sample_rate=24000) for c in chunks)
                                  for chunks in chunks_per_call]
    return model


class Reply:
    """Stands in for the client's connection: records what the server sends back."""

    def __init__(self) -> None:
        self.sent = b""
        self.closed = threading.Event()

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def close(self) -> None:
        self.closed.set()


class TestSpeaker(TempState):
    def test_a_new_request_stops_the_one_playing_and_answers_its_client(self) -> None:
        audio = FakeAudio()
        speaker = serve.Speaker(fake_model(["a1", "a2"], ["b1"]), audio)
        first, second = Reply(), Reply()
        speaker.say(first, "first", "voice.safetensors", 1.0, "a")
        deadline = time.monotonic() + 5
        while not audio.played and time.monotonic() < deadline:
            time.sleep(0.01)
        speaker.say(second, "second", "voice.safetensors", 1.0, "a")
        self.assertTrue(first.closed.wait(5))
        self.assertEqual(first.sent, b"done\n")
        self.assertNotIn("a2", audio.played)  # the first reply's next sentence never plays
        audio.finish()
        self.assertTrue(second.closed.wait(5))
        self.assertEqual(audio.played, ["a1", "b1"])
        self.assertGreaterEqual(audio.stops, 1)

    def test_a_lost_output_device_restarts_portaudio_and_plays(self) -> None:
        audio = FakeAudio()
        audio.lost_device = True  # the output changed after the server started
        speaker = serve.Speaker(fake_model(["a1"], ["a1"]), audio)
        reply = Reply()
        speaker.say(reply, "hello", "voice.safetensors", 1.0, "a")
        deadline = time.monotonic() + 5
        while not audio.played and time.monotonic() < deadline:
            time.sleep(0.01)
        audio.finish()
        self.assertTrue(reply.closed.wait(5))
        self.assertEqual(audio.restarts, 1)
        self.assertEqual(audio.played, ["a1"])

    def test_stop_with_nothing_playing_does_nothing(self) -> None:
        audio = FakeAudio()
        serve.Speaker(fake_model(), audio).stop()
        self.assertEqual(audio.stops, 0)

    def test_warming_makes_a_sentence_without_playing_it(self) -> None:
        audio = FakeAudio()
        model = fake_model(["w"])
        with mock.patch.object(speak, "voice_file", return_value="v.safetensors"):
            serve.Speaker(model, audio).warm()
        model.generate.assert_called_once()
        self.assertEqual(audio.played, [])


class TestRequests(TempState):
    def setUp(self) -> None:
        super().setUp()
        self.speaker = mock.Mock()
        self.conn = Reply()

    def test_ping_answers_ok(self) -> None:
        self.assertTrue(serve.handle(self.speaker, self.conn, {"op": "ping"}))
        self.assertEqual(self.conn.sent, b"ok\n")

    def test_playing_says_whether_anything_is_playing(self) -> None:
        self.speaker.playing.return_value = True
        serve.handle(self.speaker, self.conn, {"op": "playing"})
        self.assertEqual(self.conn.sent, b"yes\n")

    def test_say_resolves_the_voice_and_hands_the_connection_to_the_speaker(self) -> None:
        with mock.patch.object(speak, "voice_file", return_value="/m/voices/bf_emma.safetensors"):
            serve.handle(self.speaker, self.conn, {"op": "say", "text": "hi", "voice": "bf_emma", "speed": 1.2})
        self.speaker.say.assert_called_once_with(self.conn, "hi", "/m/voices/bf_emma.safetensors", 1.2, "b")
        self.assertFalse(self.conn.closed.is_set())  # the speaker answers once playback ends

    def test_an_unknown_voice_is_an_error_and_plays_nothing(self) -> None:
        serve.handle(self.speaker, self.conn, {"op": "say", "text": "hi", "voice": "../x", "speed": 1.0})
        self.speaker.say.assert_not_called()
        self.assertTrue(self.conn.sent.startswith(b"error"))

    def test_a_bad_request_is_an_error(self) -> None:
        for message in ({"op": "say", "text": 3}, {"op": "say", "text": "hi", "speed": 9}, {"op": "nope"}, []):
            conn = Reply()
            self.assertTrue(serve.handle(self.speaker, conn, message))
            self.assertTrue(conn.sent.startswith(b"error"), message)
        self.speaker.say.assert_not_called()

    def test_stop_and_quit(self) -> None:
        self.assertTrue(serve.handle(self.speaker, self.conn, {"op": "stop"}))
        self.assertFalse(serve.handle(self.speaker, Reply(), {"op": "quit"}))
        self.assertEqual(self.speaker.stop.call_count, 2)


class TestClient(TempState):
    def serve_in_thread(self, speaker) -> None:
        """Run a real server loop on the temporary socket, holding the pid lock as `voice serve` does."""
        state.folder()
        sock = serve.bind()
        self.held(serve.PIDFILE, os.getpid())
        thread = threading.Thread(target=serve.serve_forever, args=(sock, speaker), daemon=True)
        thread.start()
        self.addCleanup(thread.join, 5)
        self.addCleanup(serve.request, {"op": "quit"})

    def test_speak_goes_through_a_running_server_and_waits_for_it(self) -> None:
        speaker = mock.Mock()
        speaker.say.side_effect = lambda conn, *a: (conn.sendall(b"done\n"), conn.close())
        self.serve_in_thread(speaker)
        with mock.patch.object(speak, "voice_file", return_value="v.safetensors"), \
                mock.patch.object(speak, "play") as one_shot:
            self.assertEqual(speak.run("hello", voice="af_heart", speed=1.0), 0)
        one_shot.assert_not_called()
        self.assertEqual(speaker.say.call_args.args[1:], ("hello", "v.safetensors", 1.0, "a"))

    def test_stop_reaches_the_server(self) -> None:
        speaker = mock.Mock()
        self.serve_in_thread(speaker)
        speak.run(None, stop=True)
        speaker.stop.assert_called()

    def test_without_a_server_speak_plays_on_its_own(self) -> None:
        with mock.patch.object(speak, "play") as one_shot:
            self.assertEqual(speak.run("hello"), 0)
        one_shot.assert_called_once()

    def test_a_server_that_dies_while_starting_falls_back_to_playing_on_its_own(self) -> None:
        fd = self.held(serve.PIDFILE, None)  # starting: locked, no pid yet
        threading.Timer(0.3, os.close, args=(fd,)).start()
        with mock.patch.object(speak, "play") as one_shot:
            self.assertEqual(speak.run("hello"), 0)
        one_shot.assert_called_once()

    def test_warm_starts_one_server_in_the_background(self) -> None:
        with mock.patch.object(serve.subprocess, "Popen") as popen:
            speak.run(None, warm=True)
        argv = popen.call_args.args[0]
        self.assertEqual(argv[1:5], ["-I", "-m", "voice.cli", "serve"])
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        self.held(serve.PIDFILE, 4242)
        with mock.patch.object(serve.subprocess, "Popen") as popen:
            speak.run(None, warm=True)
        popen.assert_not_called()

    def test_release_stops_the_server(self) -> None:
        speaker = mock.Mock()
        self.serve_in_thread(speaker)
        speak.run(None, release=True)
        deadline = time.monotonic() + 5
        while serve.SOCKET.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(serve.SOCKET.exists())

    def test_a_request_line_is_one_json_object(self) -> None:
        state.folder()
        sock = serve.bind()
        self.addCleanup(sock.close)
        self.held(serve.PIDFILE, os.getpid())
        got: list[dict] = []

        def answer() -> None:
            conn, _ = sock.accept()
            with conn:
                got.append(json.loads(serve.read_line(conn)))
                conn.sendall(b"ok\n")

        threading.Thread(target=answer, daemon=True).start()
        self.assertEqual(serve.request({"op": "ping"}), "ok")
        self.assertEqual(got, [{"op": "ping"}])
        self.assertEqual(oct(os.stat(serve.SOCKET).st_mode & 0o777), "0o600")


class TestServeRun(TempState):
    def test_critical_pressure_refuses_before_loading(self) -> None:
        with mock.patch.object(models, "pressure_level", return_value=models.PRESSURE_CRITICAL), \
                mock.patch.object(models, "load_tts") as load, self.assertRaises(models.MemoryPressure):
            serve.run()
        load.assert_not_called()
        self.assertIsNone(state.holder(serve.PIDFILE))

    def test_a_second_server_exits_at_once(self) -> None:
        self.held(serve.PIDFILE, 4242)
        with mock.patch.object(models, "load_tts") as load:
            self.assertEqual(serve.run(), 0)
        load.assert_not_called()
