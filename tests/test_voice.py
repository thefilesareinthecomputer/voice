"""voice: memory gate, model release, end-of-speech detection, and the CLI surface.
Models, microphone and speaker are faked, so no download or audio device is needed."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from voice import cli, listen, models, speak

VOICE = [sys.executable, "-m", "voice.cli"]


class TestPressure(unittest.TestCase):
    def test_critical_refuses_a_load(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}):
            with self.assertRaises(models.MemoryPressure):
                models.check_pressure()
            models.check_pressure(force=True)  # the override loads anyway

    def test_warn_and_normal_load(self) -> None:
        for level in ("1", "2"):
            with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": level}):
                models.check_pressure()


class TestLoaded(unittest.TestCase):
    def test_model_is_released_after_the_block_even_on_error(self) -> None:
        loader = mock.Mock(return_value="model")
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(models, "release") as release:
            with self.assertRaises(ValueError):
                with models.loaded(loader, "some/repo") as model:
                    self.assertEqual(model, "model")
                    raise ValueError
        loader.assert_called_once_with("some/repo")
        release.assert_called_once()

    def test_loads_offline_and_refuses_under_pressure(self) -> None:
        loader = mock.Mock()
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            with self.assertRaises(models.MemoryPressure):
                with models.loaded(loader, "some/repo"):
                    pass
        loader.assert_not_called()
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}):
            with models.loaded(loader, "some/repo"):
                self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")


class TestEndOfSpeech(unittest.TestCase):
    loud, quiet = 0.1, 0.001

    def test_waits_for_speech_before_counting_silence(self) -> None:
        self.assertFalse(listen.end_of_speech([self.quiet] * 40, silence=1.5))

    def test_ends_after_the_pause_that_follows_speech(self) -> None:
        levels = [self.quiet] * 3 + [self.loud] * 10 + [self.quiet] * 15
        self.assertTrue(listen.end_of_speech(levels, silence=1.5))

    def test_a_short_pause_does_not_end_it(self) -> None:
        levels = [self.loud] * 10 + [self.quiet] * 5
        self.assertFalse(listen.end_of_speech(levels, silence=1.5))


class TestTranscribe(unittest.TestCase):
    def test_transcribes_through_the_configured_model_and_releases_it(self) -> None:
        model = mock.Mock()
        model.generate.return_value = SimpleNamespace(text="  hello there ")
        loader = mock.Mock(return_value=model)
        env = {"VOICE_PRESSURE_LEVEL": "1", "VOICE_STT_MODEL": "fake/stt"}
        with mock.patch.dict(os.environ, env), \
                mock.patch("mlx_audio.stt.utils.load_model", loader), \
                mock.patch.object(models, "release") as release:
            text = listen.transcribe(np.zeros(1600, dtype="float32"))
        self.assertEqual(text, "hello there")
        loader.assert_called_once_with("fake/stt")
        release.assert_called_once()

    def test_empty_audio_loads_nothing(self) -> None:
        with mock.patch("mlx_audio.stt.utils.load_model") as loader:
            self.assertEqual(listen.transcribe(np.zeros(0, dtype="float32")), "")
        loader.assert_not_called()

    def test_run_loads_while_recording_then_prints_copies_and_releases(self) -> None:
        model = mock.Mock()
        model.generate.return_value = SimpleNamespace(text="dictated text")
        order: list[str] = []
        load = mock.Mock(side_effect=lambda repo: order.append("load") or model)
        record = mock.Mock(side_effect=lambda *a: order.append("record") or np.ones(10, dtype="float32"))
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(listen, "_load", load), \
                mock.patch.object(listen, "record", record), \
                mock.patch.object(listen, "copy") as copy, \
                mock.patch.object(models, "release") as release, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(listen.run(), 0)
        self.assertEqual(out.getvalue().strip(), "dictated text")
        copy.assert_called_once_with("dictated text")
        self.assertEqual(sorted(order), ["load", "record"])  # both ran, the load on its own thread
        release.assert_called_once()


class TestLoading(unittest.TestCase):
    def test_background_load_is_released_even_on_error(self) -> None:
        loader = mock.Mock(return_value="model")
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(models, "release") as release:
            with self.assertRaises(ValueError):
                with models.loading(loader, "some/repo") as pending:
                    self.assertEqual(pending.result(), "model")
                    raise ValueError
        release.assert_called_once()

    def test_refuses_before_loading_under_pressure(self) -> None:
        loader = mock.Mock()
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}):
            with self.assertRaises(models.MemoryPressure):
                with models.loading(loader, "some/repo"):
                    pass
        loader.assert_not_called()


class TestSpeak(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.multiple(speak, STATE=Path(tmp.name), PIDFILE=Path(tmp.name) / "speak.pid")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_speaks_and_clears_its_pidfile(self) -> None:
        with mock.patch.object(speak, "play") as play, mock.patch("signal.signal"):
            self.assertEqual(speak.run("hello"), 0)
        play.assert_called_once_with("hello", False)
        self.assertFalse(speak.PIDFILE.exists())

    def test_empty_text_plays_nothing(self) -> None:
        with mock.patch.object(speak, "play") as play:
            self.assertEqual(speak.run("   "), 0)
        play.assert_not_called()


class TestCli(unittest.TestCase):
    def test_speak_reads_stdin(self) -> None:
        with mock.patch("voice.speak.run", return_value=0) as run, \
                mock.patch("sys.stdin", io.StringIO("from stdin\n")):
            self.assertEqual(cli.main(["speak"]), 0)
        run.assert_called_once_with("from stdin\n", stop=False, force=False)

    def test_memory_pressure_exits_3(self) -> None:
        with mock.patch("voice.listen.run", side_effect=models.MemoryPressure("critical")):
            self.assertEqual(cli.main(["listen"]), 3)

    def test_stop_runs_end_to_end_without_a_model(self) -> None:
        with tempfile.TemporaryDirectory() as state:
            result = subprocess.run(VOICE + ["speak", "--stop"], capture_output=True, text=True,
                                    env={**os.environ, "XDG_STATE_HOME": state})
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_listen_under_critical_pressure_exits_before_recording(self) -> None:
        result = subprocess.run(VOICE + ["listen"], capture_output=True, text=True,
                                env={**os.environ, "VOICE_PRESSURE_LEVEL": "4"}, timeout=60)
        self.assertEqual(result.returncode, 3)
        self.assertIn("memory pressure is critical", result.stderr)
        self.assertNotIn("listening", result.stderr)


if __name__ == "__main__":
    unittest.main()
