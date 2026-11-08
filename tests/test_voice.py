"""voice: memory gate, model release, end-of-speech detection, and the CLI surface.
Models, microphone and speaker are faked, so no download or audio device is needed."""

from __future__ import annotations

import contextlib
import io
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from voice import cli, listen, models, setup, shrink, speak, state

VOICE = [sys.executable, "-m", "voice.cli"]


class TempState(unittest.TestCase):
    """Points the state folder and the pid and log files at a temporary folder."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.state = Path(tmp.name) / "claude-voice"
        for patcher in (mock.patch.object(state, "STATE", self.state),
                        mock.patch.multiple(listen, PIDFILE=self.state / "listen.pid", LOG=self.state / "listen.log",
                                            TRANSCRIBING=self.state / "transcribe.lock"),
                        mock.patch.multiple(speak, PIDFILE=self.state / "speak.pid",
                                            TICKET=self.state / "speak.latest")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def held(self, path: Path, pid: int | None) -> int:
        """Hold the lock on `path` as a running process would, optionally with its pid written."""
        fd = state.hold(path)
        self.assertIsNotNone(fd)
        if pid is not None:
            state.write_pid(fd, pid)
        self.addCleanup(_close, fd)
        return fd


def _close(fd: int) -> None:
    with contextlib.suppress(OSError):  # a test may have closed it already
        os.close(fd)


class TestPidLock(TempState):
    def test_a_held_lock_names_its_pid(self) -> None:
        self.held(self.state / "x.pid", 4242)
        self.assertEqual(state.holder(self.state / "x.pid"), 4242)

    def test_a_pid_without_a_live_holder_is_never_returned(self) -> None:
        fd = self.held(self.state / "x.pid", 4242)
        os.close(fd)  # the process exited; the stale pid stays in the file
        self.assertEqual((self.state / "x.pid").read_text(), "4242")
        self.assertIsNone(state.holder(self.state / "x.pid"))
        self.assertIsNone(state.holder(self.state / "missing.pid"))

    def test_a_second_holder_is_refused(self) -> None:
        self.held(self.state / "x.pid", None)
        self.assertIsNone(state.hold(self.state / "x.pid"))
        self.assertIsNone(state.holder(self.state / "x.pid"))  # held, but no pid written yet

    def test_a_child_keeps_a_handed_over_lock_until_it_exits(self) -> None:
        path = self.state / "x.pid"
        lock = state.hold(path)
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], pass_fds=(lock,))
        self.addCleanup(child.wait)
        self.addCleanup(child.kill)
        state.write_pid(lock, child.pid)
        os.close(lock)  # the parent is gone; the child alone holds it now
        self.assertEqual(state.holder(path), child.pid)
        child.kill()
        child.wait()
        self.assertIsNone(state.holder(path))

    def test_the_folder_and_files_are_private_and_a_symlink_is_refused(self) -> None:
        self.held(self.state / "x.pid", 1)
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.state / "x.pid").stat().st_mode & 0o777, 0o600)
        (self.state / "link.pid").symlink_to(self.state / "elsewhere")
        with self.assertRaises(OSError):
            state.hold(self.state / "link.pid")
        self.assertFalse((self.state / "elsewhere").exists())

    def test_only_an_absolute_xdg_folder_counts(self) -> None:
        default = Path.home() / ".local" / "state"
        self.assertEqual(state.xdg({"XDG_STATE_HOME": "/srv/st"}, "XDG_STATE_HOME", ".local/state"), Path("/srv/st"))
        for value in ("", "relative/st"):
            self.assertEqual(state.xdg({"XDG_STATE_HOME": value}, "XDG_STATE_HOME", ".local/state"), default)
        self.assertEqual(state.xdg({}, "XDG_DATA_HOME", ".local/share"), Path.home() / ".local" / "share")

    def test_pid_one_or_less_is_never_returned(self) -> None:
        for pid in (1, 0, -1):
            path = self.state / f"p{pid + 5}.pid"
            self.held(path, pid)
            self.assertIsNone(state.holder(path))

    def test_a_symlinked_state_folder_is_refused(self) -> None:
        real = self.state.parent / "elsewhere"
        real.mkdir()
        self.state.symlink_to(real)
        with self.assertRaises(OSError):
            state.folder()

    def test_a_folder_left_open_by_an_older_version_is_closed(self) -> None:
        self.state.mkdir(mode=0o755)
        state.folder()
        self.assertEqual(self.state.stat().st_mode & 0o777, 0o700)


class TestModelFamily(unittest.TestCase):
    def test_every_load_pins_its_family_and_gets_a_resolved_folder(self) -> None:
        with mock.patch("mlx_audio.utils.base_load_model") as base, \
                mock.patch.object(models, "folder", side_effect=lambda r: Path("/resolved") / Path(r).name):
            models.load_stt("someone/qwen3-asr-1.7b")
            models.load_tts("/tmp/chatterbox_turbo")
        stt, tts = base.call_args_list
        # a folder, never a name mlx-audio would first try as a path under the current folder
        self.assertEqual(stt.args, ("/resolved/qwen3-asr-1.7b",))
        self.assertEqual(stt.kwargs, {"category": "stt", "model_remapping": {}, "model_type": "parakeet",
                                      "model_name_parts": [], "strict": False})
        self.assertEqual(tts.kwargs["model_type"], "kokoro")
        self.assertEqual((tts.kwargs["model_remapping"], tts.kwargs["model_name_parts"]), ({}, []))

    def test_a_missing_model_says_to_run_setup_before_anything_loads(self) -> None:
        with mock.patch("mlx_audio.utils.base_load_model") as base, \
                mock.patch.object(models, "folder", return_value=None), \
                self.assertRaises(models.ModelMissing) as caught:
            models.load_stt("mlx-community/parakeet-tdt-0.6b-v2")
        self.assertIn("voice setup", str(caught.exception))
        base.assert_not_called()

    def test_a_relative_name_is_a_repo_id_never_a_folder_in_the_current_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "mlx-community" / "x").mkdir(parents=True)
            cwd = os.getcwd()
            os.chdir(tmp)
            self.addCleanup(os.chdir, cwd)
            with mock.patch("huggingface_hub.snapshot_download", side_effect=OSError("not cached")) as hub:
                self.assertIsNone(models.folder("mlx-community/x"))
            hub.assert_called_once()
            self.assertEqual(models.folder(str(Path(tmp) / "mlx-community" / "x")), Path(tmp) / "mlx-community" / "x")

    def test_the_default_models_resolve_at_their_pinned_revisions(self) -> None:
        for repo in (models.TTS_MODEL, models.STT_MODEL):
            self.assertRegex(models.REVISIONS[repo], r"^[0-9a-f]{40}$")
        with mock.patch("huggingface_hub.snapshot_download", return_value="/cache/snap") as hub:
            models.folder(models.STT_MODEL)
            models.folder("someone/other")
        self.assertEqual(hub.call_args_list[0].kwargs,
                         {"revision": models.REVISIONS[models.STT_MODEL], "local_files_only": True})
        self.assertEqual(hub.call_args_list[1].kwargs, {"revision": None, "local_files_only": True})

    def test_the_16_bit_copy_is_used_without_a_setting_and_a_setting_still_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(models, "MODELS", Path(tmp)), \
                mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(models.stt_model(), models.STT_MODEL)
            (Path(tmp) / "parakeet-tdt-0.6b-v2-bf16").mkdir()
            self.assertEqual(models.stt_model(), str(Path(tmp) / "parakeet-tdt-0.6b-v2-bf16"))
            os.environ["VOICE_STT_MODEL"] = "someone/else"
            self.assertEqual(models.stt_model(), "someone/else")

    def test_voice_tts_model_names_the_speech_model_and_voice_model_still_works(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(models.tts_model(), models.TTS_MODEL)
            os.environ["VOICE_MODEL"] = "old/name"
            self.assertEqual(models.tts_model(), "old/name")
            os.environ["VOICE_TTS_MODEL"] = "new/name"
            self.assertEqual(models.tts_model(), "new/name")

    def test_a_model_that_names_another_family_still_loads_as_the_pinned_one(self) -> None:
        # Runs mlx-audio's real loader, so a library upgrade that lets config.json win again fails here.
        import mlx_audio.utils as mu

        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "qwen3_asr-evil"
            folder.mkdir()
            (folder / "config.json").write_text('{"model_type": "qwen3_asr", "architecture": "qwen3_asr"}')
            with mock.patch.object(mu.importlib, "import_module", wraps=mu.importlib.import_module) as imports, \
                    contextlib.suppress(Exception):  # the empty folder fails after the family is chosen
                models.load_stt(str(folder))
        families = [c.args[0] for c in imports.call_args_list if c.args[0].startswith("mlx_audio.stt.models.")]
        self.assertEqual(families, ["mlx_audio.stt.models.parakeet"])


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

    def test_reads_the_kernel_by_absolute_path_and_ignores_a_bad_override(self) -> None:
        out = SimpleNamespace(stdout="2\n")
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "high"}), \
                mock.patch.object(models.subprocess, "run", return_value=out) as run:
            self.assertEqual(models.pressure_level(), 2)
        self.assertEqual(run.call_args.args[0][0], "/usr/sbin/sysctl")


class TestLoaded(unittest.TestCase):
    def test_model_is_released_after_the_block_even_on_error(self) -> None:
        loader = mock.Mock(return_value="model")
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(models, "release") as release, self.assertRaises(ValueError), \
                models.loaded(loader, "some/repo") as model:
            self.assertEqual(model, "model")
            raise ValueError
        loader.assert_called_once_with("some/repo")
        release.assert_called_once()

    def test_loads_offline_and_refuses_under_pressure(self) -> None:
        loader = mock.Mock()
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            with self.assertRaises(models.MemoryPressure), models.loaded(loader, "some/repo"):
                pass
        loader.assert_not_called()
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), models.loaded(loader, "some/repo"):
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
                mock.patch.object(models, "load_stt", loader), \
                mock.patch.object(models, "release") as release:
            text = listen.transcribe(np.zeros(1600, dtype="float32"))
        self.assertEqual(text, "hello there")
        loader.assert_called_once_with("fake/stt")
        release.assert_called_once()

    def test_empty_audio_loads_nothing(self) -> None:
        with mock.patch.object(models, "load_stt") as loader:
            self.assertEqual(listen.transcribe(np.zeros(0, dtype="float32")), "")
        loader.assert_not_called()

    def test_run_loads_while_recording_then_prints_copies_and_releases(self) -> None:
        model = mock.Mock()
        model.generate.return_value = SimpleNamespace(text="dictated text")
        order: list[str] = []
        load = mock.Mock(side_effect=lambda repo: order.append("load") or model)
        record = mock.Mock(side_effect=lambda *a: order.append("record") or np.ones(10, dtype="float32"))
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(models, "require"), \
                mock.patch.object(models, "load_stt", load), \
                mock.patch.object(listen, "record", record), \
                mock.patch.object(listen, "copy") as copy, \
                mock.patch.object(models, "release") as release, \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(listen.run(), 0)
        self.assertEqual(out.getvalue().strip(), "dictated text")
        copy.assert_called_once_with("dictated text")
        self.assertEqual(sorted(order), ["load", "record"])  # both ran, the load on its own thread
        release.assert_called_once()

    def test_run_checks_the_model_is_there_before_recording(self) -> None:
        with mock.patch.object(models, "require", side_effect=models.ModelMissing("run `voice setup` first")), \
                mock.patch.object(listen, "record") as record, self.assertRaises(models.ModelMissing):
            listen.run()
        record.assert_not_called()


class FakeStream:
    """Stands in for sounddevice.InputStream: hands back quiet blocks."""

    def __init__(self, **kwargs) -> None:
        self.reads = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def read(self, n: int):
        self.reads += 1
        return np.zeros((n, 1), dtype="float32"), False


class TestToggle(TempState):
    def setUp(self) -> None:
        super().setUp()
        patcher = mock.patch.object(models, "require")  # the model is never on a test machine
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_first_press_starts_an_isolated_detached_recorder_that_holds_the_lock(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(listen.subprocess, "Popen", return_value=SimpleNamespace(pid=4242)) as popen, \
                mock.patch.object(listen.os, "kill") as kill:
            self.assertEqual(listen.toggle(1800.0, True, False), 0)
        argv, opts = popen.call_args.args[0], popen.call_args.kwargs
        # -I: the child never imports from the folder the shortcut ran in, or from PYTHONPATH
        self.assertEqual(argv[:6], [sys.executable, "-I", "-m", "voice.cli", "listen", "--until-stopped"])
        self.assertEqual(argv[6], str(opts["pass_fds"][0]))  # the child is told which descriptor is the lock
        self.assertEqual(opts["cwd"], self.state)
        self.assertTrue(opts["start_new_session"])
        # The child writes its own pid once its handler is in; a SIGUSR1 before that would kill it.
        self.assertEqual(listen.PIDFILE.read_text(), "")
        kill.assert_not_called()

    def test_second_press_signals_the_running_recorder(self) -> None:
        self.held(listen.PIDFILE, 4242)
        with mock.patch.object(listen.subprocess, "Popen") as popen, \
                mock.patch.object(listen.os, "kill") as kill:
            self.assertEqual(listen.toggle(1800.0, True, False), 0)
        kill.assert_called_once_with(4242, listen.signal.SIGUSR1)
        popen.assert_not_called()

    def test_a_recorder_that_exits_as_the_press_lands_is_not_an_error(self) -> None:
        self.held(listen.PIDFILE, 4242)
        with mock.patch.object(listen.os, "kill", side_effect=ProcessLookupError):
            self.assertEqual(listen.toggle(1800.0, True, False), 0)

    def test_a_press_while_the_recorder_is_starting_does_nothing(self) -> None:
        self.held(listen.PIDFILE, None)
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(listen.subprocess, "Popen") as popen, \
                mock.patch.object(listen.os, "kill") as kill:
            self.assertEqual(listen.toggle(1800.0, True, False), 0)
        kill.assert_not_called()
        popen.assert_not_called()

    def test_a_stale_pid_is_never_signalled_and_a_new_recorder_starts(self) -> None:
        state.folder()
        listen.PIDFILE.write_text("4242")  # left by a recorder that was killed; nobody holds the lock
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(listen.subprocess, "Popen", return_value=SimpleNamespace(pid=5151)) as popen, \
                mock.patch.object(listen.os, "kill") as kill:
            listen.toggle(1800.0, True, False)
        kill.assert_not_called()
        popen.assert_called_once()

    def test_background_recording_copies_releases_the_lock_and_keeps_the_transcript_out_of_notifications(self):
        notes: list[tuple[str, str]] = []
        lock = self.held(listen.PIDFILE, None)  # as toggle hands it over: held, no pid yet
        handler = {}

        def record(*args, on_open=None):
            self.assertIn(listen.signal.SIGUSR1, handler)  # the pid goes out only once a press can't kill it
            self.assertEqual(state.holder(listen.PIDFILE), os.getpid())  # held while it records
            self.assertEqual(cue.call_args_list, [])  # no start sound before the microphone is open
            on_open()
            return np.ones(10, dtype="float32")

        with mock.patch.object(listen, "record", side_effect=record) as recorder, \
                mock.patch.object(listen, "transcribe",
                                  side_effect=lambda *a, **kw: self.assertIsNone(state.holder(listen.PIDFILE))
                                  or "walked and talked"), \
                mock.patch.object(listen, "copy") as copy, \
                mock.patch.object(listen, "cue") as cue, \
                mock.patch.object(listen, "notify", side_effect=lambda t, m: notes.append((t, m))), \
                mock.patch.object(listen.signal, "signal", side_effect=handler.__setitem__), \
                mock.patch("sys.stdout", new_callable=io.StringIO) as out:
            self.assertEqual(listen.run_until_stopped(1800.0, True, False, lock_fd=lock), 0)
        self.assertNotIn("walked and talked", out.getvalue())  # stdout is the log file; no transcript on disk
        self.assertIsNone(recorder.call_args.args[1])  # no silence cutoff in toggle mode
        copy.assert_called_once_with("walked and talked")
        self.assertEqual([c.args[0] for c in cue.call_args_list], ["start", "stop"])
        self.assertEqual(notes[0][0], "voice: listening")
        self.assertEqual(notes[-1], ("voice: copied", "3 words"))
        for _, message in notes:  # macOS keeps notification text in its own database
            self.assertNotIn("walked", message)

    def stop_take(self, audio, transcribed="some words", **kwargs):
        """One background take with the sounds, notifications and clipboard faked; returns
        (exit code, notifications, the transcribe mock)."""
        notes: list[tuple[str, str]] = []

        def record(*args, on_open=None):
            on_open()
            return audio

        with mock.patch.object(listen, "record", side_effect=record), \
                mock.patch.object(listen, "transcribe", side_effect=kwargs.pop("transcribe", None),
                                  return_value=transcribed) as transcribe, \
                mock.patch.object(listen, "copy", return_value=kwargs.pop("copied", True)), \
                mock.patch.object(listen, "cue") as self.cues, \
                mock.patch.object(listen, "notify", side_effect=lambda t, m: notes.append((t, m))), \
                mock.patch.object(listen.signal, "signal"):
            code = listen.run_until_stopped(1800.0, True, False, **kwargs)
        return code, notes, transcribe

    def test_a_silent_take_never_loads_the_model_and_an_all_zero_one_points_at_the_microphone(self):
        _, notes, transcribe = self.stop_take(np.full(16000, 0.001, dtype="float32"))
        transcribe.assert_not_called()
        self.assertEqual(notes[-1], ("voice: nothing heard", "No speech was recognized."))
        _, notes, transcribe = self.stop_take(np.zeros(16000, dtype="float32"))
        transcribe.assert_not_called()
        self.assertIn("Microphone", notes[-1][1])

    def test_critical_pressure_at_the_stop_waits_then_keeps_the_take(self):
        levels = iter([4, 4, 1])
        with mock.patch.object(models, "pressure_level", side_effect=lambda: next(levels)), \
                mock.patch.object(listen.time, "sleep") as sleep:
            code, notes, transcribe = self.stop_take(np.ones(16000, dtype="float32"))
        self.assertEqual(code, 0)
        self.assertIn("voice: waiting for memory", [t for t, _ in notes])
        sleep.assert_called()
        self.assertTrue(transcribe.call_args.kwargs.get("force") or transcribe.call_args.args[1])
        self.assertEqual(notes[-1], ("voice: copied", "2 words"))

    def test_takes_transcribe_one_at_a_time(self):
        def transcribe(audio, force=False):
            self.assertIsNone(state.hold(listen.TRANSCRIBING))  # held by this take while it transcribes
            return "one at a time"

        code, notes, _ = self.stop_take(np.ones(16000, dtype="float32"), transcribe=transcribe)
        self.assertEqual(code, 0)
        self.assertIsNotNone(state.hold(listen.TRANSCRIBING))  # released afterwards

    def test_a_failure_notification_and_the_log_hold_no_error_text(self):
        # The notification names the log, and the log (stderr of the background half) gets
        # the error's type and stack but never its message, which could quote the take.
        notes: list[tuple[str, str]] = []
        log = io.StringIO()
        with mock.patch.object(listen, "record", side_effect=lambda *a, on_open=None: np.ones(10, dtype="float32")), \
                mock.patch.object(listen, "transcribe", side_effect=RuntimeError("the dictated words")), \
                mock.patch.object(listen, "cue"), \
                mock.patch.object(listen, "notify", side_effect=lambda t, m: notes.append((t, m))), \
                mock.patch.object(listen.signal, "signal"), contextlib.redirect_stderr(log):
            code = listen.run_until_stopped(1800.0, True, False)
        self.assertEqual(code, 1)
        self.assertEqual(notes[-1][0], "voice: failed")
        self.assertIn(str(listen.LOG), notes[-1][1])
        self.assertNotIn("dictated", notes[-1][1])
        self.assertIn("RuntimeError", log.getvalue())
        self.assertNotIn("dictated", log.getvalue())

    def test_a_failed_copy_is_reported(self):
        _, notes, _ = self.stop_take(np.ones(16000, dtype="float32"), copied=False)
        self.assertEqual(notes[-1][0], "voice: copy failed")

    def test_a_missing_model_is_named_in_a_notification(self):
        missing = models.ModelMissing("x is not on this machine; run `voice setup` first")
        _, notes, _ = self.stop_take(np.ones(16000, dtype="float32"), transcribe=missing)
        self.assertEqual(notes[-1][0], "voice: model missing")
        self.assertIn("voice setup", notes[-1][1])

    def test_the_background_half_refuses_a_descriptor_that_is_not_its_lock(self):
        with tempfile.TemporaryFile() as other, self.assertRaises(SystemExit):
            listen.run_until_stopped(1800.0, True, False, lock_fd=other.fileno())

    def test_preload_starts_loading_before_the_recording(self):
        order: list[str] = []
        model = mock.Mock()
        model.generate.return_value = SimpleNamespace(text="early load")

        @contextlib.contextmanager
        def loading(*args):
            order.append("load")
            yield SimpleNamespace(result=lambda: model)

        with mock.patch.object(models, "loading", side_effect=loading):
            code, notes, transcribe = self.stop_take(np.ones(16000, dtype="float32"), preload=True)
        self.assertEqual(order, ["load"])
        transcribe.assert_not_called()  # the preloaded model transcribes
        self.assertEqual(notes[-1], ("voice: copied", "2 words"))

    def test_a_refused_start_is_heard_and_seen(self):
        notes: list[tuple[str, str]] = []
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}), \
                mock.patch.object(models, "require"), \
                mock.patch.object(listen.subprocess, "Popen") as popen, \
                mock.patch.object(listen, "cue") as cue, \
                mock.patch.object(listen, "notify", side_effect=lambda t, m: notes.append((t, m))):
            self.assertEqual(listen.toggle(1800.0, True, False), 3)
        popen.assert_not_called()
        cue.assert_called_once_with("error")
        self.assertEqual(notes[-1][0], "voice: not started")
        # Pressure is checked first, so pin it to normal here or the machine's own level decides.
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(models, "require", side_effect=models.ModelMissing("run `voice setup` first")), \
                mock.patch.object(listen.subprocess, "Popen") as popen, \
                mock.patch.object(listen, "cue"), \
                mock.patch.object(listen, "notify", side_effect=lambda t, m: notes.append((t, m))):
            self.assertEqual(listen.toggle(1800.0, True, False), 4)
        popen.assert_not_called()
        self.assertIn("voice setup", notes[-1][1])

    def test_notify_never_waits_for_osascript(self):
        with mock.patch.object(listen.subprocess, "run", side_effect=AssertionError("blocking")), \
                mock.patch.object(listen.subprocess, "Popen") as popen:
            listen.notify("voice: t", "m")
        self.assertEqual(popen.call_args.args[0][0], "/usr/bin/osascript")

    def test_a_key_press_does_not_import_numpy(self):
        probe = "import sys; from voice import listen; print('numpy' in sys.modules)"
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
        self.assertEqual(result.stdout.strip(), "False", result.stderr)

    def _stop_with_paste(self, pasted: bool) -> tuple[mock.Mock, list[tuple[str, str]]]:
        notes: list[tuple[str, str]] = []
        with mock.patch.object(listen, "record", return_value=np.ones(10, dtype="float32")), \
                mock.patch.object(listen, "transcribe", return_value="to the cursor"), \
                mock.patch.object(listen, "copy") as copy, \
                mock.patch.object(listen, "paste", return_value=pasted) as paste, \
                mock.patch.object(listen, "cue"), \
                mock.patch.object(listen, "notify", side_effect=lambda t, m: notes.append((t, m))), \
                mock.patch.object(listen.signal, "signal"):
            self.assertEqual(listen.run_until_stopped(1800.0, True, False, paste_too=True), 0)
        copy.assert_called_once_with("to the cursor")
        paste.assert_called_once()
        return copy, notes

    def test_paste_keeps_the_text_on_the_clipboard_and_says_so(self) -> None:
        _, notes = self._stop_with_paste(pasted=True)
        self.assertEqual(notes[-1], ("voice: pasted", "3 words"))

    def test_a_blocked_paste_leaves_the_text_on_the_clipboard_and_urges_no_grant(self) -> None:
        _, notes = self._stop_with_paste(pasted=False)
        self.assertEqual(notes[-1],
                         ("voice: copied, paste blocked", "The text is on the clipboard. See README: --paste."))

    def test_paste_flag_reaches_the_background_recorder(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(listen.subprocess, "Popen", return_value=SimpleNamespace(pid=4242)) as popen:
            listen.toggle(1800.0, True, False, paste_too=True)
        self.assertIn("--paste", popen.call_args.args[0])

    def test_macos_tools_run_by_absolute_path_never_through_path(self) -> None:
        with mock.patch.object(listen.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                mock.patch.object(listen.subprocess, "Popen") as popen:
            listen.copy("x")
            listen.paste()
            listen.notify("voice: t", "m")
            listen.cue("start")
        programs = [c.args[0][0] for c in run.call_args_list + popen.call_args_list]
        self.assertEqual(programs, ["/usr/bin/osascript"] * 3 + ["/usr/bin/afplay"])

    def test_a_transcript_goes_on_the_clipboard_marked_concealed_through_stdin(self) -> None:
        # Clipboard-history apps skip an item marked org.nspasteboard.ConcealedType (nspasteboard.org).
        secret = "the dictated words"
        with mock.patch.object(listen.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run:
            listen.copy(secret)
        (call,) = run.call_args_list
        argv, opts = call.args[0], call.kwargs
        self.assertEqual(argv[:3], ["/usr/bin/osascript", "-l", "JavaScript"])
        self.assertIn("org.nspasteboard.ConcealedType", " ".join(argv))
        self.assertNotIn(secret, " ".join(argv))  # never in argv, where ps shows it
        self.assertEqual(opts["input"], secret)

    def test_a_failed_concealed_copy_falls_back_to_pbcopy(self) -> None:
        with mock.patch.object(listen.subprocess, "run", return_value=SimpleNamespace(returncode=1)) as run:
            listen.copy("words")
        self.assertEqual([c.args[0][0] for c in run.call_args_list], ["/usr/bin/osascript", "/usr/bin/pbcopy"])
        self.assertEqual(run.call_args_list[-1].kwargs["input"], "words")

    def test_record_stops_when_told_and_ignores_silence_when_none(self) -> None:
        reads = {"n": 0}

        def stop() -> bool:
            reads["n"] += 1
            return reads["n"] > 30  # three seconds of quiet, well past any silence cutoff

        fake = SimpleNamespace(InputStream=FakeStream)
        with mock.patch.dict(sys.modules, {"sounddevice": fake}):
            audio = listen.record(1800.0, None, stop)
        self.assertEqual(len(audio), 30 * int(listen.RATE * listen.FRAME))


class TestChunking(unittest.TestCase):
    def test_long_audio_is_chunked_only_where_the_model_supports_it(self) -> None:
        calls: list[dict] = []

        class Chunky:
            def generate(self, audio, *, chunk_duration=None):
                calls.append({"chunk_duration": chunk_duration})
                return SimpleNamespace(text="ok")

        class Plain:
            def generate(self, audio):
                calls.append({})
                return SimpleNamespace(text="ok")

        long = np.zeros(int(listen.RATE * (listen.CHUNK_SECONDS + 1)), dtype="float32")
        listen.transcribe_with(Chunky(), long)
        listen.transcribe_with(Plain(), long)
        listen.transcribe_with(Chunky(), long[: listen.RATE])
        self.assertEqual(calls, [{"chunk_duration": listen.CHUNK_SECONDS}, {}, {"chunk_duration": None}])


class TestLoading(unittest.TestCase):
    def test_background_load_is_released_even_on_error(self) -> None:
        loader = mock.Mock(return_value="model")
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "1"}), \
                mock.patch.object(models, "release") as release, self.assertRaises(ValueError), \
                models.loading(loader, "some/repo") as pending:
            self.assertEqual(pending.result(), "model")
            raise ValueError
        release.assert_called_once()

    def test_refuses_before_loading_under_pressure(self) -> None:
        loader = mock.Mock()
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}), \
                self.assertRaises(models.MemoryPressure), models.loading(loader, "some/repo"):
            pass
        loader.assert_not_called()


class TestSpeak(TempState):
    def test_holds_the_lock_while_it_speaks_and_releases_it_after(self) -> None:
        def play(*args) -> None:
            self.assertEqual(state.holder(speak.PIDFILE), os.getpid())

        with mock.patch.object(speak, "play", side_effect=play) as player, mock.patch("signal.signal"):
            self.assertEqual(speak.run("hello"), 0)
        player.assert_called_once_with("hello", False, None, None)
        self.assertIsNone(state.holder(speak.PIDFILE))

    def test_stop_signals_only_a_live_speaker(self) -> None:
        state.folder()
        speak.PIDFILE.write_text("4242")  # stale: its speaker is gone, and the pid may be anything now
        with mock.patch.object(speak.os, "kill") as kill:
            speak.run(None, stop=True)
        kill.assert_not_called()
        self.held(speak.PIDFILE, 4242)
        with mock.patch.object(speak.os, "kill") as kill:
            speak.run(None, stop=True)
        kill.assert_called_once_with(4242, speak.signal.SIGTERM)

    def test_flag_beats_environment_beats_default_and_accent_follows_the_voice(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_NAME": "am_michael", "VOICE_SPEED": "1.2"}):
            self.assertEqual(speak.settings(), ("am_michael", 1.2, "a"))
            self.assertEqual(speak.settings("bf_emma", 0.9), ("bf_emma", 0.9, "b"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(speak.settings(), ("af_heart", 1.0, "a"))

    def test_a_voice_plays_from_the_loaded_models_own_folder_and_nowhere_else(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "voices").mkdir()
            (Path(tmp) / "voices" / "bf_emma.safetensors").write_bytes(b"")
            model = mock.Mock()
            model.generate.return_value = []

            @contextlib.contextmanager
            def loaded(*args):
                yield model

            with mock.patch.dict(os.environ, {"VOICE_MODEL": tmp}), \
                    mock.patch.dict(sys.modules, {"sounddevice": mock.Mock()}), \
                    mock.patch.object(models, "loaded", loaded):
                self.assertEqual(speak.voices(), ["bf_emma"])
                speak.play("hello", voice="bf_emma")
                for name in ("af_missing", "../voices/bf_emma", "/etc/x.safetensors"):
                    with self.assertRaises(SystemExit):
                        speak.play("hello", voice=name)
        kwargs = model.generate.call_args.kwargs
        # a path, so mlx-audio neither falls back to its own default repo nor tries a download
        self.assertEqual(kwargs["voice"], str(Path(tmp) / "voices" / "bf_emma.safetensors"))
        self.assertEqual(kwargs["lang_code"], "b")
        model.generate.assert_called_once()

    def test_a_bad_speed_setting_is_a_clear_error(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_SPEED": "fast"}), self.assertRaises(SystemExit) as caught:
            speak.settings()
        self.assertIn("VOICE_SPEED", str(caught.exception))

    def test_empty_text_plays_nothing(self) -> None:
        with mock.patch.object(speak, "play") as play:
            self.assertEqual(speak.run("   "), 0)
        play.assert_not_called()

    def test_the_next_sentence_is_made_while_the_last_one_plays(self) -> None:
        events: list[str] = []

        def generate(*args, **kwargs):
            for n in (1, 2, 3):
                events.append(f"made {n}")
                yield SimpleNamespace(audio=n, sample_rate=24000)

        model = SimpleNamespace(generate=generate)
        sd = SimpleNamespace(play=lambda audio, rate: events.append(f"play {audio}"),
                             wait=lambda: events.append("wait"))

        @contextlib.contextmanager
        def loaded(*args):
            yield model

        with mock.patch.dict(sys.modules, {"sounddevice": sd}), mock.patch.object(models, "loaded", loaded), \
                mock.patch.object(speak, "voice_file", return_value="/v/af_heart.safetensors"):
            speak.play("three sentences")
        self.assertEqual(events, ["made 1", "wait", "play 1", "made 2", "wait", "play 2",
                                  "made 3", "wait", "play 3", "wait"])

    def test_the_speaker_lets_a_stop_end_it_at_once(self) -> None:
        # A Python handler would wait for the playback call to return; the default ends it now.
        with mock.patch.object(speak, "play"), mock.patch.object(speak.signal, "signal") as handler:
            speak.run("hello")
        self.assertNotIn(speak.signal.SIGTERM, [c.args[0] for c in handler.call_args_list])

    def test_an_older_request_still_waiting_steps_aside(self) -> None:
        real_hold = state.hold

        def hold(path, wait=False):
            speak.TICKET.write_text("a newer request")  # arrived while this one waited for the lock
            return real_hold(path, wait)

        with mock.patch.object(speak, "play") as play, mock.patch.object(speak.state, "hold", side_effect=hold):
            self.assertEqual(speak.run("old news"), 0)
        play.assert_not_called()

    def test_stop_also_cancels_requests_waiting_their_turn(self) -> None:
        speak.run(None, stop=True)
        self.assertEqual(speak.TICKET.read_text(), "stop")

    def test_before_setup_speak_says_to_run_setup(self) -> None:
        with mock.patch.object(speak, "voice_folder", return_value=None), \
                self.assertRaises(models.ModelMissing) as caught:
            speak.voice_file("af_heart")
        self.assertIn("voice setup", str(caught.exception))

    def test_a_speed_setting_out_of_range_is_a_clear_error(self) -> None:
        for value in ("0", "9", "nan"):
            with mock.patch.dict(os.environ, {"VOICE_SPEED": value}), self.assertRaises(SystemExit):
                speak.settings()


class TestShrink(unittest.TestCase):
    """A tiny real model folder, so the cast runs for real without loading a speech model."""

    def setUp(self) -> None:
        import mlx.core as mx

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.src = Path(tmp.name) / "src" / "parakeet-tiny"
        self.src.mkdir(parents=True)
        mx.save_safetensors(str(self.src / "model.safetensors"),
                            {"w": mx.ones((4, 4), dtype=mx.float32), "steps": mx.arange(3, dtype=mx.int32)})
        (self.src / "config.json").write_text('{"model_type": "tdt"}')
        (self.src / "tokenizer.model").write_bytes(b"tok")
        self.models = Path(tmp.name) / "share" / "voice" / "models"
        for patcher in (mock.patch.object(models, "MODELS", self.models),
                        mock.patch.dict(os.environ, {"VOICE_STT_MODEL": str(self.src), "VOICE_PRESSURE_LEVEL": "1"})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_shrink(self) -> str:
        with mock.patch("sys.stdout", new_callable=io.StringIO) as out, \
                mock.patch("sys.stderr", new_callable=io.StringIO):
            self.assertEqual(shrink.run(), 0)
        return out.getvalue()

    def test_writes_a_16_bit_copy_and_prints_the_setting_that_uses_it(self) -> None:
        import mlx.core as mx

        out = self.run_shrink()
        dst = self.models / "parakeet-tiny-bf16"
        weights = mx.load(str(dst / "model.safetensors"))
        self.assertEqual((weights["w"].dtype, weights["steps"].dtype), (mx.bfloat16, mx.int32))
        self.assertEqual((dst / "config.json").read_text(), '{"model_type": "tdt"}')
        self.assertEqual((dst / "tokenizer.model").read_bytes(), b"tok")
        self.assertEqual(out.strip(), f"export VOICE_STT_MODEL={dst}")
        self.assertEqual(sorted(p.name for p in self.models.iterdir()), ["parakeet-tiny-bf16"])  # no partial left

    def test_a_second_run_reuses_the_copy(self) -> None:
        first = self.run_shrink()
        stamp = (self.models / "parakeet-tiny-bf16" / "model.safetensors").stat().st_mtime_ns
        self.assertEqual(self.run_shrink(), first)
        self.assertEqual((self.models / "parakeet-tiny-bf16" / "model.safetensors").stat().st_mtime_ns, stamp)

    def test_a_model_already_in_16_bit_is_left_alone(self) -> None:
        dst = self.models / "parakeet-tiny-bf16"
        self.run_shrink()
        with mock.patch.dict(os.environ, {"VOICE_STT_MODEL": str(dst)}):
            self.assertIn(f"export VOICE_STT_MODEL={dst}", self.run_shrink())
        self.assertEqual(sorted(p.name for p in self.models.iterdir()), ["parakeet-tiny-bf16"])

    def test_critical_pressure_refuses_before_writing(self) -> None:
        with mock.patch.dict(os.environ, {"VOICE_PRESSURE_LEVEL": "4"}), self.assertRaises(models.MemoryPressure):
            shrink.run()
        self.assertFalse(self.models.exists())

    def test_the_copy_is_private_and_a_side_file_linking_elsewhere_is_left_out(self) -> None:
        outside = self.src.parent / "outside.txt"
        outside.write_text("not part of the model")
        (self.src / "vocab.txt").symlink_to(outside)
        self.run_shrink()
        dst = self.models / "parakeet-tiny-bf16"
        self.assertEqual(dst.stat().st_mode & 0o777, 0o700)
        self.assertFalse((dst / "vocab.txt").exists())
        self.assertTrue((dst / "tokenizer.model").exists())

    def test_the_printed_setting_is_quoted_for_the_shell(self) -> None:
        spaced = self.src.parent / "parakeet tiny"
        self.src.rename(spaced)
        with mock.patch.dict(os.environ, {"VOICE_STT_MODEL": str(spaced)}):
            out = self.run_shrink()
        self.assertEqual(shlex.split(out.strip()), ["export", f"VOICE_STT_MODEL={self.models / 'parakeet tiny-bf16'}"])

    def test_a_folder_without_weights_is_an_error_not_a_copy(self) -> None:
        (self.src / "model.safetensors").unlink()
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(shrink.run(), 1)
        self.assertIn("no weights", err.getvalue())
        self.assertFalse(self.models.exists())


class TestSetup(unittest.TestCase):
    def test_setup_downloads_even_when_the_environment_says_offline(self) -> None:
        # huggingface_hub reads HF_HUB_OFFLINE once, at import; setup must clear it first.
        probe = ("from voice import setup; setup.hub(); import huggingface_hub.constants as c; "
                 "print(c.HF_HUB_OFFLINE)")
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                env={**os.environ, "HF_HUB_OFFLINE": "1"})
        self.assertEqual(result.stdout.strip(), "False", result.stderr)

    def test_setup_fetches_pinned_revisions_skips_local_folders_and_says_what_next(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"VOICE_STT_MODEL": tmp}), \
                mock.patch("huggingface_hub.snapshot_download", return_value="/cache/snap") as hub, \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(setup.run(), 0)
        (call,) = hub.call_args_list  # the local speech-to-text folder needs no download
        self.assertEqual(call.args, (models.TTS_MODEL,))
        self.assertEqual(call.kwargs, {"revision": models.REVISIONS[models.TTS_MODEL]})
        self.assertIn("listen --toggle", err.getvalue())  # the line a keyboard shortcut runs
        self.assertIn("voice shrink", err.getvalue())


class TestCli(unittest.TestCase):
    def test_shrink_runs_offline(self) -> None:
        with mock.patch("voice.shrink.run", return_value=0) as run:
            self.assertEqual(cli.main(["shrink", "--force"]), 0)
        run.assert_called_once_with(force=True)
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")

    def test_speak_reads_stdin(self) -> None:
        with mock.patch("voice.speak.run", return_value=0) as run, \
                mock.patch("sys.stdin", io.StringIO("from stdin\n")):
            self.assertEqual(cli.main(["speak"]), 0)
        run.assert_called_once_with("from stdin\n", stop=False, force=False, voice=None, speed=None)

    def test_configured_flags_and_text_on_stdin_reach_the_speaker(self) -> None:
        # the Claude Code hook runs the configured command, flags included, and writes the text to stdin
        with mock.patch("voice.speak.run", return_value=0) as run, \
                mock.patch("sys.stdin", io.StringIO("the reply")):
            self.assertEqual(cli.main(["speak", "--voice", "bf_emma", "--speed", "1.1"]), 0)
        run.assert_called_once_with("the reply", stop=False, force=False, voice="bf_emma", speed=1.1)

    def test_every_command_but_setup_runs_offline_before_any_import(self) -> None:
        probe = ("import os, sys; from unittest import mock; from voice import cli\n"
                 "with mock.patch('voice.speak.run', return_value=0):\n"
                 "    cli.main(['speak', 'x'])\n"
                 "print(os.environ.get('HF_HUB_OFFLINE'), 'huggingface_hub' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                                env={**os.environ, "HF_HUB_OFFLINE": "0"})
        self.assertEqual(result.stdout.split()[0], "1", result.stderr)
        with mock.patch("voice.setup.run", return_value=0), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_HUB_OFFLINE", None)
            cli.main(["setup"])
            self.assertNotIn("HF_HUB_OFFLINE", os.environ)

    def test_the_background_half_gets_its_lock_descriptor(self) -> None:
        with mock.patch("voice.listen.run_until_stopped", return_value=0) as run:
            self.assertEqual(cli.main(["listen", "--until-stopped", "7", "--max-seconds", "1800", "--paste",
                                       "--preload"]), 0)
        run.assert_called_once_with(1800.0, True, False, True, 7, True)

    def refused(self, argv: list[str]) -> str:
        """The error argparse prints for a refused command line (exit 2)."""
        with mock.patch("sys.stderr", new_callable=io.StringIO) as err, self.assertRaises(SystemExit) as caught:
            cli.main(argv)
        self.assertEqual(caught.exception.code, 2, argv)
        return err.getvalue()

    def test_numbers_must_be_finite_and_in_range(self) -> None:
        for argv in (["listen", "--max-seconds", "0"], ["listen", "--max-seconds", "nan"],
                     ["listen", "--max-seconds", "inf"], ["listen", "--silence", "0.01"],
                     ["speak", "--speed", "0", "x"], ["speak", "--speed", "nan", "x"], ["speak", "--speed", "9", "x"]):
            self.assertIn("between", self.refused(argv))

    def test_flags_that_would_do_nothing_together_are_refused(self) -> None:
        for argv in (["listen", "--toggle", "--no-copy"], ["listen", "--paste"], ["listen", "--preload"],
                     ["listen", "--toggle", "--silence", "2"], ["speak", "--stop", "some text"]):
            self.assertTrue(self.refused(argv), argv)

    def test_speak_on_a_terminal_with_no_text_shows_usage_instead_of_waiting(self) -> None:
        tty = io.StringIO()
        tty.isatty = lambda: True
        with mock.patch("voice.speak.run") as run, mock.patch("sys.stdin", tty), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(cli.main(["speak"]), 2)
        run.assert_not_called()
        self.assertIn("usage", err.getvalue())

    def test_overlong_text_on_stdin_is_refused(self) -> None:
        with mock.patch("voice.speak.run") as run, mock.patch("sys.stdin", io.StringIO("x" * 100_001)), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(cli.main(["speak"]), 2)
        run.assert_not_called()
        self.assertIn("100000", err.getvalue().replace(",", ""))

    def test_a_missing_model_exits_4_and_names_setup(self) -> None:
        missing = models.ModelMissing("x is not on this machine; run `voice setup` first")
        with mock.patch("voice.listen.run", side_effect=missing), \
                mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            self.assertEqual(cli.main(["listen"]), 4)
        self.assertIn("voice setup", err.getvalue())

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
