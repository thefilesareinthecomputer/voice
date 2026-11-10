"""Shared test helpers: a temporary state folder, held pid locks and a quiet microphone."""

from __future__ import annotations

import contextlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from voice import chat, listen, serve, speak, state

VOICE = [sys.executable, "-m", "voice.cli"]


class TempState(unittest.TestCase):
    """Points the state folder and every file in it at a temporary folder."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        # Short: a socket path must fit in 104 bytes on macOS.
        self.state = Path(os.path.realpath(tmp.name)) / "cv"
        for patcher in (mock.patch.object(state, "STATE", self.state),
                        mock.patch.multiple(listen, PIDFILE=self.state / "listen.pid", LOG=self.state / "listen.log",
                                            TRANSCRIBING=self.state / "transcribe.lock"),
                        mock.patch.multiple(speak, PIDFILE=self.state / "speak.pid",
                                            TICKET=self.state / "speak.latest"),
                        mock.patch.multiple(serve, PIDFILE=self.state / "serve.pid", SOCKET=self.state / "speak.sock",
                                            LOG=self.state / "serve.log"),
                        mock.patch.object(chat, "LOG", self.state / "chat.log")):
            patcher.start()
            self.addCleanup(patcher.stop)

    def held(self, path: Path, pid: int | None) -> int:
        """Hold the lock on `path` as a running process would, optionally with its pid written."""
        fd = state.hold(path)
        self.assertIsNotNone(fd)
        if pid is not None:
            state.write_pid(fd, pid)
        self.addCleanup(close, fd)
        return fd


def close(fd: int) -> None:
    with contextlib.suppress(OSError):  # a test may have closed it already
        os.close(fd)


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
