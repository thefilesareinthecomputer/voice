"""Model choice, the memory check before a load, and the release after it.

Every command loads its model inside one short-lived process and releases it in
`finally`, so nothing stays resident once the command returns.
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

# Defaults are overridden by the environment, never by a config file.
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"  # English only, which is what dictation here needs

# The kernel's own verdict, the level behind Activity Monitor's pressure graph.
PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4


class MemoryPressure(RuntimeError):
    """Raised instead of loading a model while the machine is under critical pressure."""


def tts_model() -> str:
    return os.environ.get("VOICE_MODEL", TTS_MODEL)


def stt_model() -> str:
    return os.environ.get("VOICE_STT_MODEL", STT_MODEL)


def pressure_level() -> int:
    """The kernel's memory pressure level, or normal where it cannot be read."""
    forced = os.environ.get("VOICE_PRESSURE_LEVEL")
    if forced:
        return int(forced)
    try:
        out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return PRESSURE_NORMAL


def check_pressure(force: bool = False) -> None:
    """Refuse a load at critical pressure, where a few hundred megabytes tips the machine into swap."""
    if pressure_level() >= PRESSURE_CRITICAL and not force:
        raise MemoryPressure("memory pressure is critical; close something and retry, or pass --force")


def release() -> None:
    """Drop MLX's cached buffers and collect, so the memory returns even before the process exits."""
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear:
        clear()


@contextlib.contextmanager
def loaded(loader: Any, repo: str, force: bool = False) -> Iterator[Any]:
    """Load one model from the local cache for the length of a block, then release it."""
    check_pressure(force)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = loader(repo)
    try:
        yield model
    finally:
        del model
        release()


@contextlib.contextmanager
def loading(loader: Any, repo: str, force: bool = False) -> Iterator[Future]:
    """Start loading one model in the background for the length of a block, then release it.
    The block does its own work meanwhile (recording speech) and calls `.result()` for the model."""
    check_pressure(force)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    pool = ThreadPoolExecutor(max_workers=1)
    pending = pool.submit(loader, repo)
    try:
        yield pending
    finally:
        pool.shutdown(wait=True)
        del pending
        release()
