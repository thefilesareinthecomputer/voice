"""Model choice, the memory check before a load, and the release after it.

Every command loads its model inside one short-lived process. `release` drops MLX's
cached buffers when the block ends; the weights themselves go when the process exits,
moments later, so nothing stays resident once the command returns.
"""

from __future__ import annotations

import contextlib
import gc
import os
import subprocess
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from voice import state

# Defaults are overridden by the environment, never by a config file.
TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"  # English only, which is what dictation here needs
# The commits of the defaults this version was tested against: setup fetches these and every
# command reads these, so a later upload to the same repo changes nothing until a release says so.
REVISIONS = {
    TTS_MODEL: "a71e4d38b236d968966a2002c4c895dbd12b1c3c",
    STT_MODEL: "8ae155301e23d820d82aa60d24817c900e69e487",
}
# Where `voice shrink` writes its 16-bit copies; the default model's copy is used once it exists.
MODELS = state.xdg(os.environ, "XDG_DATA_HOME", ".local/share") / "voice" / "models"

# The only model families voice loads, whatever the model setting names. Left to itself, mlx-audio
# picks the family, and with it the code that runs, from the model's own config.json and folder
# name, and several families run code shipped inside the model. These two read weights only.
TTS_FAMILY, STT_FAMILY = "kokoro", "parakeet"

SYSCTL = "/usr/sbin/sysctl"

# The kernel's own verdict, the level behind Activity Monitor's pressure graph.
PRESSURE_NORMAL, PRESSURE_WARN, PRESSURE_CRITICAL = 1, 2, 4


class MemoryPressure(RuntimeError):
    """Raised instead of loading a model while the machine is under critical pressure."""


class ModelMissing(RuntimeError):
    """Raised instead of a library error when a model is not on this machine."""


def tts_model() -> str:
    """VOICE_TTS_MODEL, or VOICE_MODEL, its older name."""
    return os.environ.get("VOICE_TTS_MODEL") or os.environ.get("VOICE_MODEL") or TTS_MODEL


def shrunk(repo: str) -> Path:
    """Where `voice shrink` puts the 16-bit copy of a repo or folder."""
    return MODELS / f"{Path(repo).expanduser().name}-bf16"


def stt_model() -> str:
    """VOICE_STT_MODEL, or the 16-bit copy of the default once `voice shrink` has made it."""
    if os.environ.get("VOICE_STT_MODEL"):
        return os.environ["VOICE_STT_MODEL"]
    copy = shrunk(STT_MODEL)
    return str(copy) if copy.is_dir() else STT_MODEL


def folder(repo: str) -> Path | None:
    """A model's files: an absolute (or ~) folder as given, or a repo's snapshot in the local
    cache at its pinned revision. A relative name is always a repo id, never a folder in the
    current directory. None when it is neither, as before `voice setup`."""
    path = Path(repo).expanduser()
    if path.is_absolute():
        return path if path.is_dir() else None
    from huggingface_hub import snapshot_download

    try:
        return Path(snapshot_download(repo, revision=REVISIONS.get(repo), local_files_only=True))
    except Exception:
        return None


def require(repo: str) -> Path:
    path = folder(repo)
    if path is None:
        raise ModelMissing(f"{repo} is not on this machine; run `voice setup` first")
    return path


def pressure_level() -> int:
    """The kernel's memory pressure level, or normal where it cannot be read."""
    forced = os.environ.get("VOICE_PRESSURE_LEVEL", "")
    if forced.isdigit():
        return int(forced)
    try:
        out = subprocess.run([SYSCTL, "-n", "kern.memorystatus_vm_pressure_level"],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return PRESSURE_NORMAL


def check_pressure(force: bool = False) -> None:
    """Refuse a load at critical pressure, where a few hundred megabytes tips the machine into swap."""
    if pressure_level() >= PRESSURE_CRITICAL and not force:
        raise MemoryPressure("memory pressure is critical; close something and retry, or pass --force")


def _load(category: str, family: str, repo: str, strict: bool) -> Any:
    path = require(repo)
    from mlx_audio.utils import base_load_model

    # An empty remapping and no name parts leave mlx-audio nothing to choose: the family is fixed.
    # It gets the resolved folder, so it never tries a repo name as a path itself.
    return base_load_model(str(path), category=category, model_remapping={}, model_type=family,
                           model_name_parts=[], strict=strict)


def load_tts(repo: str) -> Any:
    return _load("tts", TTS_FAMILY, repo, strict=True)


def load_stt(repo: str) -> Any:
    return _load("stt", STT_FAMILY, repo, strict=False)


def release() -> None:
    """Collect and drop MLX's cached buffers. Arrays the caller still holds stay until the
    process exits, which follows at once for every command here."""
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
