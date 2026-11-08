"""`voice shrink`: write a 16-bit copy of the speech-to-text model.

The default Parakeet weights are 32-bit floats. Casting them to bfloat16 halves the file
(2.5 GB to 1.24 GB) and lowers the memory a transcription needs by about 0.3 GB, with the
same transcript in testing. The copy goes to its own private folder; for the default model
every command picks it up on its own, and for another one VOICE_STT_MODEL selects it.
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from pathlib import Path

from voice import models


def setting(path: Path) -> str:
    """The shell line that selects a model folder, quoted for any path."""
    return "export " + shlex.quote(f"VOICE_STT_MODEL={path}")


def side_files(src: Path) -> list[Path]:
    """The model's other files (config, tokenizer). Each must be a regular file inside the model
    folder, or, for a cached repo, inside that repo's own blob store, where the cache's links
    point; a link anywhere else is left out, so the copy never picks up an unrelated file."""
    roots = [src.resolve(), (src / ".." / ".." / "blobs").resolve()]
    files = []
    for path in sorted(src.iterdir()):
        if path.suffix == ".safetensors":
            continue
        real = path.resolve()
        if real.is_file() and any(real.is_relative_to(root) for root in roots):
            files.append(path)
    return files


def run(force: bool = False) -> int:
    # The setting or the default, never the copy itself, which stt_model() returns once it exists.
    repo = os.environ.get("VOICE_STT_MODEL") or models.STT_MODEL
    src = models.folder(repo)
    if src is None:
        print(f"voice: {repo} is not on this machine; run `voice setup` first", file=sys.stderr)
        return 1
    weights = sorted(src.glob("*.safetensors"))
    if not weights:
        print(f"voice: {src} holds no weights (*.safetensors); nothing to shrink", file=sys.stderr)
        return 1
    dst = models.shrunk(repo)
    if dst.is_dir():
        print(f"{dst} is already there", file=sys.stderr)
        print(setting(dst))
        return 0
    models.check_pressure(force)
    import mlx.core as mx

    print(f"casting {repo} to 16 bits; this needs about 3.5 GB of memory for a minute", file=sys.stderr)
    partial = dst.with_name(dst.name + ".partial")
    tensors = []
    try:
        tensors = [mx.load(str(w)) for w in weights]
        if not any(t.dtype == mx.float32 for weights_file in tensors for t in weights_file.values()):
            print(f"{repo} is already 16-bit", file=sys.stderr)
            print(setting(src))
            return 0
        shutil.rmtree(partial, ignore_errors=True)
        partial.mkdir(mode=0o700, parents=True)
        for path, weights_file in zip(weights, tensors, strict=True):
            cast = {k: v.astype(mx.bfloat16) if v.dtype == mx.float32 else v for k, v in weights_file.items()}
            mx.save_safetensors(str(partial / path.name), cast, metadata={"format": "mlx"})
            del cast
        for path in side_files(src):
            shutil.copyfile(path, partial / path.name)
        partial.rename(dst)
    finally:
        shutil.rmtree(partial, ignore_errors=True)
        del tensors
        models.release()
    size = sum(p.stat().st_size for p in dst.glob("*.safetensors")) / 1e9
    print(f"wrote {dst} ({size:.2f} GB)", file=sys.stderr)
    print(setting(dst))
    return 0
