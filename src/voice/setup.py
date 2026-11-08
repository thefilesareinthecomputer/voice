"""`voice setup`: download the default models into the Hugging Face cache, once, at the
revisions this version was tested against. The other commands run offline and never download."""

from __future__ import annotations

import os
import shutil
import sys
from types import ModuleType

from voice import models


def hub() -> ModuleType:
    """huggingface_hub, imported online. It reads HF_HUB_OFFLINE once, at import, so the
    variable goes first, whatever the caller's environment set."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    import huggingface_hub

    return huggingface_hub


def run() -> int:
    snapshot_download = hub().snapshot_download
    for repo in (models.tts_model(), models.stt_model()):
        if os.path.isabs(os.path.expanduser(repo)):
            print(f"{repo} is a local folder; nothing to fetch", file=sys.stderr)
            continue
        print(f"fetching {repo}", file=sys.stderr)
        path = snapshot_download(repo, revision=models.REVISIONS.get(repo))
        print(f"  ready in {path}", file=sys.stderr)
    voice = shutil.which("voice") or "voice"
    print(f"\nready. Try `voice listen` or `voice speak \"hello\"`. A keyboard shortcut runs:\n"
          f"  {voice} listen --toggle\n"
          "`voice shrink` makes a 16-bit copy of the speech-to-text model: half the size, less memory.",
          file=sys.stderr)
    return 0
