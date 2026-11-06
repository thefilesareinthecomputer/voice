"""`voice setup`: download the default models into the Hugging Face cache, once.
The other commands run offline and never download."""

from __future__ import annotations

import os
import sys

from voice import models


def run() -> int:
    from huggingface_hub import snapshot_download

    os.environ.pop("HF_HUB_OFFLINE", None)
    for repo in (models.tts_model(), models.stt_model()):
        print(f"fetching {repo}", file=sys.stderr)
        path = snapshot_download(repo)
        print(f"  ready in {path}", file=sys.stderr)
    return 0
