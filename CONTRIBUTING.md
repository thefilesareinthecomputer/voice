# Contributing to voice

## Before you start

For anything bigger than a small fix, open an issue first and agree on the change before you build it.

## Setting up

You need an Apple Silicon Mac, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thefilesareinthecomputer/voice.git && cd voice
uv sync
uv run python -m unittest discover -s tests
uv run ruff check
```

To try your changes as the installed command, run `uv tool install --editable .` from the clone.

## What every change needs

- A test that fails without the change. The tests fake the models, microphone and speaker, so they run without a download or an audio device; keep it that way.
- The tests and `uv run ruff check` passing.
- An entry in `CHANGELOG.md` under Unreleased.
- The README updated wherever behavior changes.

## Design rules

- One process per call: a command loads its model, does one job and releases it before it exits. Nothing stays resident.
- No network at run time. Only `voice setup` downloads, and every other command loads models from the local cache.
- No model weights, audio or other large files in the repository.
