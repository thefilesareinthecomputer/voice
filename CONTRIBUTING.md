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

## Running your changes

Install `voice` editable from the clone once, pinned to the lock as in the README. The `voice` command on your PATH then runs the code in this folder:

```bash
uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
uv tool install --editable --constraints requirements.txt .
voice setup
```

Don't run `./install.sh` in a development clone. It installs a copy over the editable install, and your edits stop reaching `voice` until you run the editable install again. To see which one you have, run `grep editable "$(uv tool dir)/voice/uv-receipt.toml"`; it prints a line only for an editable install.

**After you change code.** The next `voice` command you start runs the new code. A running `voice chat` and the background speaker keep the old code, so quit `voice chat` and start it again (`voice chat --continue` resumes the conversation), and run `voice speak --release`.

**After you pull.** Run:

```bash
git pull
uv sync
uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
uv tool install --editable --constraints requirements.txt .
```

`uv sync` updates the clone's environment for the tests, and the last two commands update the installed command's package versions from `uv.lock`. They are safe to run when nothing changed. If the pull changed `REVISIONS` in `src/voice/models.py`, run `voice setup` to download the new models. Then restart `voice chat` and the speaker as above.

## What every change needs

- A test that fails without the change. The tests fake the models, microphone and speaker, so they run without a download or an audio device; keep it that way.
- The tests and `uv run ruff check` passing.
- An entry in `CHANGELOG.md` under Unreleased.
- The README updated wherever behavior changes.

## Design rules

- One process per call: a command loads its model, does one job and releases it before it exits. A model stays resident only in a mode the user turned on and can turn off: the background speaker (`voice speak --warm` and `--release`) and the `voice chat` loop.
- Check memory pressure before every model load, and never load a real model in a test.
- No network at run time. Only `voice setup` downloads, and every other command loads models from the local cache.
- No model weights, audio or other large files in the repository.
