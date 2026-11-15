# AGENT INSTRUCTIONS

This repo is `voice`: local speech-to-text and text-to-speech on Apple Silicon, with a hands-free spoken loop for Claude Code (`voice chat`). `README.md` says how to use it, `CONTRIBUTING.md` holds the design rules and what every change needs, and `CHANGELOG.md` lists changes under Unreleased. Read those rather than restating them here.

If present, `tasks/todo.md` (where things stand) and `tasks/plan.md` (the active work) come first. Both are local and gitignored.

## Working here

- Run the tests with `uv run python -m unittest discover -s tests` and lint with `uv run ruff check`.
- Whether the `voice` command on PATH runs this clone's code depends on how it was installed. Check before giving update or restart steps, and never run `./install.sh` in a development clone. "Running your changes" in `CONTRIBUTING.md` has the check and the steps for a development clone, and "Updating" in `README.md` has them for an ordinary install.
- Keep every change backward compatible. Before adding a mechanism, check that it neither duplicates nor contradicts one already there, for example the start-sound handling and the noise floor in `listen.record`.
- Work on `develop`; `main` only fast-forwards from it.
