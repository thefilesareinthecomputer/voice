#!/usr/bin/env python3
"""UserPromptSubmit + Stop - advisory only, NEVER blocks.

Reads a short spoken summary of each reply aloud, through a local speaker
command the user configures. Off unless the config file says enabled.

The model decides what is worth hearing, not the hook: while voice is on,
UserPromptSubmit asks it to end the reply with one <speak>...</speak> line -
what it did or found and what it needs from the user next. Stop speaks that
line only when it ends the reply, never the status line, code, the full reply
or a tag quoted from a file mid-reply. A reply without one is silent.

UserPromptSubmit also stops whatever is still playing, so a new message cuts
the old answer off.

Interactive terminal sessions only (CLAUDE_CODE_ENTRYPOINT=cli): a headless
`claude -p` run (the bench, /task, app seats) gets no instruction and no
speech, so its output stays unchanged.

Config: ~/.claude/voice.json, and nowhere else:
    {"enabled": true, "command": ["/path/to/speaker", "..."]}
It names a command every session runs, so it lives inside ~/.claude, beside
the settings, and an edit to it deserves the same care as an edit to them. No
variable of the hook's own can point it at another file; only HOME, which also
decides which settings.json runs the hook, moves it.
The command gets the text on stdin, or "--stop" as its one argument. The text
never goes in argv, where any local account could read it in ps and the
speaker could take it for an option. It runs detached, so the hook returns at
once; the speaker owns its own lifetime and single-instance playback.

`voice_reply.py on|off|status` is the /voice command's toggle: it rewrites
only `enabled`; `on` says "Voice is on." aloud and `off` stops playback.

As a hook it fails open by construction: no config, a bad config, a missing
command, a malformed payload - every path exits 0 with no output. Stdlib only.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

CONFIG = Path.home() / ".claude" / "voice.json"
# The last <speak> line, and only when nothing but whitespace follows it.
SPEAK = re.compile(r"<speak>((?:(?!<speak>).)*?)</speak>\s*\Z", re.DOTALL | re.IGNORECASE)
# Markdown marks; an underscore only where it opens or closes a word, so `voice_reply` survives.
MARKS = re.compile(r"[*`#>]+|(?<!\w)_+|_+(?!\w)")
CONTROL = re.compile(r"[\x00-\x1f\x7f]")
MAX_CHARS = 400

INSTRUCTION = (
    "Voice is on: the user listens to a spoken summary instead of reading. End "
    "every reply with one line <speak>...</speak>: one to three plain spoken "
    "sentences saying what you did or found and what you need from the user next. "
    "No file paths, code, commands, URLs or symbols in it; say them in words or "
    "say they are on screen. The line is read aloud and the rest is not."
)


def read_config(path: Path) -> dict | None:
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


def speaker(cfg: dict) -> list[str] | None:
    cmd = cfg.get("command")
    if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) for c in cmd):
        return None
    return [os.path.expanduser(c) for c in cmd]


def load_config() -> list[str] | None:
    cfg = read_config(CONFIG)
    if cfg is None or cfg.get("enabled") is not True:
        return None
    return speaker(cfg)


def toggle(arg: str) -> int:
    """`voice_reply.py on|off|status`, run by the /voice command. It changes
    only `enabled`; the speaker command changes only through a guarded edit."""
    path = CONFIG
    cfg = read_config(path)
    if arg == "status":
        state = "on" if cfg and cfg.get("enabled") is True and speaker(cfg) else "off"
        print(f"Voice is {state}, for interactive sessions only; a headless claude -p run stays silent."
              if cfg else f"Voice is off: no config at {path}.")
        return 0
    if cfg is None or speaker(cfg) is None:
        print(f"No speaker command in {path}. Add one before turning voice on or off.")
        return 1
    cfg["enabled"] = arg == "on"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, indent=2) + "\n")
    os.chmod(tmp, path.stat().st_mode & 0o777)
    os.replace(tmp, path)
    message = f"Voice is {arg}."
    if arg == "off":
        stop(speaker(cfg))
    else:
        launch(speaker(cfg), message)  # heard at once, so the user knows it works before any reply
    print(message)
    return 0


def launch(cmd: list[str], text: str | None = None) -> None:
    """Start the speaker detached: with `text` on stdin, or with no input when None."""
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL if text is None else subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        if text is not None:
            # At most MAX_CHARS, well under a pipe's buffer, so this never waits on the speaker.
            with proc.stdin:
                proc.stdin.write(text.encode())
    except OSError:  # includes BrokenPipeError from a speaker that exits at once
        pass


def stop(cmd: list[str]) -> None:
    launch(cmd + ["--stop"])


def spoken(text: str) -> str:
    """The speak line that ends the reply, cleaned, or "" when there is none."""
    found = SPEAK.search(text)
    if not found:
        return ""
    # The text is model output: no control characters, bounded length.
    return " ".join(CONTROL.sub(" ", MARKS.sub("", found.group(1))).split())[:MAX_CHARS]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except ValueError:
        return 0
    if not isinstance(payload, dict) or os.environ.get("CLAUDE_CODE_ENTRYPOINT") != "cli":
        return 0
    cmd = load_config()
    if cmd is None:
        return 0
    event = payload.get("hook_event_name", "")

    if event == "UserPromptSubmit":
        stop(cmd)
        print(json.dumps({
            "suppressOutput": True,
            "hookSpecificOutput": {"hookEventName": event, "additionalContext": INSTRUCTION},
        }))
        return 0

    if event == "Stop":
        text = payload.get("last_assistant_message")
        line = spoken(text if isinstance(text, str) else "")
        if line:
            launch(cmd, line)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:  # the /voice command, never the harness
        if sys.argv[1:] not in (["on"], ["off"], ["status"]):
            print("usage: voice_reply.py on|off|status")
            sys.exit(2)
        sys.exit(toggle(sys.argv[1]))
    try:
        sys.exit(main())
    except Exception:  # advisory hook: never block on its own failure
        sys.exit(0)
