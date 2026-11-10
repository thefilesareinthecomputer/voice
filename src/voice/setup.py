"""`voice setup`: download the default models into the Hugging Face cache, once, at the
revisions this version was tested against. The other commands run offline and never download.

In a terminal it then walks a new user through the rest: it plays a sentence, listens to one,
checks for Claude Code, and offers to open the Shortcuts app for the keyboard shortcuts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import ModuleType

from voice import models

CLAUDE_CODE = "https://code.claude.com/docs/en/overview"


def hub() -> ModuleType:
    """huggingface_hub, imported online. It reads HF_HUB_OFFLINE once, at import, so the
    variable goes first, whatever the caller's environment set."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    import huggingface_hub

    return huggingface_hub


def download() -> None:
    snapshot_download = hub().snapshot_download
    for repo in (models.tts_model(), models.stt_model()):
        if os.path.isabs(os.path.expanduser(repo)):
            print(f"{repo} is a local folder; nothing to fetch", file=sys.stderr)
            continue
        print(f"fetching {repo}", file=sys.stderr)
        path = snapshot_download(repo, revision=models.REVISIONS.get(repo))
        print(f"  ready in {path}", file=sys.stderr)


def say(line: str = "") -> None:
    print(line, file=sys.stderr)


def ask() -> str:
    """One line from the user; Control+D counts as no answer."""
    try:
        return input()
    except EOFError:
        return ""


def check_speaker() -> None:
    from voice import speak

    say("\n1. Speaker. You should hear one sentence.")
    try:
        speak.run("Voice is set up. This is how I sound.")
    except models.MemoryPressure:
        say("   Skipped: memory is low right now. Try `voice speak \"hello\"` later.")


def check_microphone() -> None:
    from voice import listen

    say("\n2. Microphone. Press Enter, say a few words, then pause.")
    ask()
    try:
        with models.loaded(models.load_stt, models.stt_model()) as model:
            say("   listening...")
            text = listen.transcribe_with(model, listen.record(30, 1.5, patience=10))
    except models.MemoryPressure:
        say("   Skipped: memory is low right now. Try `voice listen` later.")
        return
    if text:
        say(f"   Heard: {text}")
    else:
        say("   Heard nothing. Allow your terminal in System Settings > Privacy & Security > Microphone,"
            " then run `voice listen`.")


def check_claude() -> None:
    if shutil.which("claude"):
        say("\n3. Claude Code is installed. In any project folder, run `voice chat` to talk with it.")
    else:
        say(f"\n3. Claude Code is not installed. Install it for `voice chat`: {CLAUDE_CODE}")


def offer_shortcuts() -> None:
    voice = shutil.which("voice") or "~/.local/bin/voice"
    say("\n4. Keyboard shortcuts (optional). In the Shortcuts app, turn on Settings > Advanced >")
    say("   Allow Running Scripts. Then make one shortcut per command, each with a single")
    say("   Run Shell Script action, and give it a key:")
    say(f"     dictation:   {voice} listen --toggle")
    say(f"     voice chat:  {voice} chat --toggle")
    say("   The README's Dictation section has the steps.")
    say("   Open the Shortcuts app now? [y/N] ")
    if ask().strip().lower().startswith("y"):
        subprocess.run(["/usr/bin/open", "-a", "Shortcuts"], check=False)


def run() -> int:
    download()
    if not sys.stdin.isatty():
        say("\nready. `voice speak \"hello\"` and `voice listen` check the speaker and the microphone.")
        return 0
    say("\nModels ready. Four quick steps.")
    check_speaker()
    check_microphone()
    check_claude()
    offer_shortcuts()
    say("\nAll set. `voice shrink` makes a smaller copy of the dictation model if you want to save memory.")
    return 0
