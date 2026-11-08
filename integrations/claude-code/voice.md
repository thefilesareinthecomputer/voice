---
description: Turn the spoken reply summaries on or off, or say whether they are on. Fired by the user as /voice on, /voice off or /voice status.
argument-hint: on | off | status
disable-model-invocation: true
allowed-tools: Bash(python3 ~/.claude/hooks/voice_reply.py on), Bash(python3 ~/.claude/hooks/voice_reply.py off), Bash(python3 ~/.claude/hooks/voice_reply.py status)
---

Run exactly one command, with `on`, `off` or `status` taken from the arguments below (`status` when they are empty or name none of the three):

    python3 ~/.claude/hooks/voice_reply.py <on|off|status>

Reply with the line it prints and nothing else.
The command changes only whether voice is on, and the change applies from the next reply without a restart.
If it prints that no speaker command is configured, say that the speaker is set in `~/.claude/voice.json` and stop; do not create or edit that file.

Arguments: $ARGUMENTS
