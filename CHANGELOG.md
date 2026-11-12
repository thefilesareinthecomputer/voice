# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `install.sh` installs `voice` pinned to the tested versions and runs the setup, so installing is clone, `cd` and `./install.sh`. Run it again after `git pull` to update.
- `voice setup` walks a new user through the rest in a terminal: it plays a sentence, listens to one, checks for Claude Code, and shows the keyboard shortcuts with an offer to open the Shortcuts app.
- `voice chat` runs Claude Code in the terminal with a hands-free spoken loop, and it is the one way to use `voice` with Claude Code: the earlier `/voice` hook and command for ordinary sessions are gone. Claude speaks a short summary of each reply, the microphone opens once playback ends, and a pause of three seconds, or whatever `--silence` sets, submits what you said. Any key takes over from the loop, and Control+Q turns it off and on. `voice chat --toggle` does the same from a global keyboard shortcut. It needs no change to `~/.claude`. Claude is asked for the spoken summary only while the loop is on; with the loop off, Claude Code works as it always does.
- A background speaker: `voice speak --warm` keeps Kokoro loaded, so speech starts in about half a second instead of about four, and `voice speak --release` frees it. `voice serve` runs it in the foreground.
- `voice listen --toggle` for a keyboard shortcut: the first press starts a background recording with a sound and a notification, the next press stops it, and the text is transcribed and copied. No silence cutoff; a recording stops on its own after 30 minutes.
- `--paste` also pastes the text at the cursor when a toggle recording stops, and keeps it on the clipboard. It is opt-in; the README explains the System Events permission it asks for.
- Recordings longer than a minute are transcribed in chunks where the model supports it.
- `voice shrink` writes a 16-bit copy of the speech-to-text model: half the size on disk and about 0.3 GB less memory while it transcribes.
- `--preload` with `--toggle` loads the model while you talk, for a shorter wait after the stop press.
- `VOICE_TTS_MODEL` names the text-to-speech model; `VOICE_MODEL` still works.
- A Troubleshooting section in the README.

### Changed

- The background speaker stops and frees its model after ten minutes with no request, and `voice chat` runs its cleanup when the terminal window closes or the process is killed. Before, a session that ended that way left the speaker loaded until `voice speak --release`.
- In `voice chat`, what you say while Claude is mid-turn is no longer typed into the terminal, where a question or permission dialog on screen would take the text and drop it. The message is held and sent when the turn ends, and a question or permission dialog counts as a turn even when another agent started it. Every message is also written to `~/.local/state/claude-voice/chat-<pid>.unsent.txt` before it is typed and removed only once Claude Code reports that same message as a prompt, so a message that does not arrive, or a session that is killed, leaves the text on disk.
- The start sound and notification come once the microphone is open, and notifications no longer hold up the recording, so the first words are no longer lost.
- The microphone ignores what it hears while the start sound plays. Through built-in speakers that sound counted as speech, so a pause ended the turn before you said a word.
- In `voice chat`, a pause after a sound with no words (a click, a cough, room noise) keeps listening instead of ending the turn; mouse movement and clicks no longer stop the speech or the listening; and turning a loop on switches any other running voice chat to plain typing.
- The background speaker and the microphone recover when the audio device changes or the Mac wakes from sleep, instead of failing every time until restarted.
- `voice chat` no longer goes quiet after Escape or Control+C stops Claude, after a command such as `/clear`, or after a turn Claude starts by itself; a microphone or model failure says so instead of reading as silence; a speaker that died is started again; and stopping speech never freezes the terminal. A turn Claude starts by itself no longer cuts off the summary playing or you mid-sentence, a reply that lands during a summary plays after it, and Control+Q on a loop left waiting starts listening instead of turning it off.
- MLX and the Parakeet code import while you talk, which shortens the wait after the stop press.
- Takes transcribe one at a time, so a quick second take never loads a second model or overwrites a newer clipboard.
- A start that is refused (critical memory pressure, a missing model) says so with the error sound and a notification.
- Critical memory pressure at the stop press waits up to a minute and then transcribes, instead of discarding the recording.
- A silent take skips the model, and an all-zero one points at the microphone permission.
- A missing model says to run `voice setup`, for `listen`, `speak` and `voices` alike; `setup` skips local folders and prints what to do next.
- The 16-bit copy from `voice shrink` is used without a setting once it exists.
- `voice speak` makes each sentence while the last one plays, a stop ends playback at once, and a request that waited for its turn steps aside for a newer one or a stop.
- Numbers on the command line must be finite and in range, and flag combinations that would do nothing (`--toggle --no-copy`, `--paste` without `--toggle`, `--stop` with text) are refused. `voice speak` with no text on a terminal shows its usage instead of waiting.

### Security

- Models load only through the Kokoro and Parakeet loaders, whatever `VOICE_MODEL` or `VOICE_STT_MODEL` names, so a model can no longer steer mlx-audio into a family that runs code shipped with it.
- A pid file counts only while its process holds a lock on it, so the pid of an exited process is never signalled, a pid of 1 or less never is, and two quick presses can no longer start two recorders. The background half accepts only its own lock as `--until-stopped`, and a symlinked state folder is refused.
- The default models download and load at pinned revisions, and every load gets a resolved folder, so a folder in the current directory can never stand in for a repo name.
- `voice shrink` copies a model's side files only from the model folder or its own cache blobs, writes a private folder, and quotes the setting it prints.
- A failure notification names the log file instead of quoting the error, and the log records the error's type and stack without its message, so neither can hold what was said.
- The CI vulnerability audit runs on macOS, where the MLX packages' darwin-only markers match, and checkout keeps no credentials.
- The background recorder starts in isolated mode (`python -I`) from the state folder, so it never imports from the folder the shortcut ran in or from `PYTHONPATH`.
- `pbcopy`, `osascript`, `afplay` and `sysctl` run by absolute path, never through `PATH`.
- The state folder is readable only by its owner, and the pid and log files refuse symlinks.
- Notifications give a word count instead of the start of the transcript, because macOS keeps notification text.
- A transcript goes on the clipboard marked `org.nspasteboard.ConcealedType`, so clipboard-history apps that follow the convention leave it out of their history; plain `pbcopy` remains the fallback.
- The spaCy model's download URL includes its hash, and the README installs from the lock.
- A voice plays from the loaded model's own `voices` folder, and only a voice `voice voices` lists; before, mlx-audio looked the name up in its own default repo, so most voices failed offline.

## 0.1.0 - 2026-11-06

### Added

- `voice speak` reads text aloud with Kokoro-82M, from arguments or stdin; a new call stops the one already playing, and `--stop` stops it outright.
- `--voice` and `--speed` pick the speaking voice and speed for a call, and `voice voices` lists the voices in the local cache. The accent follows the voice name.
- `voice listen` records until a pause, transcribes with Parakeet TDT 0.6B v2, prints the text and copies it to the clipboard. The model loads while you speak.
- `voice setup` downloads the default models once; every other command runs offline.
- Each command loads one model in its own process and releases it before exiting, and refuses to load under critical memory pressure unless `--force` is given.
