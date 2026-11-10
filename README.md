<h1 align="center">voice</h1>

<p align="center">
  Talk with Claude Code, dictate into any app, and have text read aloud. The speech models are fully local.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-MLX-000000?logo=apple&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2EA44F">
</p>

<p align="center">
  <a href="#install"><b>Install</b></a> &nbsp;|&nbsp;
  <a href="#voice-chat"><b>Voice chat</b></a> &nbsp;|&nbsp;
  <a href="#dictation"><b>Dictation</b></a> &nbsp;|&nbsp;
  <a href="#reading-aloud"><b>Reading aloud</b></a> &nbsp;|&nbsp;
  <a href="#troubleshooting"><b>Troubleshooting</b></a> &nbsp;|&nbsp;
  <a href="#reference"><b>Reference</b></a>
</p>

<br>

`voice` is a small command-line tool for Apple Silicon Macs. It does three things:

- `voice chat` lets you talk with Claude Code hands-free. You speak, and Claude answers out loud.
- `voice listen` turns your speech into text, from a keyboard shortcut in any app.
- `voice speak` reads text aloud.

It runs open models through [mlx-audio](https://github.com/Blaizzy/mlx-audio) on your Mac. No audio or text goes to a server, and after setup it works offline.

It doesn't need expensive hardware. It runs fast on an M1 MacBook Air with 16 GB of memory, the machine it was built and tested on.

## Install

You need:

- A Mac with Apple Silicon (M1 or newer), with about 3 GB of free disk space.
- [uv](https://docs.astral.sh/uv/). Install it with `brew install uv`.
- [Claude Code](https://code.claude.com/docs/en/overview), for voice chat.

Then run:

```bash
git clone https://github.com/thefilesareinthecomputer/voice.git
cd voice
./install.sh
```

The script installs `voice` and downloads the models, about 2.9 GB, once. Then it walks you through a short setup:

1. It plays a test sentence.
2. It listens to you say a few words. Allow microphone access when macOS asks.
3. It checks that Claude Code is installed.
4. It shows the keyboard shortcuts you can add, and offers to open the Shortcuts app.

To update later, run `git pull` and then `./install.sh` again.

## Voice chat

In any project folder, run:

```bash
voice chat
```

Talk after the tick. Pause for three seconds when you're done. Claude does the work, reads you a short summary, and then listens again.

- **Continue a conversation.** `voice chat --continue` reopens the last conversation in this folder. To move a Claude Code session you have open into voice, quit it, then run this.
- **Type any time.** Any key stops the speech or the listening, and goes to Claude as usual. The mouse doesn't stop anything.
- **Control+Q.** If it's talking or listening, Control+Q turns it off. If it's quiet, Control+Q makes it listen.
- **It turns itself off** after 30 seconds with no words, or when you say "voice chat off".
- **Sounds.** A tick means it's listening. A pop means your words are on their way.
- **Use other apps.** The terminal can sit in the background. To approve a permission request, switch back to it.
- **One at a time.** Starting a second voice chat turns the first one's loop off.

`--silence 5` waits five seconds before ending your turn. Any other option goes to `claude`, such as `--model` or `--resume`.

**How it works.** `voice chat` starts Claude Code itself, so you install nothing in Claude Code and nothing in `~/.claude` changes. While the loop is on, each message you send includes a short note. It asks Claude to end its reply with one `<speak>...</speak>` line, and only that line is read aloud. While the loop is off, Claude Code works as usual. While the loop is on, voice chat uses about 2.2 GB of memory.

## Dictation

In a terminal, run `voice listen` and talk. When you pause, the text is printed and copied to the clipboard.

### From a keyboard shortcut

Press a key to start recording in any app. Press it again to stop. The text goes to your clipboard.

1. Open the **Shortcuts** app. Turn on **Settings > Advanced > Allow Running Scripts**. This lets every shortcut run shell commands, so only import shortcuts you trust.
2. Make a new shortcut with one **Run Shell Script** action. Set its script to:
   ```bash
   ~/.local/bin/voice listen --toggle
   ```
3. In the shortcut's details, choose **Add Keyboard Shortcut** and press a key combination, such as Control+Option+Command+D.
4. Press it to start. You hear a sound, and the orange microphone dot shows in the menu bar.
5. Press it again to stop. Then press Command+V to paste.

For voice chat, make a second shortcut the same way with `~/.local/bin/voice chat --toggle`. It turns the newest voice chat's loop on or off from any app.

Use the full path in Shortcuts, because Shortcuts can't find `voice` by name. `which voice` shows the path on your Mac.

A recording stops on its own after 30 minutes. The text takes a few seconds to arrive after you stop. Add `--preload` to load the model while you talk instead. The text arrives sooner, but about 1.7 GB of memory stays in use while you record.

<details>
<summary><b>Pasting automatically with <code>--paste</code>, and what it asks for</b></summary>

<br>

`voice listen --toggle --paste` also presses Command+V at your cursor when the recording stops, through System Events. macOS credits that request to whichever app is in front, so each app you dictate into asks once for permission to control System Events.

Allowing it gives that app, and any plugins or scripts it runs, lasting control of System Events, which can type into and click through every other app. If you decline, the text stays on the clipboard. The paste goes into whatever app is in front when the recording stops. For these reasons `--paste` is off unless you add it.

</details>

| `voice listen` option | Default | What it does |
|---|---|---|
| `--toggle` | off | start a background recording, or stop the one running |
| `--silence` | 1.5 s | how long a pause ends the recording, from 0.2 to 30 s; not with `--toggle` |
| `--max-seconds` | 60 s, or 30 min with `--toggle` | the longest recording, from 1 s to 3 hours |
| `--no-copy` | off | print only, leave the clipboard alone; not with `--toggle` |
| `--preload` | off | with `--toggle`, load the model while you talk |
| `--paste` | off | with `--toggle`, also paste at the cursor (see **Pasting automatically with `--paste`**) |

## Reading aloud

```bash
voice speak "The build passed and the release is ready."
echo "Text from another program" | voice speak
voice speak --voice bf_emma --speed 1.1 "A different voice, a little faster."
voice speak --stop
```

- A new `speak` stops the one already playing.
- On its own, `speak` takes about 4 seconds to start. `voice speak --warm` keeps the model loaded in the background, about 450 MB, so each `speak` starts in about half a second. `voice speak --release` frees it.
- `--speed` goes from 0.5 to 2.0.
- `voice voices` lists the 54 voices. The first letter is the accent (`a` American, `b` British). The second is `f` female or `m` male. The default is `af_heart`.

## A smaller dictation model

`voice shrink` makes a 16-bit copy of the dictation model. The copy is half the size (1.24 GB instead of 2.5 GB), uses about 0.3 GB less memory, and gave the same transcripts in testing.

```bash
voice shrink
```

It needs about 3.5 GB of free memory for a minute. After that, every command uses the copy automatically.

## Troubleshooting

- **`command not found: voice`.** Run `uv tool update-shell`, then open a new terminal window.
- **Nothing is heard, or "No sound reached the microphone."** The app that runs `voice` has no microphone access. That's your terminal, or Shortcuts for the keyboard shortcut. Turn it on in **System Settings > Privacy & Security > Microphone**.
- **No sound plays.** Run `voice speak "hello"`. If that's silent too, check your Mac's sound output.
- **The shortcut does nothing, or you hear the error sound.** A notification says why. "model missing" means you need to run `voice setup`.
- **"Waiting for memory."** Memory was very low when you stopped. Your recording is kept, and it transcribes when memory frees up.
- **"Failed."** The error is in `~/.local/state/claude-voice/`, in `listen.log` or `chat.log`. The logs never hold what you said.

## Reference

### Models

| Job | Default model | Download | License |
|---|---|---|---|
| Speech-to-text | [`mlx-community/parakeet-tdt-0.6b-v2`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2), English | about 2.5 GB | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), by NVIDIA |
| Text-to-speech | [`mlx-community/Kokoro-82M-bf16`](https://huggingface.co/mlx-community/Kokoro-82M-bf16) | about 390 MB | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M), by hexgrad |

- The models are not part of this repository. `voice setup` downloads them from Hugging Face, under their own licenses.
- They are pinned to the exact revisions this version was tested against, and every command reads those revisions. A later upload to the same Hugging Face repo changes nothing until a `voice` release says so.
- By default, mlx-audio picks the loader from the model's own files, and some model families run code that ships with the model. `voice` loads only these two families, whose loaders read weight files and nothing else.
- You can point `voice` at another build of either model with the variables under [Configuration](#configuration), such as the multilingual `mlx-community/parakeet-tdt-0.6b-v3`, which covers 25 languages.

Measured on an M1 with 16 GB: Parakeet loads in about 3 s and transcribes 8 seconds of speech in under half a second, peaking at 1.7 to 1.9 GB. The 16-bit copy peaked at 1.65 GB against 1.94 GB for the original on the same recording. Kokoro peaks at about 450 MB.

### Configuration

| Variable | Default | What it sets |
|---|---|---|
| `VOICE_STT_MODEL` | the 16-bit copy once `voice shrink` made it, else `mlx-community/parakeet-tdt-0.6b-v2` | the speech-to-text model: a Parakeet build, as a Hugging Face repo or a local folder |
| `VOICE_TTS_MODEL` (or `VOICE_MODEL`) | `mlx-community/Kokoro-82M-bf16` | the text-to-speech model: a Kokoro build, as a repo or a folder; the voices come from its `voices` folder |
| `VOICE_NAME` | `af_heart` | the speaking voice; `--voice` overrides it |
| `VOICE_SPEED` | `1.0` | the speaking speed, from 0.5 to 2.0; `--speed` overrides it |

A local folder is an absolute path or one starting with `~`. Any other name is a Hugging Face repo, never a folder in the current directory.

### Installing by hand

`install.sh` runs these two commands, then `voice setup`:

```bash
uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
uv tool install --reinstall-package voice --constraints requirements.txt .
```

The first lists the exact package versions from `uv.lock`, the ones the tests and the weekly vulnerability audit use. The second installs `voice` with those versions.

### Memory

Every command is its own short process: it loads one model, does its job and frees the model before it exits. There are two exceptions: the background speaker holds Kokoro from `voice speak --warm` until `--release`, and `voice chat` holds Parakeet while its loop is on.

Before loading a model, `voice` checks macOS's memory-pressure level. At critical, it refuses with a message instead of pushing the machine into swap. `--force` loads anyway.

### Privacy

- **Audio:** your speech is recorded into memory, transcribed and discarded.
- **Transcript:** the text goes to your clipboard, and to your cursor with `--paste`. In a terminal, `voice listen` also prints it. On the clipboard it is marked concealed, the [nspasteboard.org](http://nspasteboard.org) convention that tells clipboard-history apps to leave it out of their history. Notifications say how many words were heard, never what they were, because macOS keeps notification text.
- **Universal Clipboard:** when Handoff is on, macOS can share the clipboard with your other nearby Apple devices. Turn off Handoff in **System Settings > General > AirDrop & Handoff** to keep the text on this Mac.
- **Network:** after `voice setup`, `voice` makes no network requests. Every other command sets `HF_HUB_OFFLINE=1` before any model library is imported.
- **Voice chat:** what you say is typed into Claude Code as your message, so it goes wherever your Claude Code messages go. The summary Claude speaks is played locally and never stored.
- **Voice chat and other programs:** any program running as you can reach a session's socket and turn its loop on, the same as `voice chat --toggle`, and the start sound plays when it does. It can also send a made-up prompt or reply, so the loop stops listening or speaks text of its choosing, or turn the loop off. While the microphone is open, any sound it hears, including speech another program plays, is typed in as your message. A leading `/`, `!` or `#` is stripped first, so it arrives as plain text and can't run a slash command, a shell command or a memory write.
- **Files:** `voice` keeps its lock files, sockets and error logs in `~/.local/state/claude-voice/` (under `$XDG_STATE_HOME` when that is set), a folder only you can read. `voice shrink` writes its copies to `~/.local/share/voice/models/`, each in a folder only you can read. The logs never hold a transcript. `voice` trusts a pid file only while its process is alive and holds a lock on that file, so it never signals a process that has exited.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) to set up a clone and for what a change needs, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

The code is [MIT](LICENSE). The models are downloaded separately and keep their own licenses, listed under [Models](#models).
