<h1 align="center">voice</h1>

<p align="center">
  Fast, private speech for your Mac: dictate into anything, and have text read back to you, with every model running on your own machine.
</p>

<p align="center">
  <img alt="Python 3.12" src="https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white">
  <img alt="Apple Silicon" src="https://img.shields.io/badge/Apple%20Silicon-MLX-000000?logo=apple&logoColor=white">
  <img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-2EA44F">
</p>

<p align="center">
  <a href="#install"><b>Install</b></a> &nbsp;|&nbsp;
  <a href="#dictation"><b>Dictation</b></a> &nbsp;|&nbsp;
  <a href="#reading-aloud"><b>Reading aloud</b></a> &nbsp;|&nbsp;
  <a href="#voice-chat"><b>Voice chat</b></a> &nbsp;|&nbsp;
  <a href="#claude-code"><b>Claude Code</b></a> &nbsp;|&nbsp;
  <a href="#a-smaller-dictation-model"><b>Smaller model</b></a> &nbsp;|&nbsp;
  <a href="#reference"><b>Reference</b></a>
</p>

<br>

`voice` is a small command-line tool for Apple Silicon Macs. It does three things:

- `voice listen` turns what you say into text on your clipboard.
- `voice speak` reads text aloud.
- `voice chat` lets you talk with Claude Code hands-free.

Everything runs on your Mac through [mlx-audio](https://github.com/Blaizzy/mlx-audio), so no audio or text goes to a server.

- **Fast, accurate dictation.** On an M1, NVIDIA's Parakeet transcribes normal speech in about half a second once the model is loaded.
- **Natural voices.** Kokoro-82M has 54 voices. `voice` reads English, so the American and British voices fit best.
- **Hands-free while you work.** Leave `voice chat` running in a terminal and switch to your browser or any other app. What you say still goes to Claude, and a global shortcut turns the loop on and off from anywhere.
- **Light on memory.** Each command loads its model, does one job and frees it before it exits. A model stays loaded only while the background speaker or a voice chat is running.
- **Offline after setup.** `voice setup` downloads the models once, and every other command works without a network connection.

## Install

### What you need

- A Mac with Apple Silicon (M1 or newer).
- About 3 GB of free disk space for the models.
- [uv](https://docs.astral.sh/uv/), a Python package tool. Install it with `brew install uv` or [uv's installer](https://docs.astral.sh/uv/getting-started/installation/). uv downloads Python 3.12 for you if you don't have it.
- Git. If it's missing, macOS offers to install it the first time you run `git`.

### Steps

1. Download the code:
   ```bash
   git clone https://github.com/thefilesareinthecomputer/voice.git
   cd voice
   ```
2. Install the `voice` command:
   ```bash
   uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
   uv tool install --constraints requirements.txt .
   ```
   The first line lists the exact package versions from `uv.lock`, the ones the tests and the weekly vulnerability audit use. The second line installs `voice` with those versions.
3. Download the models, about 2.9 GB, once:
   ```bash
   voice setup
   ```

If your terminal says `command not found: voice`, run `uv tool update-shell`, open a new terminal window and try again.

### Check that it works

```bash
voice speak "Hello, it works."
voice listen
```

You should hear the sentence. Then `voice listen` waits for you to talk: say something, pause, and your words are printed. The first time, macOS asks whether your terminal may use the microphone. Allow it.

## Dictation

In a terminal, run `voice listen` and start talking. When you pause for a second and a half, the text is printed and copied to the clipboard.

### From a keyboard shortcut

Press a keyboard shortcut to start recording, talk as long as you like, and press it again to stop. The text goes to your clipboard, ready to paste.

1. Open the **Shortcuts** app and turn on **Settings > Advanced > Allow Running Scripts**. This lets every shortcut run shell commands, including ones you import later, so import only shortcuts you trust.
2. Create a shortcut with one **Run Shell Script** action, and set its script to:
   ```bash
   ~/.local/bin/voice listen --toggle
   ```
   Use the full path, because Shortcuts can't find `voice` by name. `which voice` shows the path on your Mac.
3. Open the shortcut's details, choose **Add Keyboard Shortcut**, and press a combination with a modifier, such as Control+Option+Command+D.
4. Press the keyboard shortcut to start. The first time, macOS asks for microphone access. A sound, a notification and the orange microphone dot in the menu bar tell you it's recording.
5. Press it again to stop. A notification says how many words it heard, and Command+V pastes the text.

A recording stops on its own after 30 minutes. The model loads only after you stop, so the text takes a few seconds to arrive. Add `--preload` to load the model while you talk instead: the text arrives sooner, but about 1.7 GB of memory stays in use for the whole recording. Several takes in a row are transcribed in order, so a quick second take never overwrites a newer result.

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

- A new `speak` stops the one already playing, so you always hear the latest text.
- Long text starts quickly, because each sentence is made while the one before it plays.
- On its own, `speak` takes about 4 seconds to start talking. `voice speak --warm` starts a background speaker that keeps the model loaded, about 450 MB, and then every `speak` starts in about half a second. `voice speak --release` stops it and frees the memory.
- `--speed` goes from 0.5 to 2.0.
- `voice voices` lists the 54 voices. In a name, the first letter is the accent (`a` American, `b` British) and the second is `f` female or `m` male. The default is `af_heart`, an American female voice.

## Voice chat

`voice chat` starts Claude Code in your terminal with a spoken loop. You talk, a three-second pause ends your turn, and your words are typed into Claude's prompt and sent. When Claude finishes, it reads you a short summary, and then the microphone opens for your reply.

```bash
voice chat                  # start a new conversation
voice chat --continue       # pick up the last conversation in this folder
voice chat --silence 5      # wait for five seconds of quiet before ending your turn (0.2 to 30)
```

Any option `voice` doesn't know goes to `claude`, such as `--model` or `--resume`.

- **You can work in other apps meanwhile.** Your words are typed into Claude's own terminal, not the window in front, so the terminal can sit in the background. Closing it ends the session. Approving a permission request still needs that terminal.
- **Typing always takes over.** Any key you press while it talks or listens stops that, and the key still goes to Claude. Moving or clicking the mouse does not. After Escape or Control+C stops Claude, or after a command such as `/clear`, the microphone opens again on its own.
- **Control+Q turns the loop off and on.** So does saying "voice chat off" on its own. After 30 seconds with no words, it turns off by itself, with a sound and a notification. A click, a cough or other noise without words doesn't end your turn.
- **One voice chat listens at a time.** Starting a voice chat, or turning one on, switches any other voice chat to plain typing, with a notification.
- **Sounds mark each step.** A tick means the microphone is open. A pop means it closed and your words are on their way.
- **The microphone never hears `voice` itself.** It opens only after the speaking and the on sound end, and it ignores what it hears while its tick plays, so its own sounds never count as you talking.
- **A global shortcut works from any app.** This is a second shortcut, separate from dictation's. Bind `~/.local/bin/voice chat --toggle` to a key such as Control+Option+Command+C, the same way as [the dictation shortcut](#from-a-keyboard-shortcut). It turns the loop of the newest running session off or on. A terminal never passes Command shortcuts to the program inside it, so this is the only way to use one.

Claude Code needs no setup for this. `voice chat` adds its hooks and its summary instruction to this one session through command-line flags, and nothing in `~/.claude` changes. Your terminal app needs microphone access, which macOS asks for the first time.

While the loop is on, Parakeet holds about 1.7 GB and the background speaker about 450 MB more. Turning the loop off frees Parakeet, and quitting Claude stops the speaker if `voice chat` started it.

## Claude Code

This makes ordinary Claude Code sessions read a short summary of each reply aloud. `voice chat` handles its own speech, so it doesn't need any of this.

While it's on, Claude ends every reply with a one-to-three sentence summary of what it did and what it needs from you, and only that summary is spoken. A reply without one stays silent, sending your next message stops the playback, and headless `claude -p` runs stay silent.

The hook passes the summary to `voice speak` and returns at once. It reads its settings only from `~/.claude/voice.json`, and it needs only Python's standard library, so the `python3` that ships with macOS runs it. The `-I` flag keeps your environment variables and user packages from loading into it.

1. Copy the hook into place:
   ```bash
   mkdir -p ~/.claude/hooks && cp -i integrations/claude-code/voice_reply.py ~/.claude/hooks/
   ```
2. Add it to the `hooks` key of `~/.claude/settings.json`, beside any hooks you already have:
   ```json
   {
     "hooks": {
       "UserPromptSubmit": [{"hooks": [{"type": "command", "command": "python3 -I ~/.claude/hooks/voice_reply.py", "timeout": 5}]}],
       "Stop": [{"hooks": [{"type": "command", "command": "python3 -I ~/.claude/hooks/voice_reply.py", "timeout": 5}]}]
     }
   }
   ```
3. Create `~/.claude/voice.json`, naming `voice speak` and any voice you like:
   ```json
   {"enabled": true, "command": ["~/.local/bin/voice", "speak", "--voice", "af_heart"]}
   ```
4. Optionally, add the `/voice` command, which takes `on`, `off` and `status`. `on` says "Voice is on." aloud, so you know it works before the next reply:
   ```bash
   mkdir -p ~/.claude/commands && cp -i integrations/claude-code/voice.md ~/.claude/commands/
   ```

Each summary takes about 4 seconds to start unless the background speaker is running. Run `voice speak --warm` while voice is on, and each summary starts in about half a second. `voice speak --release` frees that memory again.

## A smaller dictation model

`voice shrink` writes a 16-bit copy of Parakeet, whose published weights are 32-bit. The copy is half the size (1.24 GB instead of 2.5 GB), uses about 0.3 GB less memory while it transcribes, and gave the same transcript in testing.

```bash
voice shrink
```

The conversion needs about 3.5 GB of free memory for a minute. After that, every command, the keyboard shortcut included, uses the copy automatically. If you chose your own model with `VOICE_STT_MODEL`, the command instead prints the `export` line that selects its copy.

## Troubleshooting

- **`command not found: voice`.** Run `uv tool update-shell` and open a new terminal window.
- **The shortcut does nothing, or you hear the error sound.** A notification says why: "not started" names critical memory pressure or a missing model, and "model missing" means `voice setup` hasn't run yet.
- **"No sound reached the microphone."** The app that runs `voice`, Shortcuts or your terminal, has no microphone access. Turn it on in **System Settings > Privacy & Security > Microphone**.
- **"Waiting for memory."** Memory pressure was critical when you stopped. Your recording is kept and transcribes once the pressure eases, within a minute.
- **"Failed."** The error is in `~/.local/state/claude-voice/listen.log`, which never holds a transcript.

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
- **Files:** `voice` keeps its lock files, sockets and error logs in `~/.local/state/claude-voice/` (under `$XDG_STATE_HOME` when that is set), a folder only you can read. `voice shrink` writes its copies to `~/.local/share/voice/models/`, each in a folder only you can read. The log never holds a transcript. `voice` trusts a pid file only while its process is alive and holds a lock on that file, so it never signals a process that has exited.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) to set up a clone and for what a change needs, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

The code is [MIT](LICENSE). The models are downloaded separately and keep their own licenses, listed under [Models](#models).
