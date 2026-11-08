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
  <a href="#claude-code"><b>Claude Code</b></a> &nbsp;|&nbsp;
  <a href="#a-smaller-dictation-model"><b>Smaller model</b></a> &nbsp;|&nbsp;
  <a href="#reference"><b>Reference</b></a>
</p>

<br>

`voice` is a small command-line tool for Apple Silicon Macs. `voice listen` turns what you say into text on your clipboard, and `voice speak` reads text aloud. Both run open models through [mlx-audio](https://github.com/Blaizzy/mlx-audio) on your own machine, so no audio or text goes to a server.

- **Accurate dictation.** On an M1, NVIDIA's Parakeet transcribes normal-pace speech in about half a second once the model is loaded.
- **Natural voices.** Kokoro-82M ships 54 voices. `voice` reads English text, so its American and British voices fit best.
- **Light on memory.** Each command loads its model, does one job and releases it before it exits. Nothing stays resident.
- **Offline after setup.** One `voice setup` downloads the models, and every other command runs without a network connection.

## Install

You need an Apple Silicon Mac, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/thefilesareinthecomputer/voice.git && cd voice
uv export --frozen --no-dev --no-emit-project --output-file requirements.txt
uv tool install --constraints requirements.txt .
voice setup
```

The export pins the install to the versions in `uv.lock`, the ones the tests and the weekly vulnerability audit run against. `voice setup` downloads about 2.9 GB of models into the Hugging Face cache, once.

## Dictation

In a terminal, run `voice listen` and start talking. When you pause for a second and a half, the text is printed and copied to the clipboard.

### From a keyboard shortcut

The keyboard shortcut toggles a background recording: you press it to start, talk for as long as you like, and press it again to stop. The text then goes to your clipboard, ready to paste.

1. Open the **Shortcuts** app and turn on **Settings > Advanced > Allow Running Scripts**. This lets every shortcut run shell commands, including ones you import later, so import only shortcuts you trust.
2. Create a shortcut with one **Run Shell Script** action and set its script to the command below. It uses the full path of `voice` because Shortcuts does not read your shell's `PATH`. This is where uv installs it, and `which voice` confirms it:
   ```bash
   ~/.local/bin/voice listen --toggle
   ```
3. Open the shortcut's details, choose **Add Keyboard Shortcut**, and press a combination with a modifier, such as Control-Option-Command-D.
4. Press the keyboard shortcut once to start. macOS asks for microphone access the first time. You hear a sound and see a notification, and the orange microphone dot stays in the menu bar while it records.
5. Press it again to stop. A notification says how many words it heard, and Command-V pastes the text.

A recording stops on its own after 30 minutes. The model loads only after you stop, so nothing heavy sits in memory while you talk, and the text takes a few seconds to arrive after you stop. Add `--preload` for a shorter wait: the model loads while you talk and holds about 1.7 GB for the whole recording. Take after take, each one transcribes in turn, so a quick second take never overwrites a newer result.

<details>
<summary><b>Pasting automatically with <code>--paste</code>, and what it asks for</b></summary>

<br>

`voice listen --toggle --paste` also presses Command-V at your cursor when the recording stops, through System Events. macOS credits that request to whichever app is in front when you stop, so each app you dictate into asks once for permission to control System Events.

Allowing it gives that app, and any plugins or scripts it runs, lasting control of System Events, which can type into and click through every other app. If you decline, the text stays on the clipboard. When the paste does go through, it goes into whatever app is in front at that moment. For these reasons `--paste` is off unless you add it.

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

A new `speak` stops the one already playing, so you always hear the latest text, and each sentence is made while the one before it plays. `--speed` goes from 0.5 to 2.0. `voice voices` lists the 54 voices. The first letter of a name is the accent (`a` American, `b` British) and the second is the voice (`f` female, `m` male), so `af_heart`, the default, is an American female voice and `bm_george` a British male one.

## Claude Code

[`integrations/claude-code/`](integrations/claude-code) makes Claude Code read a short summary of each reply aloud. While it is on, Claude ends every reply with a one-to-three sentence summary of what it did and what it needs from you, and only that summary is spoken; a reply that ends without one stays silent. Sending your next message stops the playback, and headless `claude -p` runs stay silent.

The hook sends the summary to `voice speak` on stdin and returns at once. It reads its settings from `~/.claude/voice.json` only, and it needs only the `python3` that ships with macOS, run with `-I` so nothing outside the standard library can load into it.

1. Copy the hook into place:
   ```bash
   mkdir -p ~/.claude/hooks && cp integrations/claude-code/voice_reply.py ~/.claude/hooks/
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
4. Optionally, add the `/voice` command, which takes `on`, `off` and `status`; `on` says "Voice is on." aloud, so you know it works before the next reply:
   ```bash
   mkdir -p ~/.claude/commands && cp integrations/claude-code/voice.md ~/.claude/commands/
   ```

## A smaller dictation model

`voice shrink` writes a 16-bit copy of Parakeet, whose published weights are 32-bit. The copy is half the size (1.24 GB instead of 2.5 GB), needs about 0.3 GB less memory while it transcribes, and gave the same transcript in testing.

```bash
voice shrink
```

The conversion needs about 3.5 GB of free memory for a minute. From then on every command, the keyboard shortcut included, uses the copy of the default model on its own. For a model you chose with `VOICE_STT_MODEL`, the command prints the `export` line that selects its copy instead.

## Troubleshooting

- **The shortcut does nothing, or you hear the error sound.** A notification says why: "not started" names critical memory pressure or a missing model, and "model missing" means `voice setup` has not run yet.
- **"No sound reached the microphone."** The app that runs `voice`, Shortcuts or your terminal, has no microphone access. Turn it on in **System Settings > Privacy & Security > Microphone**.
- **"Waiting for memory."** Memory pressure was critical when you stopped. Your recording is kept and transcribes once the pressure eases, within a minute.
- **"Failed."** The error is in `~/.local/state/claude-voice/listen.log`, which never holds a transcript.

## Reference

### Models

| Job | Default model | Download | License |
|---|---|---|---|
| Speech-to-text | [`mlx-community/parakeet-tdt-0.6b-v2`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2), English | about 2.5 GB | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), by NVIDIA |
| Text-to-speech | [`mlx-community/Kokoro-82M-bf16`](https://huggingface.co/mlx-community/Kokoro-82M-bf16) | about 390 MB | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M), by hexgrad |

The models are not part of this repository; `voice setup` downloads them from Hugging Face under their own licenses, at the exact revisions this version was tested against, and every command reads those revisions, so a later upload to the same repo changes nothing until a release says so. By default, mlx-audio picks the loader from the model's own files, and some model families run code that ships with the model. `voice` loads only these two families, whose loaders read weight files and nothing else. You can still point `voice` at another build of either, such as the multilingual `mlx-community/parakeet-tdt-0.6b-v3`, which covers 25 languages.

Measured on an M1 with 16 GB: Parakeet loads in about 3 s and transcribes 8 seconds of speech in under half a second, peaking at 1.7 to 1.9 GB. The 16-bit copy peaked at 1.65 GB against 1.94 GB for the original on the same recording. Kokoro peaks at about 450 MB.

### Configuration

| Variable | Default | What it sets |
|---|---|---|
| `VOICE_STT_MODEL` | the 16-bit copy once `voice shrink` made it, else `mlx-community/parakeet-tdt-0.6b-v2` | the speech-to-text model: a Parakeet build, as a Hugging Face repo or a local folder |
| `VOICE_TTS_MODEL` (or `VOICE_MODEL`) | `mlx-community/Kokoro-82M-bf16` | the text-to-speech model: a Kokoro build, as a repo or a folder; the voices come from its `voices` folder |
| `VOICE_NAME` | `af_heart` | the speaking voice; `--voice` overrides it |
| `VOICE_SPEED` | `1.0` | the speaking speed, from 0.5 to 2.0; `--speed` overrides it |

A local folder is an absolute path or one starting with `~`; any other name is a Hugging Face repo, never a folder in the current directory.

### Memory

Every command is its own short process: it loads one model, does its job and frees the model before it exits. Before a load it reads macOS's memory-pressure level, and at critical it refuses with a message instead of tipping the machine into swap. `--force` loads anyway.

### Privacy

- **Audio:** your speech is recorded into memory, transcribed and discarded.
- **Transcript:** the text goes to your clipboard, and to your cursor with `--paste`; in a terminal, `voice listen` also prints it. On the clipboard it is marked concealed, the [nspasteboard.org](http://nspasteboard.org) convention that tells clipboard-history apps to leave an item out of their history. Notifications say how many words were heard, never what they were, because macOS keeps notification text.
- **Universal Clipboard:** when Handoff is on, macOS can make the clipboard available to your other nearby Apple devices. Turn off Handoff in **System Settings > General > AirDrop & Handoff** to keep the text on this Mac.
- **Network:** after `voice setup`, `voice` makes no network requests. Every other command sets `HF_HUB_OFFLINE=1` before any model library is imported.
- **Files:** `voice` keeps its lock files and an error log in `~/.local/state/claude-voice/` (under `$XDG_STATE_HOME` when that is set), a folder only you can read, and `voice shrink` writes its copies to `~/.local/share/voice/models/`, each copy in a folder only you can read. The log never holds a transcript. `voice` trusts a pid file only while its process is alive and holds a lock on that file, so it never signals a process that has exited.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) to set up a clone and for what a change needs, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

The code is [MIT](LICENSE). The models are downloaded separately and keep their own licenses, listed under [Models](#models).
