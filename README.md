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
  <a href="#models"><b>Models</b></a> &nbsp;|&nbsp;
  <a href="#memory"><b>Memory</b></a>
</p>

<br>

`voice` is a small command-line tool for Apple Silicon Macs. `voice listen` turns what you say into text on your clipboard, and `voice speak` reads text aloud. Both run current open models through [mlx-audio](https://github.com/Blaizzy/mlx-audio), so nothing you say or type leaves the machine.

- **Accurate dictation.** NVIDIA's Parakeet transcribes normal-pace speech accurately and fast, about half a second after you stop talking on an M1.
- **Natural voices.** Kokoro-82M ships 54 voices across American and British English and several other languages.
- **Light on memory.** Each command loads its model, does one job and releases it before it exits. Nothing stays resident in the background.
- **Offline after setup.** One `voice setup` downloads the models, and every other command runs without a network connection.

## Install

You need an Apple Silicon Mac, Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv tool install git+https://github.com/thefilesareinthecomputer/voice
voice setup
```

`voice setup` downloads about 2.9 GB of models into the Hugging Face cache (`~/.cache/huggingface`), once. The first `voice listen` asks for microphone permission for the app that runs it: your terminal, or Shortcuts.

## Dictation

```bash
voice listen
```

Start talking. When you pause for a second and a half, `voice` transcribes what you said, prints it and copies it to the clipboard, ready to paste.

| Option | Default | What it does |
|---|---|---|
| `--silence 1.5` | 1.5 s | how long a pause ends the recording |
| `--max-seconds 60` | 60 s | the longest recording |
| `--no-copy` | off | print only, leave the clipboard alone |

### From a keyboard shortcut

Dictation is most useful from anywhere, without a terminal. In the **Shortcuts** app, create a shortcut with one **Run Shell Script** action that runs the full path to `voice listen` (find it with `which voice`), then give the shortcut a keyboard shortcut in its details. Press it, speak, pause, and paste.

## Reading aloud

```bash
voice speak "The build passed and the release is ready."
echo "Text from another program" | voice speak
voice speak --voice bf_emma --speed 1.1 "A different voice, a little faster."
voice speak --stop
```

A new `speak` stops the one already playing, so the latest text always wins. `voice voices` lists every installed voice: the first letter of a name is the accent (`a` American, `b` British) and the second is the voice (`f` female, `m` male), so `af_heart` is the default American female voice and `bm_george` a British male one.

## Claude Code

Claude Code can read each of its replies aloud through a Stop hook that runs a speaker command with the reply as its last argument, or with `--stop`. Point that command at `voice speak`, with any voice you like:

```json
{"enabled": true, "command": ["/full/path/to/voice", "speak", "--voice", "bf_emma"]}
```

The hook speaks every reply on its own, so no extra skill or model call is involved. Dictating into Claude Code is the same as anywhere else: `voice listen` from a keyboard shortcut, then paste.

## Models

| Job | Default model | Download | License |
|---|---|---|---|
| Speech-to-text | [`mlx-community/parakeet-tdt-0.6b-v2`](https://huggingface.co/mlx-community/parakeet-tdt-0.6b-v2), English | about 2.5 GB | [CC-BY-4.0](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), by NVIDIA |
| Text-to-speech | [`mlx-community/Kokoro-82M-bf16`](https://huggingface.co/mlx-community/Kokoro-82M-bf16) | about 390 MB | [Apache-2.0](https://huggingface.co/hexgrad/Kokoro-82M), by hexgrad |

The models are not part of this repository; `voice setup` downloads them from Hugging Face under their own licenses. Any speech-to-text model that mlx-audio can load works in place of Parakeet. For example, `VOICE_STT_MODEL=mlx-community/whisper-large-v3-turbo-q4` is a 464 MB download with somewhat lower accuracy, and the multilingual `mlx-community/parakeet-tdt-0.6b-v3` covers 25 languages.

Measured on an M1 with 16 GB: Parakeet loads in about 3 s and transcribes 8 seconds of speech in under half a second, with a peak of about 1.7 GB while it runs. `listen` starts loading the model the moment you start speaking, so the load is hidden behind your own speech. Kokoro peaks at about 450 MB.

## Memory

Every command is its own short process: it loads one model, does its job and frees the model before it exits, so `voice` holds no memory between calls. Before it loads a model it reads macOS's memory-pressure level, and at critical it refuses with a message instead of tipping the machine into swap. `--force` loads anyway.

## Configuration

| Variable | Default | What it sets |
|---|---|---|
| `VOICE_STT_MODEL` | `mlx-community/parakeet-tdt-0.6b-v2` | the speech-to-text model |
| `VOICE_MODEL` | `mlx-community/Kokoro-82M-bf16` | the text-to-speech model |
| `VOICE_NAME` | `af_heart` | the speaking voice; `--voice` overrides it |
| `VOICE_SPEED` | `1.0` | the speaking speed; `--speed` overrides it |

## Privacy

Audio is recorded into memory, transcribed and discarded; nothing is written to disk except the transcript on your clipboard. After `voice setup`, `voice` makes no network requests: every other command sets `HF_HUB_OFFLINE=1` before any model library is imported. The one file it keeps is a small pid file in `~/.local/state/claude-voice/`, used to stop a speaker that is still talking.

## Development

```bash
git clone https://github.com/thefilesareinthecomputer/voice.git && cd voice
uv sync
uv run python -m unittest discover -s tests
uv run ruff check
```

The tests fake the models, the microphone and the speaker, so they need no download and no audio device. See [CONTRIBUTING.md](CONTRIBUTING.md) for what a change needs, and [SECURITY.md](SECURITY.md) to report a vulnerability.

## License

The code is [MIT](LICENSE). The models are downloaded separately and keep their own licenses, listed under [Models](#models).
