# voice

Local text-to-speech and speech-to-text on Apple Silicon, through [mlx-audio](https://github.com/Blaizzy/mlx-audio). Nothing leaves the machine after the one-time model download.

Each command runs as its own short process: it loads one model, does one job and releases the model before it exits, so nothing stays in memory between calls. Before it loads a model, the command reads the kernel's memory-pressure level and refuses to load at critical (`--force` overrides).

## Install

```bash
uv tool install git+https://github.com/<owner>/voice   # or: uv tool install /path/to/a/clone
voice setup                                          # downloads the default models once
```

Python 3.12 and an Apple Silicon Mac are required.

## Commands

| Command | What it does |
|---|---|
| `voice speak "text"` | Reads the text aloud. Text on stdin works too. A new call stops the one already playing. |
| `voice speak --stop` | Stops whatever is playing. |
| `voice listen` | Records from the microphone until you pause, transcribes, prints the text and copies it to the clipboard. `--silence 1.5` sets the pause length in seconds, `--max-seconds 60` the longest recording, and `--no-copy` prints only. |
| `voice setup` | Downloads the default models into the Hugging Face cache. Every other command runs offline. |

## Models

| Job | Default | Size | License | Override |
|---|---|---|---|---|
| Speech-to-text | `mlx-community/parakeet-tdt-0.6b-v2`, English | about 2.5 GB | CC-BY-4.0 (NVIDIA) | `VOICE_STT_MODEL` |
| Text-to-speech | `mlx-community/Kokoro-82M-bf16` | about 390 MB | Apache-2.0 | `VOICE_MODEL` |

Any speech-to-text model that mlx-audio loads works in `VOICE_STT_MODEL`, such as `mlx-community/whisper-large-v3-turbo-q4` for a smaller download. `VOICE_NAME` (default `af_heart`) and `VOICE_SPEED` (default `1.0`) set the speaking voice.

## Dictation from a hotkey

In the Shortcuts app, make a shortcut with one "Run Shell Script" action running the full path to `voice listen` (see `which voice`), and give it a keyboard shortcut. Press it, speak, pause, and the text is on the clipboard.

## Claude Code

Claude Code can read each reply aloud through a Stop hook that runs a configured speaker command with the text as its last argument, or `--stop`. Point that command at `voice speak`:

```json
{"enabled": true, "command": ["/full/path/to/voice", "speak"]}
```

The hook speaks every reply by itself, so no skill or model call is involved.

## Tests

```bash
uv run python -m unittest discover -s tests
```

The tests use fakes for the models, the microphone and the speaker, so they need no download and no audio device.
