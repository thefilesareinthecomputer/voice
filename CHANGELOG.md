# Changelog

All notable changes to this project are listed here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## Unreleased

### Added

- `voice listen --toggle` for a keyboard shortcut: the first press starts a background recording with a sound and a notification, the next press stops it, and the text is transcribed and copied. No silence cutoff; a recording stops on its own after 30 minutes.
- `--paste` also pastes the text at the cursor when a toggle recording stops, and keeps it on the clipboard.
- Recordings longer than a minute are transcribed in chunks where the model supports it.

## 0.1.0 - 2026-11-06

### Added

- `voice speak` reads text aloud with Kokoro-82M, from arguments or stdin; a new call stops the one already playing, and `--stop` stops it outright.
- `--voice` and `--speed` pick the speaking voice and speed for a call, and `voice voices` lists the voices in the local cache. The accent follows the voice name.
- `voice listen` records until a pause, transcribes with Parakeet TDT 0.6B v2, prints the text and copies it to the clipboard. The model loads while you speak.
- `voice setup` downloads the default models once; every other command runs offline.
- Each command loads one model in its own process and releases it before exiting, and refuses to load under critical memory pressure unless `--force` is given.
