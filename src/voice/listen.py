"""`voice listen`: record from the microphone until you stop talking, transcribe on
device, print the text and copy it to the clipboard."""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from voice import models

RATE = 16_000          # what every speech-to-text model here expects
FRAME = 0.1            # seconds per level reading
SPEECH_LEVEL = 0.01    # RMS above which a frame counts as speech


def end_of_speech(levels: list[float], silence: float, threshold: float = SPEECH_LEVEL) -> bool:
    """True once speech has started and the last `silence` seconds were all quiet."""
    quiet = int(round(silence / FRAME))
    if quiet < 1 or len(levels) <= quiet or not any(lvl > threshold for lvl in levels):
        return False
    return all(lvl <= threshold for lvl in levels[-quiet:])


def record(max_seconds: float, silence: float) -> np.ndarray:
    """Record mono 16 kHz audio until `silence` seconds of quiet after speech, or `max_seconds`."""
    import sounddevice as sd

    frames: list[np.ndarray] = []
    levels: list[float] = []
    block = int(RATE * FRAME)
    print("listening... (stops after a pause)", file=sys.stderr)
    with sd.InputStream(samplerate=RATE, channels=1, dtype="float32", blocksize=block) as stream:
        while len(frames) * FRAME < max_seconds:
            chunk, _ = stream.read(block)
            frames.append(chunk[:, 0].copy())
            levels.append(float(np.sqrt(np.mean(chunk ** 2))))
            if end_of_speech(levels, silence):
                break
    return np.concatenate(frames) if frames else np.zeros(0, dtype="float32")


def transcribe_with(model, audio: np.ndarray) -> str:
    import mlx.core as mx

    if audio.size == 0:
        return ""
    result = model.generate(mx.array(audio))
    return str(getattr(result, "text", "")).strip()


def transcribe(audio: np.ndarray, force: bool = False) -> str:
    """Load the model, transcribe one recording, release the model."""
    from mlx_audio.stt.utils import load_model

    if audio.size == 0:
        return ""
    with models.loaded(load_model, models.stt_model(), force) as model:
        return transcribe_with(model, audio)


def copy(text: str) -> None:
    subprocess.run(["pbcopy"], input=text, text=True, check=False)


def _load(repo: str):
    from mlx_audio.stt.utils import load_model

    return load_model(repo)


def run(max_seconds: float = 60.0, silence: float = 1.5, clipboard: bool = True, force: bool = False) -> int:
    # The model loads while the speaker talks, so the wait after the pause is only the transcription.
    # The pressure check runs first, so a refusal costs no speech.
    with models.loading(_load, models.stt_model(), force) as pending:
        audio = record(max_seconds, silence)
        text = transcribe_with(pending.result(), audio)
    if text:
        print(text)
        if clipboard:
            copy(text)
    return 0
