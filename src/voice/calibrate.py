"""`voice calibrate`: measure the room's noise and the user's voice through the current
microphone, rate each spot the user speaks from, and save a speech level for that microphone.

Optional. A microphone with no saved level uses the defaults in listen.py, and every recording
still measures the room's noise as it goes. Every take starts on the start sound and ends on
the stop sound, so the user always knows when the microphone is listening."""

from __future__ import annotations

import json
import math
import re
from typing import TYPE_CHECKING

from voice import listen, models, state
from voice.setup import ask, say

if TYPE_CHECKING:
    import numpy as np

SENTENCE = "Please read me a short summary of what changed today, then wait for my answer."
QUIET_SECONDS = 3.0
NOISE_MIN = 0.001  # RMS; a room reading below this is the microphone's own floor, not the room
GOOD, BORDERLINE = 10.0, 4.0  # voice-to-noise ratios: 20 dB and 12 dB
HEARD = 0.6  # share of the sentence's words a transcript must contain to count as heard right
CHANGING = 2.0  # noise whose loud moments pass this many times its quiet ones keeps changing
SPOTS = ("where you usually sit", "the other side of your screen", "across the room",
         "where you usually sit, facing away from the microphone")
# Short sounds that should not start the end-of-turn countdown, then short words that should.
NOISES = (("cough", "Cough."), ("throat", "Clear your throat."), ("sneeze", "Fake a sneeze."),
          ("chair", "Shift or roll your chair."), ("clap", "Clap."), ("tap", "Tap the desk."))
WORDS = (("hey", 'Say "hey".'), ("end", 'Say "end".'), ("what's up", 'Say "what\'s up?"'))
EVENT_SECONDS = 3.0
WINDOW = 0.02  # seconds per window when timing a sound
PITCH_WINDOW = 0.04  # seconds per window when checking a sound for pitch
PITCH_LOW, PITCH_HIGH = 70, 400  # Hz, the range of a speaking voice's pitch
EVENTS = state.STATE / "sound-events.json"  # the last measurement, numbers only, for tuning the rule
STEPS = 4
FIX = "move closer, raise the input level in System Settings > Sound > Input, or use a headset"


def windows(audio: np.ndarray, seconds: float) -> np.ndarray:
    n = int(listen.RATE * seconds)
    count = len(audio) // n
    return audio[:count * n].reshape(count, n)


def rms(blocks: np.ndarray) -> np.ndarray:
    import numpy as np

    return np.sqrt(np.mean(blocks ** 2, axis=1))


def frame_levels(audio: np.ndarray) -> list[float]:
    """The RMS of each FRAME-long block of `audio`, as record() measures it."""
    blocks = windows(audio, listen.FRAME)
    return [float(x) for x in rms(blocks)] if len(blocks) else []


def room_noise(levels: list[float]) -> float:
    """The room's steady noise: the median level of a quiet take."""
    import numpy as np

    return max(NOISE_MIN, float(np.median(levels))) if levels else NOISE_MIN


def voice_level(levels: list[float]) -> float:
    """How loud the user's speech reaches the microphone: the 90th percentile of a take,
    which skips the pauses before, between and after the words."""
    import numpy as np

    return float(np.percentile(levels, 90)) if levels else 0.0


def changing(levels: list[float]) -> bool:
    """True for noise that rises and falls (music, voices, a TV) rather than holding steady (a fan)."""
    import numpy as np

    if not levels:
        return False
    return float(np.percentile(levels, 90)) > CHANGING * max(NOISE_MIN, float(np.percentile(levels, 10)))


def decibels(level: float, noise: float) -> float:
    return 20 * math.log10(level / noise) if level > 0 else 0.0


def rating(voice: float, noise: float) -> str:
    ratio = voice / noise
    return "good" if ratio >= GOOD else "borderline" if ratio >= BORDERLINE else "too quiet"


def threshold(voice: float, noise: float) -> float:
    """Halfway between the room and the voice in decibels: the geometric mean."""
    return math.sqrt(voice * noise)


def heard_right(text: str) -> bool:
    words = set(re.findall(r"[a-z']+", SENTENCE.lower()))
    return len(words & set(re.findall(r"[a-z']+", text.lower()))) >= HEARD * len(words)


def describe_room(noise: float) -> str:
    if noise < 0.003:
        return "quiet"
    if noise < listen.SPEECH_LEVEL:
        return "some background noise"
    return "noisy"


def voicing(window: np.ndarray) -> float:
    """How strongly `window` repeats at a speaking voice's pitch, from 0 (noise) to 1 (a pure
    tone): the highest normalized autocorrelation over the pitch range. Vowels score high;
    coughs, sneezes, claps and creaks are mostly noise and score low."""
    import numpy as np

    window = window - window.mean()
    best = 0.0
    for lag in range(listen.RATE // PITCH_HIGH, listen.RATE // PITCH_LOW + 1):
        a, b = window[:-lag], window[lag:]
        energy = float(np.dot(a, a) * np.dot(b, b))
        if energy > 0:
            best = max(best, float(np.dot(a, b)) / math.sqrt(energy))
    return best


def event(audio: np.ndarray, threshold: float) -> dict[str, float]:
    """One sound's length in seconds (first to last window above `threshold`), its peak level,
    and its median voicing over the windows above `threshold`; zeros when nothing passed."""
    import numpy as np

    timed = windows(audio, WINDOW)
    levels = rms(timed) if len(timed) else np.zeros(0)
    loud = np.flatnonzero(levels > threshold)
    if len(loud) == 0:
        return {"length": 0.0, "peak": 0.0, "voicing": 0.0}
    pitched = windows(audio, PITCH_WINDOW)
    scores = [voicing(w) for w, level in zip(pitched, rms(pitched), strict=True) if level > threshold]
    return {"length": round(float(loud[-1] - loud[0] + 1) * WINDOW, 2),
            "peak": round(float(levels.max()), 4),
            "voicing": round(float(np.median(scores)) if scores else 0.0, 2)}


def take(seconds: float, silence: float | None = None, patience: float | None = None) -> np.ndarray:
    """One recording, from the start sound to the stop sound."""
    say("  Listening...")
    audio = listen.record(seconds, silence, on_open=lambda: listen.cue("start"), patience=patience)
    listen.cue("stop")
    return audio


def heading(number: int, title: str) -> None:
    say(f"\nStep {number} of {STEPS}: {title}")


def go(prompt: str, skip: str = "", finish: str = "") -> str:
    """Every prompt reads the same way: what to do, then Press Enter to start, then the other
    choices. Returns "start", "skip" or "finish"."""
    say(f"  {prompt} Press Enter to start.")
    choices = [f"{key} to {what}" for key, what in (("s", skip), ("q", finish)) if what]
    if choices:
        say(f"  Or type {', or '.join(choices)}.")
    reply = ask().strip().lower()
    if finish and reply.startswith("q"):
        return "finish"
    return "skip" if skip and reply.startswith("s") else "start"


def measure_noise() -> tuple[float, bool]:
    """Record QUIET_SECONDS of the room; returns its noise level and whether it keeps changing."""
    levels = frame_levels(take(QUIET_SECONDS))
    return room_noise(levels), changing(levels)


def spot(model, noise: float) -> tuple[float, str] | None:
    """Record the sentence and print one line on it; returns the voice level and its rating,
    or None when nothing rose above the room."""
    audio = take(30, 1.5, patience=10)
    voice = voice_level(frame_levels(audio))
    if voice <= noise:
        say("  Heard nothing above the room's noise.")
        return None
    result = rating(voice, noise)
    text = listen.transcribe_with(model, audio)
    heard = "heard every word" if heard_right(text) else f'heard "{text or "no words"}"'
    say(f"  {result.capitalize()}: {decibels(voice, noise):.0f} dB above the room, {heard}.")
    return voice, result


def sound_events(noise: float) -> None:
    """Record each short sound and short word, print their length, loudness and voicing side by
    side, say which measure tells them apart on this microphone, and save the numbers to EVENTS."""
    cutoff = max(noise, NOISE_MIN) * listen.NOISE_MARGIN
    measured: dict[str, dict[str, dict[str, float]]] = {"sounds": {}, "words": {}}
    for group, prompts in (("sounds", NOISES), ("words", WORDS)):
        for name, prompt in prompts:
            if go(f"{prompt} Do it once, right after the tick.", skip="skip it") == "skip":
                continue
            result = event(take(EVENT_SECONDS), cutoff)
            while not result["length"]:
                if go("Heard nothing above the room. Try again:", skip="skip it") == "skip":
                    break
                result = event(take(EVENT_SECONDS), cutoff)
            if result["length"]:
                measured[group][name] = result
    state.write(EVENTS, json.dumps({"noise": noise, **measured}, indent=2))
    sounds, words = measured["sounds"], measured["words"]
    if not sounds and not words:
        return
    say(f"\n  {'':12}{'length':>8}{'loudness':>11}{'voicing':>9}")
    for name, result in [*sounds.items(), *words.items()]:
        say(f"  {name:12}{result['length']:>6.2f} s{decibels(result['peak'], noise):>8.0f} dB"
            f"{result['voicing']:>9.2f}")
    if not sounds or not words:
        return
    apart = [label for label in ("length", "peak", "voicing")
             if max(r[label] for r in sounds.values()) < min(r[label] for r in words.values())
             or max(r[label] for r in words.values()) < min(r[label] for r in sounds.values())]
    names = {"length": "length", "peak": "loudness", "voicing": "voicing"}
    if apart:
        verb = "tells" if len(apart) == 1 else "tell"
        say(f"  On this microphone, {' and '.join(names[a] for a in apart)} {verb} the sounds from the words.")
    else:
        say("  No single measure tells the sounds from the words here.")


def run() -> int:
    import sys

    import sounddevice as sd

    if not sys.stdin.isatty():
        say("voice calibrate asks questions; run it in a terminal")
        return 2
    device = listen.input_name(sd)
    if device is None:  # nothing to save the level under, and most likely nothing to record from
        say("No microphone found. Pick one in System Settings > Sound > Input, then try again.")
        return 1
    volume = listen.output_volume()
    say(f"Calibrating {device}. It takes one to three minutes.")
    say("Each recording starts on a tick and ends on a pop.")

    heading(1, "Room noise")
    go("Sit where you usually do, and stay quiet from the tick until the pop.")
    noise, _ = measure_noise()
    say(f"  The room: {describe_room(noise)}.")

    results: list[tuple[str, str]] = []  # (spot, rating)
    voices: list[float] = []
    with models.loaded(models.load_stt, models.stt_model()) as model:
        heading(2, "Your voice")
        say("  At each spot, read this sentence after the tick:")
        say(f'    "{SENTENCE}"')
        say("  Skip spots where you won't talk to voice. Each one you keep makes it listen for a quieter voice.")
        for number, place in enumerate(SPOTS):
            label = f"Spot {number + 1} of {len(SPOTS)}: {place}."
            say()
            if number == 0:
                go(label)
            else:
                choice = go(f"{label} Go there first.", skip="skip it", finish="finish this step")
                if choice == "finish":
                    break
                if choice == "skip":
                    continue
            heard = spot(model, noise)
            results.append((place, heard[1] if heard else "too quiet"))
            if heard and heard[0] / noise >= BORDERLINE:
                voices.append(heard[0])

        heading(3, "Background noise (optional)")
        say("  Turn on a noise you often work with: music, a fan, a TV, or someone talking.")
        if go("Once it's on, stay quiet from the tick until the pop.", skip="skip this step") == "start":
            loud, varies = measure_noise()
            kind = ("rises and falls, so its loud moments can sound like speech; a headset microphone is "
                    "the reliable fix" if varies else "is steady, so voice measures it and sets it aside")
            say(f"  With it on, the room is {describe_room(loud)}. This noise {kind}.")
            go("Now read the sentence over the noise.")
            spot(model, loud)  # reported only: the saved level comes from the usual room
            say("  Turn the noise off now.")

    heading(4, "Short sounds (optional)")
    say("  From where you usually sit, you make a few noises, such as a cough, then say a few short")
    say("  words. It measures what tells them apart, so a cough can be kept from ending your turn later.")
    if go("Each one gets its own tick.", skip="skip this step") == "start":
        sound_events(noise)

    say("\nSummary")
    say(f"  Microphone: {device}. Room: {describe_room(noise)}.")
    for place, result in results:
        say(f"  {result.capitalize():11}{place}")
    if any(result != "good" for _, result in results):
        say(f"  To hear you better: {FIX}.")
    if not voices:
        say("  Not saved: no spot was clear enough of the room, so voice keeps its defaults.")
        return 1
    listen.save_calibration(device, threshold(min(voices), noise), volume)  # the quietest usable spot
    say(f"  Saved for {device}. Calibrate again after you move rooms or change microphones.")
    return 0
