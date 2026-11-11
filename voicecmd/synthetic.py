"""Formant-based synthetic speech for demos, tests and reproducible evaluation.

Each "word" is a sequence of phoneme-like tokens (vowels, nasal, fricative, plosive)
synthesised with a glottal pulse train filtered by cascaded formant resonators.
Speakers differ in pitch, vocal-tract scale and speaking rate. This is *not* natural
speech: it exists so that the whole pipeline can be exercised and measured without
needing a recorded corpus. Real recordings can be enrolled through the app or CLI.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

SR = 16000

VOWELS = {  # F1, F2, F3 in Hz (adult male-ish reference values)
    "a": (700, 1220, 2600),
    "e": (530, 1840, 2480),
    "i": (270, 2290, 3010),
    "o": (570, 840, 2410),
    "u": (300, 870, 2240),
}
NASAL = (250, 1000, 2200)
BANDWIDTHS = (80, 100, 150)

# token durations in seconds at tempo 1.0
DURATIONS = {"vowel": 0.15, "n": 0.09, "s": 0.14, "p": 0.07}

WORDS = {
    "go": "uo",
    "stop": "spop",
    "left": "nesp",
    "right": "saip",
    "up": "aup",
    "down": "paun",
}


def speaker_params(speaker_id: int) -> dict:
    """Deterministic pseudo-speaker: pitch, vocal-tract scale and tempo."""
    rng = np.random.default_rng(1000 + int(speaker_id))
    f0 = float(rng.uniform(95, 230))
    scale = float(np.clip(0.88 + (f0 - 95) / (230 - 95) * 0.22 + rng.normal(0, 0.02), 0.85, 1.2))
    tempo = float(rng.uniform(0.85, 1.2))
    return {"f0": f0, "scale": scale, "tempo": tempo}


def _resonate(x: np.ndarray, freq: float, bw: float) -> np.ndarray:
    r = np.exp(-np.pi * bw / SR)
    theta = 2 * np.pi * freq / SR
    a = [1.0, -2 * r * np.cos(theta), r * r]
    b = [1 - 2 * r * np.cos(theta) + r * r]
    return lfilter(b, a, x)


def _glottal(n: int, f0: float, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / SR
    f = f0 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t + rng.uniform(0, 2 * np.pi)))
    phase = np.cumsum(f) / SR
    pulses = (np.diff(np.floor(phase), prepend=np.floor(phase[0])) > 0).astype(float)
    return lfilter([1.0], [1.0, -0.95], pulses)


def _fade(x: np.ndarray, ms: float = 8.0) -> np.ndarray:
    k = min(int(SR * ms / 1000), len(x) // 2)
    if k > 0:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(k) / k)
        x = x.copy()
        x[:k] *= ramp
        x[-k:] *= ramp[::-1]
    return x


def _rms_normalise(x: np.ndarray, level: float) -> np.ndarray:
    r = np.sqrt(np.mean(x ** 2)) + 1e-12
    return x * (level / r)


def _token(tok: str, dur: float, spk: dict, rng: np.random.Generator) -> np.ndarray:
    n = max(int(dur * SR), 32)
    if tok in VOWELS or tok == "n":
        formants = VOWELS.get(tok, NASAL)
        sig = _glottal(n, spk["f0"], rng)
        for f, bw in zip(formants, BANDWIDTHS):
            f_s = min(f * spk["scale"] * rng.uniform(0.97, 1.03), 0.45 * SR)
            sig = _resonate(sig, f_s, bw)
        return _fade(_rms_normalise(sig, 0.30 if tok != "n" else 0.12))
    if tok == "s":
        b, a = butter(4, 4000 / (SR / 2), btype="high")
        return _fade(_rms_normalise(lfilter(b, a, rng.standard_normal(n)), 0.15))
    if tok == "p":
        closure = np.zeros(int(0.75 * n))
        burst_n = n - len(closure)
        burst = rng.standard_normal(burst_n) * np.exp(-np.arange(burst_n) / (0.25 * burst_n + 1))
        return np.concatenate([closure, _rms_normalise(burst, 0.30)])
    raise ValueError(f"Unknown token: {tok}")


def synth_word(word: str, speaker_id: int = 0, seed: int = 0, snr_db: float | None = None) -> np.ndarray:
    """Render one utterance of ``word`` (float32, 16 kHz, with leading/trailing silence)."""
    if word not in WORDS:
        raise KeyError(f"Unknown synthetic word: {word!r}. Choose from {sorted(WORDS)}.")
    spk = speaker_params(speaker_id)
    rng = np.random.default_rng(seed)
    parts = []
    for tok in WORDS[word]:
        base = DURATIONS["vowel"] if tok in VOWELS else DURATIONS[tok]
        dur = base / spk["tempo"] * rng.uniform(0.85, 1.15)
        parts.append(_token(tok, dur, spk, rng))
    speech = np.concatenate(parts)
    speech_power = float(np.mean(speech ** 2))

    lead = int(rng.uniform(0.15, 0.40) * SR)
    trail = int(rng.uniform(0.15, 0.40) * SR)
    y = np.concatenate([np.zeros(lead), speech, np.zeros(trail)])
    noise_power = 1e-6 if snr_db is None else speech_power / (10 ** (snr_db / 10.0))
    y = y + rng.standard_normal(len(y)) * np.sqrt(noise_power)
    y = y / (np.max(np.abs(y)) + 1e-12) * 0.8
    return y.astype(np.float32)


def make_dataset(words=None, speakers=range(5), reps: int = 2, snr_db: float | None = None, seed: int = 0):
    """Return ``[(word, samples, SR), ...]`` for every word x speaker x repetition."""
    words = list(words) if words is not None else sorted(WORDS)
    data = []
    for wi, w in enumerate(words):
        for s in speakers:
            for r in range(reps):
                utt_seed = seed * 1_000_003 + (wi + 1) * 10_007 + int(s) * 101 + r
                data.append((w, synth_word(w, s, utt_seed, snr_db), SR))
    return data


def write_wav(path: str, y: np.ndarray, sr: int = SR) -> None:
    """Write float samples in [-1, 1] as a 16-bit PCM WAV file."""
    from scipy.io import wavfile
    wavfile.write(path, sr, (np.clip(y, -1, 1) * 32767).astype(np.int16))


def write_demo_dataset(root: str) -> dict:
    """Create ``enroll/``, ``test/`` and ``test_noisy/`` folders of WAV clips under ``root``."""
    import os
    plan = {
        "enroll": dict(speakers=range(0, 3), reps=2, snr_db=None, seed=11),
        "test": dict(speakers=range(20, 22), reps=2, snr_db=None, seed=12),
        "test_noisy": dict(speakers=range(22, 24), reps=1, snr_db=10, seed=13),
    }
    counts = {}
    for split, kw in plan.items():
        n = 0
        for word in sorted(WORDS):
            os.makedirs(os.path.join(root, split, word), exist_ok=True)
        for wi, word in enumerate(sorted(WORDS)):
            for s in kw["speakers"]:
                for r in range(kw["reps"]):
                    seed = kw["seed"] * 1_000_003 + (wi + 1) * 10_007 + s * 101 + r
                    y = synth_word(word, s, seed, kw["snr_db"])
                    write_wav(os.path.join(root, split, word, f"spk{s:02d}_rep{r}.wav"), y)
                    n += 1
        counts[split] = n
    return counts
