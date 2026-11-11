"""Audio loading, validation and pre-processing (WAV only, no external codecs)."""

from __future__ import annotations

import io
import math
import os
from typing import BinaryIO, Union

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

TARGET_SR = 16000
MIN_DURATION_S = 0.10
MAX_DURATION_S = 10.0
SILENCE_RMS = 1e-4

Source = Union[str, os.PathLike, bytes, bytearray, BinaryIO]


class AudioError(ValueError):
    """Raised when an audio input is missing, unreadable or unusable."""


class NoSpeechError(AudioError):
    """Raised when no speech segment can be found in an otherwise valid clip."""


def load_wav(source: Source) -> tuple[np.ndarray, int]:
    """Read a WAV file (path, bytes or file-like object).

    Returns mono float32 samples in [-1, 1] and the original sample rate.
    """
    if isinstance(source, (str, os.PathLike)):
        if not os.fspath(source).lower().endswith(".wav"):
            raise AudioError("Unsupported file type: only .wav files are accepted.")
    elif isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    try:
        sr, data = wavfile.read(source)
    except FileNotFoundError as exc:
        raise AudioError(f"File not found: {source}") from exc
    except Exception as exc:  # scipy raises several different error types
        raise AudioError(f"Could not read WAV data: {exc}") from exc

    if data.dtype == np.int16:
        y = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        y = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.uint8:
        y = (data.astype(np.float32) - 128.0) / 128.0
    elif data.dtype in (np.float32, np.float64):
        y = data.astype(np.float32)
    else:
        raise AudioError(f"Unsupported sample format: {data.dtype}")

    if y.ndim == 2:  # stereo / multichannel -> mono
        y = y.mean(axis=1)
    return y.astype(np.float32), int(sr)


def resample(y: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Polyphase resampling to ``target_sr`` (no-op when rates already match)."""
    if sr <= 0:
        raise AudioError("Sample rate must be positive.")
    if sr == target_sr:
        return y.astype(np.float32)
    g = math.gcd(int(sr), int(target_sr))
    out = resample_poly(y, target_sr // g, sr // g)
    return out.astype(np.float32)


def validate_audio(y: np.ndarray, sr: int, min_duration: float = MIN_DURATION_S,
                   max_duration: float = MAX_DURATION_S) -> float:
    """Check an audio clip and return its duration in seconds."""
    if y is None or len(y) == 0:
        raise AudioError("The audio clip is empty.")
    if not np.all(np.isfinite(y)):
        raise AudioError("The audio contains NaN or infinite values.")
    duration = len(y) / float(sr)
    if duration < min_duration:
        raise AudioError(f"Audio too short ({duration:.2f}s); minimum is {min_duration:.2f}s.")
    if duration > max_duration:
        raise AudioError(f"Audio too long ({duration:.1f}s); maximum is {max_duration:.0f}s.")
    if float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) < SILENCE_RMS:
        raise AudioError("The audio is silent.")
    return duration


def preprocess(source: Source) -> tuple[np.ndarray, int]:
    """Load -> mono -> resample to 16 kHz -> validate -> DC removal -> peak-normalise."""
    y, sr = load_wav(source)
    y = resample(y, sr, TARGET_SR)
    validate_audio(y, TARGET_SR)
    y = y - float(np.mean(y))
    peak = float(np.max(np.abs(y)))
    if peak > 0:
        y = y * (0.95 / peak)
    return y.astype(np.float32), TARGET_SR
