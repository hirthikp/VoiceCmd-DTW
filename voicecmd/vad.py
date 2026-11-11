"""Energy-based voice activity detection (end-point detection)."""

from __future__ import annotations

import numpy as np

from .features import frame_signal


def frame_energy_db(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0) -> np.ndarray:
    frame_len = int(round(sr * frame_ms / 1000.0))
    hop = int(round(sr * hop_ms / 1000.0))
    frames = frame_signal(np.asarray(y, dtype=np.float64), frame_len, hop)
    return 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12)


def detect_speech(y: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0,
                  min_delta_db: float = 3.0, min_run_frames: int = 4, pad_ms: float = 60.0):
    """Locate the speech segment of a single-utterance clip.

    The threshold is adaptive: ``noise_floor + max(min_delta_db, 0.3*(peak - noise_floor))``,
    where the noise floor is the 10th percentile and the peak the 95th percentile of the
    frame energies. Runs shorter than ``min_run_frames`` (~55 ms; clicks) are discarded, and the
    segment spans from the first to the last remaining run, so short closures inside a
    word (e.g. the silence before a plosive) are kept.

    Returns ``(start_sample, end_sample)`` or ``None`` when no speech is found.
    """
    energy = frame_energy_db(y, sr, frame_ms, hop_ms)
    floor = float(np.percentile(energy, 10))
    peak = float(np.percentile(energy, 95))
    if peak - floor < min_delta_db:
        return None
    thr = floor + max(min_delta_db, 0.3 * (peak - floor))
    active = energy > thr

    runs = []  # (start_frame, end_frame_exclusive)
    start = None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        elif not a and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(active)))
    runs = [r for r in runs if r[1] - r[0] >= min_run_frames]
    if not runs:
        return None

    hop = int(round(sr * hop_ms / 1000.0))
    pad = int(round(sr * pad_ms / 1000.0))
    s = max(0, runs[0][0] * hop - pad)
    e = min(len(y), runs[-1][1] * hop + int(round(sr * frame_ms / 1000.0)) + pad)
    return int(s), int(e)
