"""MFCC feature extraction implemented from scratch (NumPy + SciPy FFT/DCT)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.fft import dct


@dataclass
class FeatureConfig:
    sr: int = 16000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 12            # cepstra c1..c12
    use_energy: bool = True     # append log frame energy -> 13 static coefficients
    preemph: float = 0.97
    lifter: int = 22
    fmin: float = 20.0
    fmax: float | None = None   # defaults to sr/2
    use_delta: bool = True
    delta_width: int = 2
    cmvn: bool = True           # per-utterance mean/variance normalisation

    def __post_init__(self):
        if self.frame_len > self.n_fft:
            raise ValueError("n_fft must be >= frame length in samples.")
        if self.n_mfcc >= self.n_mels:
            raise ValueError("n_mfcc must be smaller than n_mels.")

    @property
    def frame_len(self) -> int:
        return int(round(self.sr * self.frame_ms / 1000.0))

    @property
    def hop(self) -> int:
        return int(round(self.sr * self.hop_ms / 1000.0))

    @property
    def dim(self) -> int:
        static = self.n_mfcc + (1 if self.use_energy else 0)
        return static * (2 if self.use_delta else 1)

    def to_dict(self) -> dict:
        return asdict(self)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 0.0, fmax: float | None = None) -> np.ndarray:
    """Triangular mel filterbank, shape ``(n_mels, n_fft//2 + 1)``."""
    fmax = sr / 2.0 if fmax is None else fmax
    if not 0 <= fmin < fmax <= sr / 2.0:
        raise ValueError("Require 0 <= fmin < fmax <= sr/2.")
    mel_pts = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    freqs = np.linspace(0.0, sr / 2.0, n_fft // 2 + 1)
    fb = np.zeros((n_mels, len(freqs)))
    for m in range(n_mels):
        left, centre, right = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (freqs - left) / (centre - left)
        down = (right - freqs) / (right - centre)
        fb[m] = np.maximum(0.0, np.minimum(up, down))
    return fb


def frame_signal(y: np.ndarray, frame_len: int, hop: int) -> np.ndarray:
    """Slice ``y`` into overlapping frames; the tail is zero-padded."""
    n = len(y)
    n_frames = 1 if n <= frame_len else 1 + int(np.ceil((n - frame_len) / hop))
    total = (n_frames - 1) * hop + frame_len
    padded = np.pad(y, (0, total - n))
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    return padded[idx]


def delta(feat: np.ndarray, width: int = 2) -> np.ndarray:
    """Regression-based delta coefficients (edge-padded)."""
    T = len(feat)
    padded = np.pad(feat, ((width, width), (0, 0)), mode="edge")
    denom = 2.0 * sum(n * n for n in range(1, width + 1))
    out = np.zeros_like(feat, dtype=np.float64)
    for n in range(1, width + 1):
        out += n * (padded[width + n: width + n + T] - padded[width - n: width - n + T])
    return out / denom


def extract_features(y: np.ndarray, cfg: FeatureConfig | None = None) -> np.ndarray:
    """Return a ``(n_frames, cfg.dim)`` feature matrix for a mono 16 kHz signal."""
    cfg = cfg or FeatureConfig()
    y = np.asarray(y, dtype=np.float64)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError("Expected a non-empty 1-D signal.")

    emph = np.append(y[0], y[1:] - cfg.preemph * y[:-1])
    frames = frame_signal(emph, cfg.frame_len, cfg.hop)
    window = np.hamming(cfg.frame_len)
    spec = np.abs(np.fft.rfft(frames * window, n=cfg.n_fft, axis=1)) ** 2 / cfg.n_fft

    fb = mel_filterbank(cfg.sr, cfg.n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    log_mel = np.log(np.maximum(spec @ fb.T, 1e-10))
    cep = dct(log_mel, type=2, axis=1, norm="ortho")[:, 1: cfg.n_mfcc + 1]
    if cfg.lifter > 0:
        n = np.arange(1, cfg.n_mfcc + 1)
        cep = cep * (1.0 + (cfg.lifter / 2.0) * np.sin(np.pi * n / cfg.lifter))

    static = cep
    if cfg.use_energy:
        log_e = np.log(np.sum(frames ** 2, axis=1) + 1e-10)
        static = np.hstack([cep, log_e[:, None]])

    feats = np.hstack([static, delta(static, cfg.delta_width)]) if cfg.use_delta else static
    if cfg.cmvn:
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)
    return feats
