import numpy as np

from voicecmd import detect_speech


def make_clip(sr=16000, lead=0.5, speech=0.4, trail=0.5, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    n = int(sr * (lead + speech + trail))
    y = rng.standard_normal(n) * noise
    s0, s1 = int(sr * lead), int(sr * (lead + speech))
    t = np.arange(s1 - s0) / sr
    y[s0:s1] += 0.3 * np.sin(2 * np.pi * 220 * t)
    return y, s0, s1


def test_detects_segment_boundaries_within_tolerance():
    y, s0, s1 = make_clip()
    seg = detect_speech(y, 16000)
    assert seg is not None
    start, end = seg
    assert abs(start - s0) < 16000 * 0.12
    assert abs(end - s1) < 16000 * 0.12


def test_pure_noise_returns_none():
    y = np.random.default_rng(1).standard_normal(16000) * 1e-3
    assert detect_speech(y, 16000) is None


def test_short_click_is_ignored_but_word_is_found():
    y, s0, s1 = make_clip()
    y[2000:2020] += 0.8      # ~1 ms click: spans at most 3 frames, below min_run_frames
    start, end = detect_speech(y, 16000)
    assert start > 2400      # click not included as the start of speech


def test_internal_closure_does_not_split_the_segment():
    y, s0, s1 = make_clip(speech=0.6)
    mid = (s0 + s1) // 2
    y[mid - 400: mid + 400] = np.random.default_rng(2).standard_normal(800) * 1e-3   # 50 ms gap
    start, end = detect_speech(y, 16000)
    assert start < mid < end and end - start > 0.55 * 16000
