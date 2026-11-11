import numpy as np
import pytest

from voicecmd import FeatureConfig, extract_features
from voicecmd.features import delta, frame_signal, hz_to_mel, mel_filterbank, mel_to_hz


def test_mel_scale_roundtrip_and_reference_point():
    f = np.array([0.0, 100.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f)
    assert hz_to_mel(1000.0) == pytest.approx(1000.0, abs=1.0)   # ~1000 mel at 1 kHz


def test_filterbank_shape_range_and_coverage():
    fb = mel_filterbank(16000, 512, 26, 20.0)
    assert fb.shape == (26, 257)
    assert fb.min() >= 0 and fb.max() <= 1.0 + 1e-9
    assert np.all(fb.sum(axis=1) > 0)
    centres = fb.argmax(axis=1)
    assert np.all(np.diff(centres) > 0)       # filters ordered low -> high
    with pytest.raises(ValueError):
        mel_filterbank(16000, 512, 26, 9000.0)


def test_frame_signal_counts_and_padding():
    y = np.arange(1000, dtype=float)
    f = frame_signal(y, 400, 160)
    assert f.shape == (1 + int(np.ceil((1000 - 400) / 160)), 400)
    assert np.array_equal(f[0], y[:400])
    assert frame_signal(np.ones(100), 400, 160).shape == (1, 400)


def test_delta_of_linear_ramp_is_constant_slope():
    ramp = np.arange(20, dtype=float)[:, None] * 2.0
    d = delta(ramp, 2)
    assert np.allclose(d[3:-3], 2.0)


def test_feature_shape_dim_and_cmvn():
    sr = 16000
    y = np.random.default_rng(0).standard_normal(sr)
    cfg = FeatureConfig()
    f = extract_features(y, cfg)
    assert f.shape[1] == cfg.dim == 26
    assert f.shape[0] == 1 + int(np.ceil((sr - cfg.frame_len) / cfg.hop))
    assert np.allclose(f.mean(axis=0), 0, atol=1e-6)
    assert np.allclose(f.std(axis=0), 1, atol=1e-3)
    assert np.all(np.isfinite(f))


def test_feature_options_change_dimension():
    y = np.random.default_rng(1).standard_normal(8000)
    assert extract_features(y, FeatureConfig(use_delta=False)).shape[1] == 13
    assert extract_features(y, FeatureConfig(use_energy=False, use_delta=False)).shape[1] == 12


def test_invalid_config_and_input():
    with pytest.raises(ValueError):
        FeatureConfig(n_fft=256)
    with pytest.raises(ValueError):
        FeatureConfig(n_mfcc=30)
    with pytest.raises(ValueError):
        extract_features(np.array([]))


def test_tone_energy_lands_in_expected_mel_band():
    sr = 16000
    t = np.arange(sr) / sr
    cfg = FeatureConfig(cmvn=False, use_delta=False, use_energy=False, preemph=0.0)
    low = extract_features(np.sin(2 * np.pi * 300 * t), cfg)
    high = extract_features(np.sin(2 * np.pi * 3000 * t), cfg)
    assert not np.allclose(low.mean(axis=0), high.mean(axis=0), atol=0.5)
