import io

import numpy as np
import pytest
from scipy.io import wavfile

from voicecmd import AudioError, load_wav, preprocess, resample, validate_audio


def wav_bytes(y, sr=16000, dtype=np.int16):
    buf = io.BytesIO()
    wavfile.write(buf, sr, y.astype(dtype))
    return buf.getvalue()


def tone(sr=16000, dur=0.5, f=440.0):
    t = np.arange(int(sr * dur)) / sr
    return 0.5 * np.sin(2 * np.pi * f * t)


def test_load_int16_scaling():
    y, sr = load_wav(wav_bytes(tone() * 32767))
    assert sr == 16000 and y.dtype == np.float32
    assert 0.45 < np.max(y) <= 0.5


def test_load_stereo_is_mixed_to_mono():
    stereo = np.stack([tone(), tone()], axis=1) * 32767
    y, _ = load_wav(wav_bytes(stereo))
    assert y.ndim == 1


def test_load_float32_and_uint8():
    y, _ = load_wav(wav_bytes(tone(), dtype=np.float32))
    assert np.isclose(np.max(y), 0.5, atol=1e-3)
    y8, _ = load_wav(wav_bytes(tone() * 127 + 128, dtype=np.uint8))
    assert np.isclose(np.max(y8), 0.5, atol=0.02)


def test_load_int32():
    y, _ = load_wav(wav_bytes(tone() * 2147483647 * 0.999, dtype=np.int32))
    assert np.isclose(np.max(y), 0.5, atol=1e-3)


def test_corrupt_and_missing_files_raise_audio_error(tmp_path):
    with pytest.raises(AudioError):
        load_wav(b"this is not a wav file")
    with pytest.raises(AudioError):
        load_wav(str(tmp_path / "missing.wav"))
    with pytest.raises(AudioError):
        load_wav(str(tmp_path / "clip.mp3"))


def test_resample_changes_length_proportionally():
    y = tone(sr=8000, dur=1.0)
    out = resample(y, 8000, 16000)
    assert len(out) == 16000
    assert resample(y, 8000, 8000).shape == y.shape


def test_validate_audio_rules():
    sr = 16000
    assert validate_audio(tone(), sr) == pytest.approx(0.5)
    with pytest.raises(AudioError):
        validate_audio(np.array([]), sr)
    with pytest.raises(AudioError):
        validate_audio(tone(dur=0.05), sr)          # too short
    with pytest.raises(AudioError):
        validate_audio(tone(dur=11), sr)            # too long
    with pytest.raises(AudioError):
        validate_audio(np.zeros(sr), sr)            # silent
    bad = tone(); bad[10] = np.nan
    with pytest.raises(AudioError):
        validate_audio(bad, sr)


def test_preprocess_resamples_and_normalises():
    y, sr = preprocess(wav_bytes(tone(sr=8000) * 32767, sr=8000))
    assert sr == 16000
    assert np.max(np.abs(y)) == pytest.approx(0.95, abs=1e-3)
    assert abs(float(np.mean(y))) < 1e-3
