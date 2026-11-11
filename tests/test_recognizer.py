import io

import numpy as np
import pytest

from voicecmd import AudioError, DTWRecognizer, NoSpeechError, evaluate
from voicecmd.synthetic import SR, WORDS, make_dataset, synth_word, write_demo_dataset
from voicecmd.datasets import enroll_dir, load_labelled_dir


@pytest.fixture(scope="module")
def trained():
    rec = DTWRecognizer()
    for w, y, sr in make_dataset(speakers=range(0, 3), reps=2, seed=1):
        rec.add_template(w, y, sr)
    return rec


def test_synth_is_deterministic_and_speakers_differ():
    a = synth_word("go", 0, seed=5)
    assert np.array_equal(a, synth_word("go", 0, seed=5))
    assert not np.array_equal(a, synth_word("go", 1, seed=5))
    with pytest.raises(KeyError):
        synth_word("banana")


def test_enrolment_bookkeeping(trained):
    assert trained.words == sorted(WORDS)
    assert trained.n_templates == 6 * 3 * 2
    assert set(trained.template_counts().values()) == {6}


def test_recognises_unseen_speakers(trained):
    report = evaluate(trained, make_dataset(speakers=range(10, 13), reps=2, seed=9))
    assert report.accuracy >= 0.95
    assert report.confusion.sum() == report.n == 36


def test_result_fields_are_consistent(trained):
    res = trained.recognize(synth_word("stop", 11, seed=3), SR)
    assert res.word == "stop" and res.accepted
    assert res.ranking[0][0] == res.word and res.ranking == sorted(res.ranking, key=lambda r: r[1])
    assert 0.0 <= res.margin <= 1.0
    assert res.rtf == pytest.approx(res.elapsed_s / res.audio_duration_s)
    s, e = res.segment_s
    assert 0 <= s < e <= res.audio_duration_s + 1e-6
    d = res.to_dict()
    assert d["word"] == "stop" and len(d["ranking"]) == 6


def test_recognises_other_sample_rates(trained):
    from scipy.signal import resample_poly
    y8k = resample_poly(synth_word("left", 12, seed=4), 1, 2).astype(np.float32)
    assert trained.recognize(y8k, 8000).word == "left"


def test_errors_for_bad_input_and_empty_model(trained):
    with pytest.raises(RuntimeError):
        DTWRecognizer().recognize(synth_word("go"), SR)
    with pytest.raises(AudioError):
        trained.recognize(np.zeros(SR, dtype=np.float32), SR)
    with pytest.raises(NoSpeechError):
        trained.recognize((np.random.default_rng(0).standard_normal(SR) * 1e-3).astype(np.float32), SR)
    with pytest.raises(ValueError):
        trained.add_template("   ", synth_word("go"), SR)
    with pytest.raises(ValueError):
        DTWRecognizer(k=0)


def test_labels_are_normalised():
    rec = DTWRecognizer()
    rec.add_template("  GO ", synth_word("go", 0, 1), SR)
    assert rec.words == ["go"]


def test_rejection_of_out_of_vocabulary_word():
    rec = DTWRecognizer()
    for w, y, sr in make_dataset(words=["go", "left", "right", "up", "down"], speakers=range(0, 4), reps=2, seed=1):
        rec.add_template(w, y, sr)
    rec.reject_threshold = rec.suggest_threshold()
    assert rec.reject_threshold > 0
    known = rec.recognize(synth_word("left", 11, seed=2), SR)
    unknown = rec.recognize(synth_word("stop", 11, seed=2), SR)
    assert known.accepted and not unknown.accepted


def test_suggest_threshold_needs_repeated_templates():
    rec = DTWRecognizer()
    rec.add_template("go", synth_word("go", 0, 1), SR)
    rec.add_template("up", synth_word("up", 0, 1), SR)
    with pytest.raises(RuntimeError):
        rec.suggest_threshold()


def test_save_and_load_roundtrip(trained, tmp_path):
    path = tmp_path / "m.npz"
    trained.save(str(path))
    loaded = DTWRecognizer.load(str(path))
    assert loaded.words == trained.words and loaded.n_templates == trained.n_templates
    assert loaded.band == trained.band and loaded.config == trained.config
    y = synth_word("up", 11, seed=8)
    a, b = loaded.recognize(y, SR).ranking, trained.recognize(y, SR).ranking
    assert [w for w, _ in a] == [w for w, _ in b]
    assert np.allclose([d for _, d in a], [d for _, d in b])
    buf = io.BytesIO()
    trained.save(buf)
    assert DTWRecognizer.load(buf.getvalue()).n_templates == trained.n_templates


def test_load_rejects_bad_files(tmp_path):
    with pytest.raises(AudioError):
        DTWRecognizer.load(str(tmp_path / "nope.npz"))
    bad = tmp_path / "bad.npz"
    np.savez(bad, x=np.zeros(3))
    with pytest.raises(ValueError):
        DTWRecognizer.load(str(bad))


def test_k_nearest_averaging_changes_scores(trained):
    y = synth_word("down", 12, seed=6)
    d1 = dict(trained.recognize(y, SR).ranking)
    trained.k = 3
    try:
        d3 = dict(trained.recognize(y, SR).ranking)
    finally:
        trained.k = 1
    assert all(d3[w] >= d1[w] - 1e-9 for w in d1)


def test_demo_dataset_folder_roundtrip(tmp_path):
    counts = write_demo_dataset(str(tmp_path))
    assert counts == {"enroll": 36, "test": 24, "test_noisy": 12}
    rec = DTWRecognizer()
    assert enroll_dir(rec, str(tmp_path / "enroll")) == 36
    report = evaluate(rec, load_labelled_dir(str(tmp_path / "test")))
    assert report.accuracy >= 0.95 and report.n == 24
    assert report.per_class()["go"]["support"] == 4
    with pytest.raises(FileNotFoundError):
        load_labelled_dir(str(tmp_path / "missing"))
