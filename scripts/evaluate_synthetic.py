"""Reproducible speaker-independent evaluation on synthetic speech.

Templates come from speakers 0-4 (2 clean repetitions each); tests use *unseen*
speakers 5-14 (3 repetitions each) at several signal-to-noise ratios, plus ablations.
Results are written to results/results.json and results/confusion_clean.png.
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")

from voicecmd import DTWRecognizer, FeatureConfig, evaluate
from voicecmd.plots import plot_confusion
from voicecmd.synthetic import make_dataset

TRAIN_SPEAKERS = range(0, 5)
TEST_SPEAKERS = range(5, 15)
SNRS = [None, 30, 20, 15, 10, 5, 0]


def build(config=None, **kw):
    rec = DTWRecognizer(config or FeatureConfig(), **kw)
    for w, y, sr in make_dataset(speakers=TRAIN_SPEAKERS, reps=2, seed=1):
        rec.add_template(w, y, sr)
    return rec


def main():
    out = {"setup": {"train_speakers": list(TRAIN_SPEAKERS), "test_speakers": list(TEST_SPEAKERS),
                     "templates_per_word": 10, "test_utterances": 6 * len(TEST_SPEAKERS) * 3}}
    t0 = time.time()
    clean_test = make_dataset(speakers=TEST_SPEAKERS, reps=3, seed=2)
    base = build()

    # 1) noise robustness
    noise = []
    for snr in SNRS:
        data = clean_test if snr is None else make_dataset(speakers=TEST_SPEAKERS, reps=3, snr_db=snr, seed=2)
        rep = evaluate(base, data)
        noise.append({"snr_db": "clean" if snr is None else snr, "accuracy": rep.accuracy,
                      "failed": rep.n_failed, "mean_rtf": rep.mean_rtf})
        print(f"SNR {snr}: acc={rep.accuracy:.3f} failed={rep.n_failed}", flush=True)
    out["noise"] = noise

    base_rep = evaluate(base, clean_test)
    out["clean"] = base_rep.to_dict()
    fig = plot_confusion(base_rep.confusion, base_rep.labels, "Confusion matrix (clean, unseen speakers)")
    fig.savefig(os.path.join(ROOT, "results", "confusion_clean.png"), dpi=160)

    noisy10 = make_dataset(speakers=TEST_SPEAKERS, reps=3, snr_db=10, seed=2)
    # 2) ablations, scored on clean and on 10 dB noisy test data
    variants = {
        "Baseline (MFCC+energy+delta, CMVN, VAD, band 0.3, k=1)": build(),
        "No delta features": build(FeatureConfig(use_delta=False)),
        "No CMVN": build(FeatureConfig(cmvn=False)),
        "No VAD (whole clip)": build(use_vad=False),
        "Narrow band (0.1)": build(band=0.1),
        "No band (full DTW)": build(band=None),
        "k = 3 nearest templates": build(k=3),
    }
    abl = []
    for name, rec in variants.items():
        c = evaluate(rec, clean_test)
        n = evaluate(rec, noisy10)
        abl.append({"variant": name, "clean": c.accuracy, "snr10": n.accuracy,
                    "clean_rtf": c.mean_rtf})
        print(f"{name}: clean={c.accuracy:.3f} 10dB={n.accuracy:.3f} rtf={c.mean_rtf:.4f}", flush=True)
    out["ablation"] = abl

    out["rejection"] = rejection_test(base)
    out["elapsed_s"] = time.time() - t0
    with open(os.path.join(ROOT, "results", "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("saved results/results.json in %.1fs" % out["elapsed_s"])


def rejection_test(base):
    """Out-of-vocabulary check: the vocabulary minus 'stop' is enrolled and 'stop' is
    presented as an unknown word; a threshold from leave-one-out calibration is used."""
    rec = DTWRecognizer()
    for w, y, sr in make_dataset(words=["go", "left", "right", "up", "down"],
                                 speakers=TRAIN_SPEAKERS, reps=2, seed=1):
        rec.add_template(w, y, sr)
    thr = rec.suggest_threshold()
    rec.reject_threshold = thr
    known = make_dataset(words=["go", "left", "right", "up", "down"], speakers=TEST_SPEAKERS, reps=1, seed=3)
    unknown = make_dataset(words=["stop"], speakers=TEST_SPEAKERS, reps=3, seed=3)
    acc_known = sum(rec.recognize(y, sr).accepted for _, y, sr in known) / len(known)
    rej_unknown = sum(not rec.recognize(y, sr).accepted for _, y, sr in unknown) / len(unknown)
    return {"threshold": thr, "known_accepted": acc_known, "unknown_rejected": rej_unknown,
            "n_known": len(known), "n_unknown": len(unknown)}


if __name__ == "__main__":
    main()
