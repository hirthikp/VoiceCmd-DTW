"""Accuracy, confusion matrix and per-class metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio_io import AudioError


@dataclass
class EvalReport:
    labels: list[str]
    confusion: np.ndarray          # shape (L, L + 1); last column = failed (no speech / error)
    n: int
    n_correct: int
    n_failed: int
    mean_rtf: float

    @property
    def accuracy(self) -> float:
        return self.n_correct / self.n if self.n else 0.0

    def per_class(self) -> dict[str, dict[str, float]]:
        out = {}
        L = len(self.labels)
        for i, w in enumerate(self.labels):
            tp = self.confusion[i, i]
            support = self.confusion[i].sum()
            predicted = self.confusion[:, i].sum()
            out[w] = {
                "precision": float(tp / predicted) if predicted else 0.0,
                "recall": float(tp / support) if support else 0.0,
                "support": int(support),
            }
        return out

    def to_dict(self) -> dict:
        return {"accuracy": self.accuracy, "n": self.n, "n_correct": self.n_correct,
                "n_failed": self.n_failed, "mean_rtf": self.mean_rtf,
                "labels": self.labels, "confusion": self.confusion.tolist(),
                "per_class": self.per_class()}


def evaluate(recognizer, dataset) -> EvalReport:
    """Evaluate ``recognizer`` on an iterable of ``(true_word, samples, sample_rate)``.

    Clips that cannot be processed (e.g. no speech found) count as errors.
    """
    items = list(dataset)
    labels = sorted(set(recognizer.words) | {w for w, *_ in items})
    index = {w: i for i, w in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels) + 1), dtype=int)
    correct = failed = 0
    rtfs = []
    for word, y, sr in items:
        t = index[word]
        try:
            res = recognizer.recognize(y, sr)
        except AudioError:
            cm[t, -1] += 1
            failed += 1
            continue
        rtfs.append(res.rtf)
        p = index[res.word]
        cm[t, p] += 1
        correct += int(p == t)
    return EvalReport(labels, cm, len(items), correct, failed, float(np.mean(rtfs)) if rtfs else 0.0)
