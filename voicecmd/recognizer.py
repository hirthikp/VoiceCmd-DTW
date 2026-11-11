"""Template-based isolated-word recogniser (MFCC + DTW + k-NN per word)."""

from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from .audio_io import AudioError, NoSpeechError, resample, validate_audio, TARGET_SR
from .dtw import dtw
from .features import FeatureConfig, extract_features
from .vad import detect_speech


@dataclass
class Recognition:
    word: str                       # best-matching vocabulary word
    distance: float                 # DTW distance to that word
    margin: float                   # (second - best) / second, in [0, 1]
    accepted: bool                  # False if rejected by the distance threshold
    ranking: list[tuple[str, float]]
    audio_duration_s: float
    elapsed_s: float
    rtf: float                      # real-time factor = processing time / audio duration
    segment_s: tuple[float, float] | None
    n_frames: int
    best_template: int | None = None  # index of the closest template (for plotting)

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "accepted": self.accepted,
            "distance": round(self.distance, 4),
            "margin": round(self.margin, 4),
            "ranking": [{"word": w, "distance": round(d, 4)} for w, d in self.ranking],
            "audio_duration_s": round(self.audio_duration_s, 3),
            "processing_time_s": round(self.elapsed_s, 4),
            "real_time_factor": round(self.rtf, 4),
            "speech_segment_s": None if self.segment_s is None else [round(v, 3) for v in self.segment_s],
            "n_frames": self.n_frames,
        }


class DTWRecognizer:
    """Stores labelled feature templates and classifies new utterances by DTW distance."""

    def __init__(self, config: FeatureConfig | None = None, band: float | None = 0.3, k: int = 1,
                 reject_threshold: float | None = None, use_vad: bool = True):
        if k < 1:
            raise ValueError("k must be >= 1.")
        self.config = config or FeatureConfig()
        self.band = band
        self.k = k
        self.reject_threshold = reject_threshold
        self.use_vad = use_vad
        self._templates: list[tuple[str, np.ndarray]] = []

    # ------------------------------------------------------------------ enrolment
    @property
    def words(self) -> list[str]:
        return sorted({w for w, _ in self._templates})

    @property
    def n_templates(self) -> int:
        return len(self._templates)

    def template_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for w, _ in self._templates:
            counts[w] = counts.get(w, 0) + 1
        return dict(sorted(counts.items()))

    def clear(self) -> None:
        self._templates.clear()

    def segment(self, y: np.ndarray, sr: int = TARGET_SR):
        """Return the (possibly trimmed) signal and the speech segment in samples."""
        y = resample(np.asarray(y, dtype=np.float32), sr, self.config.sr)
        validate_audio(y, self.config.sr)
        if not self.use_vad:
            return y, (0, len(y))
        seg = detect_speech(y, self.config.sr)
        if seg is None:
            raise NoSpeechError("No speech detected in the clip.")
        return y, seg

    def features_from_audio(self, y: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
        y, (s, e) = self.segment(y, sr)
        return extract_features(y[s:e], self.config)

    def add_template(self, word: str, y: np.ndarray, sr: int = TARGET_SR) -> None:
        word = word.strip().lower()
        if not word:
            raise ValueError("Word label must not be empty.")
        self._templates.append((word, self.features_from_audio(y, sr)))

    # ---------------------------------------------------------------- recognition
    def scores(self, feats: np.ndarray, return_best_index: bool = False):
        if not self._templates:
            raise RuntimeError("No templates enrolled: add words before recognising.")
        per_word: dict[str, list[tuple[float, int]]] = {}
        for idx, (w, t) in enumerate(self._templates):
            d = dtw(feats, t, band=self.band).distance
            per_word.setdefault(w, []).append((d, idx))
        out: dict[str, float] = {}
        best_idx: dict[str, int] = {}
        for w, lst in per_word.items():
            lst.sort()
            top = lst[: self.k]
            out[w] = float(np.mean([d for d, _ in top]))
            best_idx[w] = top[0][1]
        return (out, best_idx) if return_best_index else out

    def recognize(self, y: np.ndarray, sr: int = TARGET_SR) -> Recognition:
        t0 = time.perf_counter()
        y16 = resample(np.asarray(y, dtype=np.float32), sr, self.config.sr)
        duration = validate_audio(y16, self.config.sr)
        y16, (s, e) = self.segment(y16, self.config.sr)
        feats = extract_features(y16[s:e], self.config)
        sc, best_idx = self.scores(feats, return_best_index=True)

        ranking = sorted(sc.items(), key=lambda kv: kv[1])
        best_word, best_d = ranking[0]
        margin = 1.0
        if len(ranking) > 1 and ranking[1][1] > 0:
            margin = float(np.clip((ranking[1][1] - best_d) / ranking[1][1], 0.0, 1.0))
        accepted = self.reject_threshold is None or best_d <= self.reject_threshold
        elapsed = time.perf_counter() - t0
        return Recognition(
            word=best_word, distance=float(best_d), margin=margin, accepted=bool(accepted),
            ranking=ranking, audio_duration_s=duration, elapsed_s=elapsed,
            rtf=elapsed / duration, segment_s=(s / self.config.sr, e / self.config.sr),
            n_frames=len(feats), best_template=best_idx[best_word],
        )

    def template_features(self, index: int) -> np.ndarray:
        return self._templates[index][1]

    def suggest_threshold(self, factor: float = 1.25) -> float:
        """Rejection threshold from leave-one-out distances between same-word templates.

        Takes the 95th percentile of each template's distance to its nearest same-word
        template and multiplies by ``factor``. Needs >= 2 templates for some word.
        """
        dists = []
        for i, (w, f) in enumerate(self._templates):
            same = [dtw(f, g, band=self.band).distance
                    for j, (w2, g) in enumerate(self._templates) if j != i and w2 == w]
            if same:
                dists.append(min(same))
        if not dists:
            raise RuntimeError("Need at least two templates of one word to calibrate a threshold.")
        return float(np.percentile(dists, 95) * factor)

    # ----------------------------------------------------------------- persistence
    def save(self, target) -> None:
        """Write the model to a ``.npz`` path or file-like object (no pickling)."""
        arrays = {f"t{i:05d}": f for i, (_, f) in enumerate(self._templates)}
        meta = {"config": self.config.to_dict(), "band": self.band, "k": self.k,
                "reject_threshold": self.reject_threshold, "use_vad": self.use_vad}
        labels = np.array([w for w, _ in self._templates], dtype="U64") if self._templates else np.array([], dtype="U1")
        np.savez_compressed(target, labels=labels, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, source) -> "DTWRecognizer":
        if isinstance(source, (bytes, bytearray)):
            source = io.BytesIO(source)
        elif isinstance(source, (str, os.PathLike)) and not os.path.exists(source):
            raise AudioError(f"Model file not found: {source}")
        try:
            data = np.load(source, allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            labels = [str(s) for s in data["labels"]]
            rec = cls(FeatureConfig(**meta["config"]), band=meta["band"], k=meta["k"],
                      reject_threshold=meta["reject_threshold"], use_vad=meta["use_vad"])
            for i, w in enumerate(labels):
                rec._templates.append((w, np.asarray(data[f"t{i:05d}"])))
        except (KeyError, ValueError, OSError) as exc:
            raise ValueError(f"Invalid model file: {exc}") from exc
        return rec
