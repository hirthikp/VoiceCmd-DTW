"""Helpers for folder-structured datasets: ``root/<word>/<clip>.wav``."""

from __future__ import annotations

import os

from .audio_io import load_wav


def list_labelled_files(root: str) -> list[tuple[str, str]]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    items = []
    for word in sorted(os.listdir(root)):
        folder = os.path.join(root, word)
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith(".wav"):
                items.append((word, os.path.join(folder, name)))
    return items


def load_labelled_dir(root: str) -> list[tuple[str, "object", int]]:
    """Return ``[(word, samples, sample_rate), ...]`` for every WAV in ``root/<word>/``."""
    out = []
    for word, path in list_labelled_files(root):
        y, sr = load_wav(path)
        out.append((word, y, sr))
    return out


def enroll_dir(recognizer, root: str) -> int:
    """Enrol every clip under ``root`` into ``recognizer``; returns the number enrolled."""
    n = 0
    for word, y, sr in load_labelled_dir(root):
        recognizer.add_template(word, y, sr)
        n += 1
    return n
