"""Matplotlib figure helpers used by the Streamlit app and the report."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_waveform(y, sr, segment_s=None):
    fig, ax = plt.subplots(figsize=(7, 2.2))
    t = np.arange(len(y)) / sr
    ax.plot(t, y, lw=0.6, color="#1f4e79")
    if segment_s is not None:
        ax.axvspan(segment_s[0], segment_s[1], color="#2e9e6b", alpha=0.18, label="detected speech")
        ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Waveform and end-point detection")
    fig.tight_layout()
    return fig


def plot_features(feats, title="MFCC features (static + delta, CMVN)"):
    fig, ax = plt.subplots(figsize=(7, 2.8))
    im = ax.imshow(feats.T, aspect="auto", origin="lower", cmap="magma")
    ax.set_xlabel("Frame (10 ms hop)")
    ax.set_ylabel("Coefficient")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def plot_alignment(path, n, m, label_x="Input (frames)", label_y="Best template (frames)"):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.plot([p[0] for p in path], [p[1] for p in path], color="#c0392b", lw=1.8)
    ax.plot([0, n - 1], [0, m - 1], "--", color="grey", lw=0.8)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, m - 1)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_title("DTW warping path")
    fig.tight_layout()
    return fig


def plot_distances(ranking, accepted_word=None):
    words = [w for w, _ in ranking]
    vals = [d for _, d in ranking]
    fig, ax = plt.subplots(figsize=(5.5, 2.8))
    colors = ["#2e9e6b" if i == 0 else "#7f8c8d" for i in range(len(words))]
    ax.barh(words[::-1], vals[::-1], color=colors[::-1])
    ax.set_xlabel("DTW distance (lower = closer)")
    ax.set_title("Distance to each vocabulary word")
    fig.tight_layout()
    return fig


def plot_confusion(cm, labels, title="Confusion matrix"):
    cols = list(labels) + ["failed"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thr = cm.max() / 2 if cm.max() else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    return fig
