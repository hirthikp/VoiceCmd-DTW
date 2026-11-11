"""Dynamic Time Warping with a Sakoe-Chiba band and symmetric step weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class DTWResult:
    distance: float                      # path cost normalised by (n + m)
    cost: float                          # raw accumulated cost
    path: list[tuple[int, int]] | None   # (i, j) pairs from start to end, if requested


def _accumulate(c: np.ndarray, w: float) -> np.ndarray:
    """Fill the accumulated-cost matrix inside the band (inf elsewhere).

    Steps (symmetric2): vertical ``D[i-1,j]+c``, horizontal ``D[i,j-1]+c`` and
    diagonal ``D[i-1,j-1]+2c``. The horizontal recursion is solved exactly with a
    cumulative-minimum identity so each row is computed without a Python loop.
    """
    n, m = c.shape
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    slope = m / n
    for i in range(1, n + 1):
        centre = i * slope
        lo = max(1, int(np.ceil(centre - w)))
        hi = min(m, int(np.floor(centre + w)))
        if lo > hi:
            continue
        ci = c[i - 1, lo - 1: hi]
        vert = D[i - 1, lo: hi + 1] + ci
        diag = D[i - 1, lo - 1: hi] + 2.0 * ci
        base = np.minimum(vert, diag)
        S = np.cumsum(ci)
        D[i, lo: hi + 1] = S + np.minimum.accumulate(base - S)
    return D


def dtw(x: np.ndarray, y: np.ndarray, band: float | None = 0.3, return_path: bool = False) -> DTWResult:
    """DTW between feature sequences ``x`` (n, d) and ``y`` (m, d).

    ``band`` is the Sakoe-Chiba radius as a fraction of ``max(n, m)`` measured around the
    (length-scaled) diagonal; ``None`` disables the constraint.
    """
    x = np.atleast_2d(np.asarray(x, dtype=np.float64))
    y = np.atleast_2d(np.asarray(y, dtype=np.float64))
    if x.shape[1] != y.shape[1]:
        raise ValueError("Feature dimensions differ.")
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("Empty sequence.")

    c = cdist(x, y, metric="euclidean")
    if band is None:
        w = float(max(n, m))
    else:
        if band <= 0:
            raise ValueError("band must be positive or None.")
        w = max(band * max(n, m), max(m / n, n / m) + 1.0)
    D = _accumulate(c, w)
    if not np.isfinite(D[n, m]):  # defensive fallback; should not happen
        D = _accumulate(c, float(max(n, m)))

    cost = float(D[n, m])
    path = None
    if return_path:
        i, j = n, m
        path = [(i - 1, j - 1)]
        while (i, j) != (1, 1):
            options = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
            _, i, j = min(options, key=lambda t: t[0])
            path.append((i - 1, j - 1))
        path.reverse()
    return DTWResult(distance=cost / (n + m), cost=cost, path=path)
