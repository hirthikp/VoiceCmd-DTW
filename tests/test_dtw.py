import numpy as np
import pytest

from voicecmd import dtw


def test_identical_sequences_have_zero_distance_and_diagonal_path():
    x = np.random.default_rng(0).standard_normal((30, 5))
    r = dtw(x, x, return_path=True)
    assert r.distance == pytest.approx(0.0, abs=1e-12)
    assert r.path == [(i, i) for i in range(30)]


def test_symmetry():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((25, 4)), rng.standard_normal((31, 4))
    assert dtw(a, b, band=None).distance == pytest.approx(dtw(b, a, band=None).distance)


def test_known_small_example_by_hand():
    # 1-D: x = [0, 1, 2], y = [0, 0, 1, 2]; the optimal path repeats the first x frame.
    x = np.array([[0.0], [1.0], [2.0]])
    y = np.array([[0.0], [0.0], [1.0], [2.0]])
    r = dtw(x, y, band=None, return_path=True)
    assert r.cost == pytest.approx(0.0)
    assert r.path[0] == (0, 0) and r.path[-1] == (2, 3)


def test_time_stretched_copy_is_much_closer_than_a_different_signal():
    t = np.linspace(0, 1, 60)
    base = np.stack([np.sin(2 * np.pi * 2 * t), np.cos(2 * np.pi * t)], axis=1)
    stretched = np.stack([np.sin(2 * np.pi * 2 * np.linspace(0, 1, 90)),
                          np.cos(2 * np.pi * np.linspace(0, 1, 90))], axis=1)
    other = np.random.default_rng(3).standard_normal((70, 2))
    assert dtw(base, stretched).distance < 0.2 * dtw(base, other).distance


def test_path_is_monotonic_continuous_and_spans_both_sequences():
    rng = np.random.default_rng(4)
    a, b = rng.standard_normal((20, 3)), rng.standard_normal((27, 3))
    p = dtw(a, b, return_path=True).path
    assert p[0] == (0, 0) and p[-1] == (19, 26)
    for (i0, j0), (i1, j1) in zip(p, p[1:]):
        assert (i1 - i0, j1 - j0) in {(1, 0), (0, 1), (1, 1)}


def test_band_never_lowers_cost_and_full_matches_wide_band():
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((40, 3)), rng.standard_normal((44, 3))
    full = dtw(a, b, band=None).cost
    assert dtw(a, b, band=0.1).cost >= full - 1e-9
    assert dtw(a, b, band=1.0).cost == pytest.approx(full)


def test_very_different_lengths_remain_feasible():
    a = np.random.default_rng(6).standard_normal((10, 2))
    b = np.random.default_rng(7).standard_normal((120, 2))
    assert np.isfinite(dtw(a, b, band=0.05).distance)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        dtw(np.zeros((5, 2)), np.zeros((5, 3)))
    with pytest.raises(ValueError):
        dtw(np.zeros((0, 2)), np.zeros((5, 2)))
    with pytest.raises(ValueError):
        dtw(np.zeros((5, 2)), np.zeros((5, 2)), band=0)
