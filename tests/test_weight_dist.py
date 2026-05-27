import numpy as np
import pytest

from surfanalysis.metrics.weight_dist import compute_weight_dist_front_pct


def _kp_with_feet(l_foot: tuple[float, float], r_foot: tuple[float, float]):
    arr = np.zeros((33, 4), dtype=np.float64)
    arr[31] = (*l_foot, 0.0, 0.9)
    arr[32] = (*r_foot, 0.0, 0.9)
    return arr


def test_com_at_midpoint_returns_50pct():
    kp = _kp_with_feet((0.0, 1.0), (1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(0.5, 1.0), stance="regular")
    assert pct == pytest.approx(50.0, abs=0.5)


def test_com_at_front_foot_returns_100pct_regular():
    """Regular stance: L_FOOT (idx 31) is front."""
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(0.0, 1.0), stance="regular")
    assert pct == pytest.approx(100.0)


def test_com_at_back_foot_returns_0pct_regular():
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(1.0, 1.0), stance="regular")
    assert pct == pytest.approx(0.0)


def test_goofy_swaps_front_back():
    """Goofy stance: R_FOOT (idx 32) is front."""
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(1.0, 1.0), stance="goofy")
    assert pct == pytest.approx(100.0)


def test_com_beyond_segment_clamps():
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct_far = compute_weight_dist_front_pct(kp, com=(-2.0, 1.0), stance="regular")
    assert pct_far == pytest.approx(100.0)


def test_returns_none_when_foot_missing():
    arr = np.zeros((33, 4))  # all zeros, no visibility
    pct = compute_weight_dist_front_pct(arr, com=(0.5, 1.0), stance="regular")
    assert pct is None
