import numpy as np
import pytest

from surfanalysis.metrics.com import compute_com


def _kp(positions: dict[int, tuple[float, float]], default_vis: float = 0.9):
    """Build a (33, 4) keypoint array; unset indices get visibility 0."""
    arr = np.zeros((33, 4), dtype=np.float64)
    for i, (x, y) in positions.items():
        arr[i] = (x, y, 0.0, default_vis)
    return arr


def test_com_symmetric_tpose_centered():
    """T-pose with all segments present: CoM should be near body centroid."""
    positions = {
        0: (0.5, 0.10),                                # NOSE
        11: (0.4, 0.30), 12: (0.6, 0.30),              # shoulders
        13: (0.3, 0.40), 14: (0.7, 0.40),              # elbows
        15: (0.2, 0.50), 16: (0.8, 0.50),              # wrists
        23: (0.45, 0.55), 24: (0.55, 0.55),            # hips
        25: (0.45, 0.75), 26: (0.55, 0.75),            # knees
        27: (0.45, 0.95), 28: (0.55, 0.95),            # ankles
        31: (0.45, 1.00), 32: (0.55, 1.00),            # feet
    }
    kp = _kp(positions)
    com = compute_com(kp)
    assert com is not None
    assert com[0] == pytest.approx(0.5, abs=0.01)
    assert 0.4 < com[1] < 0.7  # below shoulders, above feet


def test_com_returns_none_when_too_many_segments_missing():
    """Only NOSE visible → not enough mass to compute CoM."""
    kp = _kp({0: (0.5, 0.5)})
    assert compute_com(kp) is None


def test_com_skips_low_visibility_points():
    positions = {
        0: (0.5, 0.10),
        11: (0.4, 0.30), 12: (0.6, 0.30),
        23: (0.45, 0.55), 24: (0.55, 0.55),
        25: (0.45, 0.75), 26: (0.55, 0.75),
        27: (0.45, 0.95), 28: (0.55, 0.95),
        31: (0.45, 1.00), 32: (0.55, 1.00),
    }
    kp = _kp(positions)
    # mark left shoulder as low visibility
    kp[11, 3] = 0.2
    com = compute_com(kp)
    assert com is not None  # still enough mass present