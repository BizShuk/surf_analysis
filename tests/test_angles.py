import math
import numpy as np
import pytest

from surfanalysis.metrics.angles import (
    compute_knee_angles,
    compute_elbow_angles,
    compute_torso_lean,
    compute_shoulder_hip_diff,
)


def _kp():
    return np.zeros((33, 4), dtype=np.float64)


def test_knee_straight_returns_180():
    kp = _kp()
    kp[23] = (0.4, 0.5, 0, 0.9)  # L_HIP
    kp[25] = (0.4, 0.7, 0, 0.9)  # L_KNEE
    kp[27] = (0.4, 0.9, 0, 0.9)  # L_ANKLE
    kp[24] = (0.6, 0.5, 0, 0.9)
    kp[26] = (0.6, 0.7, 0, 0.9)
    kp[28] = (0.6, 0.9, 0, 0.9)
    left, right = compute_knee_angles(kp)
    assert left == pytest.approx(180.0)
    assert right == pytest.approx(180.0)


def test_knee_right_angle():
    kp = _kp()
    kp[23] = (0.4, 0.5, 0, 0.9)  # L_HIP
    kp[25] = (0.4, 0.7, 0, 0.9)  # L_KNEE
    kp[27] = (0.6, 0.7, 0, 0.9)  # L_ANKLE (90° bend)
    left, _ = compute_knee_angles(kp)
    assert left == pytest.approx(90.0)


def test_knee_returns_none_when_visibility_low():
    kp = _kp()
    kp[23] = (0.4, 0.5, 0, 0.9)
    kp[25] = (0.4, 0.7, 0, 0.9)
    kp[27] = (0.4, 0.9, 0, 0.9)
    left, right = compute_knee_angles(kp)
    assert left is not None
    assert right is None


def test_elbow_straight_returns_180():
    kp = _kp()
    kp[11] = (0.4, 0.3, 0, 0.9)  # L_SHOULDER
    kp[13] = (0.3, 0.4, 0, 0.9)  # L_ELBOW
    kp[15] = (0.2, 0.5, 0, 0.9)  # L_WRIST
    left, _ = compute_elbow_angles(kp)
    assert left == pytest.approx(180.0)


def test_torso_lean_upright_zero():
    kp = _kp()
    kp[11] = (0.4, 0.3, 0, 0.9)  # L_SHOULDER
    kp[12] = (0.6, 0.3, 0, 0.9)  # R_SHOULDER
    kp[23] = (0.4, 0.6, 0, 0.9)  # L_HIP
    kp[24] = (0.6, 0.6, 0, 0.9)  # R_HIP
    lean = compute_torso_lean(kp)
    assert lean == pytest.approx(0.0, abs=0.1)


def test_torso_lean_forward_positive():
    kp = _kp()
    kp[11] = (0.3, 0.3, 0, 0.9)
    kp[12] = (0.5, 0.3, 0, 0.9)
    kp[23] = (0.4, 0.6, 0, 0.9)
    kp[24] = (0.6, 0.6, 0, 0.9)
    lean = compute_torso_lean(kp)
    assert lean < 0


def test_shoulder_hip_diff_zero_when_aligned():
    kp = _kp()
    kp[11] = (0.4, 0.3, 0, 0.9)
    kp[12] = (0.6, 0.3, 0, 0.9)
    kp[23] = (0.4, 0.6, 0, 0.9)
    kp[24] = (0.6, 0.6, 0, 0.9)
    diff = compute_shoulder_hip_diff(kp)
    assert diff == pytest.approx(0.0, abs=0.5)


def test_shoulder_hip_diff_nonzero_when_twisted():
    kp = _kp()
    kp[11] = (0.4, 0.28, 0, 0.9)
    kp[12] = (0.6, 0.32, 0, 0.9)
    kp[23] = (0.4, 0.60, 0, 0.9)
    kp[24] = (0.6, 0.60, 0, 0.9)
    diff = compute_shoulder_hip_diff(kp)
    assert abs(diff) > 1.0