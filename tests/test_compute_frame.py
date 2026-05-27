import numpy as np

from surfanalysis.metrics import compute_frame_metrics
from surfanalysis.metrics.stability import StabilityWindow


def _full_kp():
    arr = np.zeros((33, 4), dtype=np.float64)
    placements = {
        0: (0.5, 0.10),
        11: (0.45, 0.30), 12: (0.55, 0.30),
        13: (0.40, 0.40), 14: (0.60, 0.40),
        15: (0.35, 0.50), 16: (0.65, 0.50),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.45, 0.92), 28: (0.55, 0.92),
        31: (0.45, 0.95), 32: (0.55, 0.95),
    }
    for i, (x, y) in placements.items():
        arr[i] = (x, y, 0.0, 0.9)
    return arr


def test_compute_frame_metrics_returns_complete_struct():
    kp = _full_kp()
    win = StabilityWindow()
    fm = compute_frame_metrics(kp, stance="regular", stability_window=win)
    assert fm is not None
    assert 0.0 <= fm.com[0] <= 1.0
    assert 0.0 <= fm.weight_dist_front_pct <= 100.0
    assert fm.knee_angle_left is not None
    assert fm.elbow_angle_left is not None
    assert fm.torso_lean_deg is not None
    assert fm.shoulder_hip_rotation_deg is not None
    assert fm.com_stability_score is None


def test_compute_frame_metrics_returns_none_when_com_unavailable():
    kp = np.zeros((33, 4), dtype=np.float64)
    win = StabilityWindow()
    fm = compute_frame_metrics(kp, stance="regular", stability_window=win)
    assert fm is None
