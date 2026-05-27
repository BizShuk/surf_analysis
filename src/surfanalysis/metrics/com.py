"""Center of Mass via Plagenhoef segmental mass approximation."""

from __future__ import annotations

import numpy as np

from surfanalysis.extraction.landmarks import (
    L_ANKLE,
    L_ELBOW,
    L_FOOT,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_FOOT,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    VISIBILITY_THRESHOLD,
)

_MIN_PRESENT_MASS = 0.8


def _point_if_visible(kp: np.ndarray, i: int) -> np.ndarray | None:
    if kp[i, 3] < VISIBILITY_THRESHOLD:
        return None
    return kp[i, :2].copy()


def _midpoint_if_both_visible(kp: np.ndarray, a: int, b: int) -> np.ndarray | None:
    if kp[a, 3] < VISIBILITY_THRESHOLD or kp[b, 3] < VISIBILITY_THRESHOLD:
        return None
    return (kp[a, :2] + kp[b, :2]) / 2.0


def _trunk_centroid(kp: np.ndarray) -> np.ndarray | None:
    """Trunk centroid from shoulders+hips. Uses 3 landmarks if 1 is low-vis."""
    landmarks = [L_SHOULDER, R_SHOULDER, L_HIP, R_HIP]
    visible = [kp[i, :2].copy() for i in landmarks
               if kp[i, 3] >= VISIBILITY_THRESHOLD]
    if len(visible) < 3:
        return None
    return np.mean(visible, axis=0)


def compute_com(kp: np.ndarray) -> tuple[float, float] | None:
    """Return (com_x, com_y) in normalized image coords, or None if insufficient data.

    kp shape: (33, 4) with columns x, y, z, visibility.
    """
    segments: list[tuple[np.ndarray | None, float]] = [
        (_point_if_visible(kp, NOSE), 0.081),
        (_trunk_centroid(kp), 0.497),
        (_midpoint_if_both_visible(kp, L_SHOULDER, L_ELBOW), 0.028),
        (_midpoint_if_both_visible(kp, R_SHOULDER, R_ELBOW), 0.028),
        (_midpoint_if_both_visible(kp, L_ELBOW, L_WRIST), 0.016),
        (_midpoint_if_both_visible(kp, R_ELBOW, R_WRIST), 0.016),
        (_midpoint_if_both_visible(kp, L_HIP, L_KNEE), 0.100),
        (_midpoint_if_both_visible(kp, R_HIP, R_KNEE), 0.100),
        (_midpoint_if_both_visible(kp, L_KNEE, L_ANKLE), 0.047),
        (_midpoint_if_both_visible(kp, R_KNEE, R_ANKLE), 0.047),
        (_midpoint_if_both_visible(kp, L_ANKLE, L_FOOT), 0.014),
        (_midpoint_if_both_visible(kp, R_ANKLE, R_FOOT), 0.014),
    ]

    total_mass = 0.0
    weighted = np.zeros(2, dtype=np.float64)
    for pos, w in segments:
        if pos is None:
            continue
        weighted += pos * w
        total_mass += w

    if total_mass < _MIN_PRESENT_MASS:
        return None
    com = weighted / total_mass
    return float(com[0]), float(com[1])