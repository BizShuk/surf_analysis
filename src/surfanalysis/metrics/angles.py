"""Joint and trunk angle computations."""

from __future__ import annotations

import math

import numpy as np

from surfanalysis.extraction.landmarks import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    VISIBILITY_THRESHOLD,
)
from surfanalysis.metrics.geometry import angle_at_vertex, midpoint, wrap_to_180


def _angle3(kp: np.ndarray, a: int, b: int, c: int) -> float | None:
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in (a, b, c)):
        return None
    val = angle_at_vertex(kp[a, :2], kp[b, :2], kp[c, :2])
    if math.isnan(val):
        return None
    return val


def compute_knee_angles(kp: np.ndarray) -> tuple[float | None, float | None]:
    left = _angle3(kp, L_HIP, L_KNEE, L_ANKLE)
    right = _angle3(kp, R_HIP, R_KNEE, R_ANKLE)
    return left, right


def compute_elbow_angles(kp: np.ndarray) -> tuple[float | None, float | None]:
    left = _angle3(kp, L_SHOULDER, L_ELBOW, L_WRIST)
    right = _angle3(kp, R_SHOULDER, R_ELBOW, R_WRIST)
    return left, right


def compute_torso_lean(kp: np.ndarray) -> float | None:
    """Angle of trunk vector (mid_hip → mid_shoulder) relative to image-up.

    Positive: shoulders shifted in +x relative to hips (rightward lean in image).
    Negative: leftward.
    """
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in needed):
        return None
    mid_sh = midpoint(kp[L_SHOULDER, :2], kp[R_SHOULDER, :2])
    mid_hp = midpoint(kp[L_HIP, :2], kp[R_HIP, :2])
    trunk = mid_sh - mid_hp
    return float(math.degrees(math.atan2(trunk[0], -trunk[1])))


def compute_shoulder_hip_diff(kp: np.ndarray) -> float | None:
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in needed):
        return None
    sh_angle = math.degrees(math.atan2(
        kp[R_SHOULDER, 1] - kp[L_SHOULDER, 1],
        kp[R_SHOULDER, 0] - kp[L_SHOULDER, 0],
    ))
    hp_angle = math.degrees(math.atan2(
        kp[R_HIP, 1] - kp[L_HIP, 1],
        kp[R_HIP, 0] - kp[L_HIP, 0],
    ))
    return wrap_to_180(sh_angle - hp_angle)