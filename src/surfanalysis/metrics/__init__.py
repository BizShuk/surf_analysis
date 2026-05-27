"""Public entry point for per-frame metric computation."""

from __future__ import annotations

from typing import Literal

import numpy as np

from surfanalysis.extraction.schema import FrameMetrics
from surfanalysis.metrics.angles import (
    compute_elbow_angles,
    compute_knee_angles,
    compute_shoulder_hip_diff,
    compute_torso_lean,
)
from surfanalysis.metrics.com import compute_com
from surfanalysis.metrics.stability import StabilityWindow
from surfanalysis.metrics.weight_dist import compute_weight_dist_front_pct

Stance = Literal["regular", "goofy"]


def compute_frame_metrics(
    kp: np.ndarray,
    stance: Stance,
    stability_window: StabilityWindow,
) -> FrameMetrics | None:
    com = compute_com(kp)
    if com is None:
        stability_window.push(None)
        return None
    weight_pct = compute_weight_dist_front_pct(kp, com, stance)
    if weight_pct is None:
        stability_window.push(None)
        return None

    stability_window.push(com)
    knee_l, knee_r = compute_knee_angles(kp)
    elbow_l, elbow_r = compute_elbow_angles(kp)
    return FrameMetrics(
        com=com,
        weight_dist_front_pct=weight_pct,
        knee_angle_left=knee_l,
        knee_angle_right=knee_r,
        elbow_angle_left=elbow_l,
        elbow_angle_right=elbow_r,
        torso_lean_deg=compute_torso_lean(kp),
        shoulder_hip_rotation_deg=compute_shoulder_hip_diff(kp),
        com_stability_score=stability_window.score(),
    )


__all__ = ["compute_frame_metrics", "StabilityWindow"]
