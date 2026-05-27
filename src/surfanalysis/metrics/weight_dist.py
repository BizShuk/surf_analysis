"""Estimate front/back foot weight distribution from CoM projection."""

from __future__ import annotations

from typing import Literal

import numpy as np

from surfanalysis.extraction.landmarks import L_FOOT, R_FOOT, VISIBILITY_THRESHOLD
from surfanalysis.metrics.geometry import project_onto_segment

Stance = Literal["regular", "goofy"]


def compute_weight_dist_front_pct(
    kp: np.ndarray,
    com: tuple[float, float],
    stance: Stance,
) -> float | None:
    if kp[L_FOOT, 3] < VISIBILITY_THRESHOLD or kp[R_FOOT, 3] < VISIBILITY_THRESHOLD:
        return None
    if stance == "regular":
        front_idx, back_idx = L_FOOT, R_FOOT
    else:
        front_idx, back_idx = R_FOOT, L_FOOT
    front = kp[front_idx, :2]
    back = kp[back_idx, :2]
    com_np = np.asarray(com, dtype=np.float64)
    # Project: t=0 means at back, t=1 means at front. front_pct = t * 100.
    t = project_onto_segment(com_np, back, front)
    return float(t * 100.0)
