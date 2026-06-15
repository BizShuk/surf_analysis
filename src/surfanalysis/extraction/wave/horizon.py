"""Sea-sky / dominant horizontal line detection -> tilt degrees."""

from __future__ import annotations

import math

import cv2
import numpy as np

from surfanalysis.metrics.geometry import wrap_to_180

_MAX_HORIZON_TILT = 25.0  # only accept near-horizontal lines as a horizon


def detect_horizon(frame: np.ndarray) -> float | None:
    """Return the horizon tilt in degrees (vs image-horizontal), or None.

    Picks the longest near-horizontal Hough segment. None means no horizon
    found; callers should then assume image-horizontal (0 deg).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_len = frame.shape[1] // 3
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=min_len, maxLineGap=20
    )
    if lines is None:
        return None
    best_angle: float | None = None
    best_len = 0.0
    for x1, y1, x2, y2 in lines[:, 0, :]:
        angle = wrap_to_180(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))))
        if abs(angle) > _MAX_HORIZON_TILT:
            continue
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length > best_len:
            best_len = length
            best_angle = angle
    return best_angle
