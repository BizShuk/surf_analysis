"""Global camera-motion magnitude between two grayscale frames."""

from __future__ import annotations

import cv2
import numpy as np


def global_motion(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    """Return the global translation magnitude (px) via phase correlation."""
    a = np.float32(prev_gray)
    b = np.float32(cur_gray)
    (dx, dy), _response = cv2.phaseCorrelate(a, b)
    return float((dx * dx + dy * dy) ** 0.5)
