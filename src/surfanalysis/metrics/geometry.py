"""Pure geometric primitives operating on 2D numpy points."""

from __future__ import annotations

import math

import numpy as np


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    mid: np.ndarray = (a + b) / 2.0
    return mid


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def project_onto_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    denom = float(np.dot(v, v))
    if denom == 0.0:
        return 0.0
    t = float(np.dot(p - a, v) / denom)
    return max(0.0, min(1.0, t))


def wrap_to_180(angle_deg: float) -> float:
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a if a != -180.0 else 180.0
