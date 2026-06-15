"""Pure geometry + aggregation for wave metrics.

mypy-strict module: fully annotated, no Pydantic / no I/O.
"""

from __future__ import annotations

import math
from statistics import median

from surfanalysis.metrics.geometry import wrap_to_180

Point = tuple[float, float]
Line = tuple[Point, Point]


def line_angle_deg(p1: Point, p2: Point) -> float:
    """Angle of the line p1->p2 vs image-horizontal, normalized to (-90, 90].

    Image y grows downward, so a line going down-to-the-right is positive.
    """
    ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return ang


def angle_vs_horizon_deg(line: Line, horizon_deg: float) -> float:
    """Line tilt relative to the detected horizon (absorbs camera roll)."""
    return wrap_to_180(line_angle_deg(line[0], line[1]) - horizon_deg)


def normalized_height(top: Point, bottom: Point) -> float:
    """Vertical extent between two normalized points (already 0-1)."""
    return abs(top[1] - bottom[1])


def classify_view(bbox_w: float, bbox_h: float, crest_line: Line) -> str:
    """Return "facing" or "side" from wave-region shape + top-edge tilt.

    Heuristic (tunable): a facing wave fills the frame width with a gently
    tilted top edge; a side/profile wave shows a steeply tilted face line.
    """
    tilt = abs(line_angle_deg(crest_line[0], crest_line[1]))
    aspect = bbox_w / bbox_h if bbox_h > 0.0 else 0.0
    if tilt < 30.0 and aspect >= 1.0:
        return "facing"
    return "side"


def median_p90(values: list[float]) -> tuple[float, float]:
    """Return (median, 90th-percentile) of values; (0.0, 0.0) if empty."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    med = float(median(ordered))
    idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return (med, float(ordered[idx]))
