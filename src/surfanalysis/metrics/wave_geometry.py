"""Pure geometry + aggregation for wave metrics.

mypy-strict module: fully annotated, no Pydantic / no I/O.
"""

from __future__ import annotations

import math
from statistics import median
from typing import NamedTuple

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


class CameraIntrinsics(NamedTuple):
    """Pinhole camera geometry for projecting image points to world (water plane).

    `focal_pixels` = image_height_px / (2 * tan(fov_half)), or equivalently
    derived from EXIF focal_length_mm and sensor_height_mm at the call site.
    `pitch_deg` is camera tilt; positive = looking down.
    `roll_deg` is camera roll about the optical axis; positive = CCW.
    """
    camera_height_m: float
    focal_pixels: float
    image_height_px: int
    pitch_deg: float
    roll_deg: float = 0.0


def _alpha_rad(y_norm: float, intr: CameraIntrinsics) -> float:
    """Angle from optical axis to pixel y (rad); positive = below axis."""
    if intr.focal_pixels <= 0 or intr.image_height_px <= 0:
        raise ValueError("focal_pixels and image_height_px must be positive")
    cy_px = intr.image_height_px / 2.0
    y_px = y_norm * intr.image_height_px
    return math.atan2(y_px - cy_px, intr.focal_pixels)


def normalized_to_world_height(
    point_norm: Point, intr: CameraIntrinsics
) -> tuple[float, float, float]:
    """World (X, Y, Z) in meters of the water-plane intersection of the ray.

    Treats the point as lying on the water surface (Y=0). Useful for ground-
    referenced anchors (e.g., the base of a wave). Lateral X is 0 (single-view;
    no baseline to recover depth-perpendicular position).

    Requires the point to be at or below the horizon (alpha + pitch in (0, 90°));
    raises ValueError for points above the horizon or with degenerate intrinsics.
    """
    if intr.camera_height_m <= 0:
        raise ValueError("camera_height_m must be > 0")
    alpha = _alpha_rad(point_norm[1], intr)
    theta = math.radians(intr.pitch_deg)
    total = theta + alpha
    if total <= 0.0:
        raise ValueError("point above horizon; no water-plane intersection")
    denom = math.tan(total)
    if abs(denom) < 1e-9:
        raise ValueError("pitch + alpha near 90°, projection undefined")
    z = intr.camera_height_m / denom
    return (0.0, 0.0, z)


def wave_height_meters(
    crest: Point, base: Point, intr: CameraIntrinsics
) -> float:
    """Wave height in meters from crest/base image points.

    Assumes base is on the water surface (Y=0) and the wave is thin enough
    that crest and base share the same depth. Derivation:

        Z_base = H / tan(theta + alpha_base)
        Y_crest = H - Z_base * tan(theta + alpha_crest)
                = H * (1 - tan(theta + alpha_crest) / tan(theta + alpha_base))

    where `theta` is the camera pitch and `alpha_*` are angles from the optical
    axis. Returns |Y_crest - Y_base| = |Y_crest| since Y_base = 0.

    Note: at pitch=0 this collapses to the simpler
    `H * (1 - tan(alpha_crest) / tan(alpha_base))` (the previous form),
    because tan(0 + alpha) = tan(alpha).
    """
    if intr.camera_height_m <= 0:
        raise ValueError("camera_height_m must be > 0")
    theta = math.radians(intr.pitch_deg)
    alpha_crest = theta + _alpha_rad(crest[1], intr)
    alpha_base = theta + _alpha_rad(base[1], intr)
    tan_crest = math.tan(alpha_crest)
    tan_base = math.tan(alpha_base)
    if tan_base <= 0:
        raise ValueError("base must be at or below horizon")
    return abs(intr.camera_height_m * (1.0 - tan_crest / tan_base))
