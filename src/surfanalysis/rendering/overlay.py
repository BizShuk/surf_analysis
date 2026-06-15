"""Draw skeleton, center-of-mass marker, and metric text onto frames."""

from __future__ import annotations

import math

import cv2
import numpy as np

from surfanalysis.extraction.landmarks import (
    L_FOOT,
    L_HIP,
    L_SHOULDER,
    R_FOOT,
    R_HIP,
    R_SHOULDER,
)
from surfanalysis.extraction.schema import FrameRecord, Stance
from surfanalysis.rendering.skeleton import SKELETON_EDGES

VISIBILITY_DRAW = 0.5
TRUNK_EXTEND = 1.25  # extend trunk line past shoulders for readability
REF_LINE_COLOR = (200, 200, 200)  # dashed vertical reference (BGR)
DASH_LEN = 8


def hex_to_bgr(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


def _dashed_line(
    frame: np.ndarray,
    p1: tuple[int, int],
    p2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if length < 1.0:
        return
    n = max(1, int(length // DASH_LEN))
    steps = np.linspace(p1, p2, n + 1)
    for i in range(0, n, 2):
        a = tuple(np.round(steps[i]).astype(int))
        b = tuple(np.round(steps[i + 1]).astype(int))
        cv2.line(frame, a, b, color, thickness)


class OverlayRenderer:
    def __init__(
        self,
        skeleton_color: str = "#00FF00",
        com_color: str = "#FFFF00",
        text_color: str = "#FFFFFF",
        angle_color: str = "#FF40FF",
        weight_color: str = "#FFA500",
        font_scale: float = 0.6,
        show_secondary: bool = False,
        stance: Stance = "regular",
    ) -> None:
        self._sk = hex_to_bgr(skeleton_color)
        self._com = hex_to_bgr(com_color)
        self._tx = hex_to_bgr(text_color)
        self._angle = hex_to_bgr(angle_color)
        self._wt = hex_to_bgr(weight_color)
        self._font_scale = font_scale
        self._show_secondary = show_secondary
        self._stance: Stance = stance

    def _text(
        self,
        frame: np.ndarray,
        text: str,
        org: tuple[int, int],
        color: tuple[int, int, int] | None = None,
    ) -> None:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._font_scale, color or self._tx, 1, cv2.LINE_AA)

    def _draw_angle_line(
        self,
        frame: np.ndarray,
        pts: list[tuple[int, int, float]],
        lean_deg: float,
    ) -> None:
        """Trunk line (mid-hip -> mid-shoulder) vs dashed vertical reference."""
        needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
        if any(pts[i][2] < VISIBILITY_DRAW for i in needed):
            return
        mid_sh = ((pts[L_SHOULDER][0] + pts[R_SHOULDER][0]) // 2,
                  (pts[L_SHOULDER][1] + pts[R_SHOULDER][1]) // 2)
        mid_hp = ((pts[L_HIP][0] + pts[R_HIP][0]) // 2,
                  (pts[L_HIP][1] + pts[R_HIP][1]) // 2)
        trunk_len = math.hypot(mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
        if trunk_len < 1.0:
            return
        ref_end = (mid_hp[0], int(mid_hp[1] - trunk_len * TRUNK_EXTEND))
        _dashed_line(frame, mid_hp, ref_end, REF_LINE_COLOR)
        ext = (int(mid_hp[0] + (mid_sh[0] - mid_hp[0]) * TRUNK_EXTEND),
               int(mid_hp[1] + (mid_sh[1] - mid_hp[1]) * TRUNK_EXTEND))
        cv2.line(frame, mid_hp, ext, self._angle, 3)
        label_y = max(15, ext[1] - 8)
        self._text(frame, f"{lean_deg:+.1f} deg", (ext[0] + 8, label_y), self._angle)

    def _draw_weight_line(
        self,
        frame: np.ndarray,
        pts: list[tuple[int, int, float]],
        com_px: tuple[int, int],
        front_pct: float,
    ) -> None:
        """Back-foot -> front-foot baseline with CoM projection marker."""
        if self._stance == "regular":
            front_idx, back_idx = L_FOOT, R_FOOT
        else:
            front_idx, back_idx = R_FOOT, L_FOOT
        fx, fy, fv = pts[front_idx]
        bx, by, bv = pts[back_idx]
        if fv < VISIBILITY_DRAW or bv < VISIBILITY_DRAW:
            return
        cv2.line(frame, (bx, by), (fx, fy), self._wt, 3)
        cv2.circle(frame, (bx, by), 5, self._wt, -1)
        cv2.circle(frame, (fx, fy), 5, self._wt, -1)
        t = front_pct / 100.0
        mx = int(bx + (fx - bx) * t)
        my = int(by + (fy - by) * t)
        _dashed_line(frame, com_px, (mx, my), self._com)
        cv2.circle(frame, (mx, my), 7, self._com, -1)
        cv2.circle(frame, (mx, my), 8, (0, 0, 0), 1)
        self._text(frame, f"F {front_pct:.0f}%", (fx - 30, fy + 24), self._wt)
        self._text(frame, f"B {100 - front_pct:.0f}%", (bx - 30, by + 24), self._wt)

    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
        if record.keypoints is None:
            return frame
        h, w = frame.shape[:2]
        pts = [(int(p[0] * w), int(p[1] * h), p[3]) for p in record.keypoints.points]

        # skeleton — drawn whenever a pose is detected. Each edge has its own
        # visibility guard, so this is independent of whether the derived
        # metrics (CoM / weight distribution) could be computed this frame.
        # Keeping it decoupled stops the overlay vanishing on frames where, e.g.,
        # a foot drops below the visibility threshold but the body is tracked.
        for a, b in SKELETON_EDGES:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < VISIBILITY_DRAW or vb < VISIBILITY_DRAW:
                continue
            cv2.line(frame, (xa, ya), (xb, yb), self._sk, 2)

        # Metric-dependent overlays (CoM marker, weight line, lean line, text)
        # only render on frames that produced a full FrameMetrics.
        if record.metrics is None:
            return frame

        cx = int(record.metrics.com[0] * w)
        cy = int(record.metrics.com[1] * h)

        # weight distribution baseline (before com marker so the marker sits on top)
        self._draw_weight_line(frame, pts, (cx, cy),
                               record.metrics.weight_dist_front_pct)

        # torso lean angle line
        if record.metrics.torso_lean_deg is not None:
            self._draw_angle_line(frame, pts, record.metrics.torso_lean_deg)

        # com marker
        cv2.circle(frame, (cx, cy), 8, self._com, -1)
        cv2.circle(frame, (cx, cy), 9, (0, 0, 0), 1)

        # primary text block (top-left)
        weight_f = record.metrics.weight_dist_front_pct
        lines = [f"F:{weight_f:.0f}%  B:{100 - weight_f:.0f}%"]
        if record.metrics.torso_lean_deg is not None:
            lines.append(f"lean: {record.metrics.torso_lean_deg:+.1f} deg")
        if record.metrics.knee_angle_left is not None:
            lines.append(f"L knee: {record.metrics.knee_angle_left:.0f} deg")
        if record.metrics.knee_angle_right is not None:
            lines.append(f"R knee: {record.metrics.knee_angle_right:.0f} deg")

        if self._show_secondary:
            if record.metrics.elbow_angle_left is not None:
                lines.append(f"L elbow: {record.metrics.elbow_angle_left:.0f} deg")
            if record.metrics.elbow_angle_right is not None:
                lines.append(f"R elbow: {record.metrics.elbow_angle_right:.0f} deg")
            if record.metrics.shoulder_hip_rotation_deg is not None:
                lines.append(f"sh-hip rot: {record.metrics.shoulder_hip_rotation_deg:+.1f} deg")
            if record.metrics.com_stability_score is not None:
                lines.append(f"stability: {record.metrics.com_stability_score:.2f}")

        for i, text in enumerate(lines):
            y = int(33 * self._font_scale) + i * int(20 * self._font_scale * 1.6)
            self._text(frame, text, (12, y))
        return frame
