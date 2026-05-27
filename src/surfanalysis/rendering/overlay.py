"""Draw skeleton, center-of-mass marker, and metric text onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import FrameRecord
from surfanalysis.rendering.skeleton import SKELETON_EDGES

VISIBILITY_DRAW = 0.5


def hex_to_bgr(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


class OverlayRenderer:
    def __init__(
        self,
        skeleton_color: str = "#00FF00",
        com_color: str = "#FFFF00",
        text_color: str = "#FFFFFF",
        font_scale: float = 0.6,
        show_secondary: bool = False,
    ) -> None:
        self._sk = hex_to_bgr(skeleton_color)
        self._com = hex_to_bgr(com_color)
        self._tx = hex_to_bgr(text_color)
        self._font_scale = font_scale
        self._show_secondary = show_secondary

    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
        if record.keypoints is None or record.metrics is None:
            return frame
        h, w = frame.shape[:2]
        pts = [(int(p[0] * w), int(p[1] * h), p[3]) for p in record.keypoints.points]

        # skeleton
        for a, b in SKELETON_EDGES:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < VISIBILITY_DRAW or vb < VISIBILITY_DRAW:
                continue
            cv2.line(frame, (xa, ya), (xb, yb), self._sk, 2)

        # com marker
        cx = int(record.metrics.com[0] * w)
        cy = int(record.metrics.com[1] * h)
        cv2.circle(frame, (cx, cy), 8, self._com, -1)
        cv2.circle(frame, (cx, cy), 9, (0, 0, 0), 1)

        # primary text block (top-left)
        weight_f = record.metrics.weight_dist_front_pct
        lines = [f"F:{weight_f:.0f}%  B:{100 - weight_f:.0f}%"]
        if record.metrics.knee_angle_left is not None:
            lines.append(f"L knee: {record.metrics.knee_angle_left:.0f} deg")
        if record.metrics.knee_angle_right is not None:
            lines.append(f"R knee: {record.metrics.knee_angle_right:.0f} deg")

        if self._show_secondary:
            if record.metrics.elbow_angle_left is not None:
                lines.append(f"L elbow: {record.metrics.elbow_angle_left:.0f} deg")
            if record.metrics.elbow_angle_right is not None:
                lines.append(f"R elbow: {record.metrics.elbow_angle_right:.0f} deg")
            if record.metrics.torso_lean_deg is not None:
                lines.append(f"torso lean: {record.metrics.torso_lean_deg:+.1f} deg")
            if record.metrics.shoulder_hip_rotation_deg is not None:
                lines.append(f"sh-hip rot: {record.metrics.shoulder_hip_rotation_deg:+.1f} deg")
            if record.metrics.com_stability_score is not None:
                lines.append(f"stability: {record.metrics.com_stability_score:.2f}")

        for i, text in enumerate(lines):
            y = 20 + i * int(20 * self._font_scale * 1.6)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        self._font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        self._font_scale, self._tx, 1, cv2.LINE_AA)
        return frame
