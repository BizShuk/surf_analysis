"""Extract wave geometry (crest/base/lines/bbox) from a binary mask."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.wave.base import Line, Point, WaveObservation


def _fit_line(pts: np.ndarray, w: int, h: int) -> Line:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    xs = pts[:, 0]
    xmin, xmax = float(xs.min()), float(xs.max())

    def at(xq: float) -> Point:
        t = (xq - float(x0)) / float(vx) if abs(float(vx)) > 1e-6 else 0.0
        return (xq / w, (float(y0) + t * float(vy)) / h)

    return (at(xmin), at(xmax))


def region_from_mask(
    mask: np.ndarray, min_area_frac: float = 0.02, horizon_deg: float | None = None
) -> WaveObservation | None:
    """Largest contour -> crest/base/crest_line/face_line/bbox/confidence.

    Returns None when no contour passes the minimum-area gate (this also
    filters out the surfer, who is a much smaller blob than the wave body).
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area_frac * w * h:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    pts = c.reshape(-1, 2)
    crest = (float(pts[pts[:, 1].argmin(), 0]) / w, float(pts[:, 1].min()) / h)
    base = (float(pts[pts[:, 1].argmax(), 0]) / w, float(pts[:, 1].max()) / h)
    top_band = pts[pts[:, 1] <= y + 0.2 * bh]
    crest_line = (
        _fit_line(top_band, w, h)
        if len(top_band) >= 2
        else ((x / w, y / h), ((x + bw) / w, y / h))
    )
    face_line = (base, crest)
    bbox = (x / w, y / h, bw / w, bh / h)
    confidence = min(1.0, area / (0.25 * w * h))
    return WaveObservation(crest, base, crest_line, face_line, bbox, confidence, horizon_deg)
