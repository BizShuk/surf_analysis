"""Horizon-anchored per-frame wave engine (ocean / moving camera)."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import EngineInfo, WaveMetrics
from surfanalysis.extraction.wave.base import WaveEngine, to_wave_metrics
from surfanalysis.extraction.wave.horizon import detect_horizon
from surfanalysis.extraction.wave.region import region_from_mask

_KERNEL = np.ones((5, 5), np.uint8)


def wave_mask(frame: np.ndarray, horizon_deg: float | None) -> np.ndarray:  # noqa: ARG001
    """Binary mask of foam (bright, low-sat) + blue-green water face."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    foam = (vv > 200) & (ss < 60)
    bluegreen = (hh > 80) & (hh < 140) & (ss > 40)
    mask = ((foam | bluegreen).astype(np.uint8)) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)


class HorizonAnchoredWaveEngine(WaveEngine):
    def __init__(self, view: str, min_confidence: float = 0.5) -> None:
        self._view = view
        self._min_conf = min_confidence

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None:  # noqa: ARG002
        horizon_deg = detect_horizon(frame)
        mask = wave_mask(frame, horizon_deg)
        obs = region_from_mask(mask, horizon_deg=horizon_deg)
        if obs is None or obs.confidence < self._min_conf:
            return None
        return to_wave_metrics(obs, self._view)

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="wave-ocean",
            version="0.1.0",
            params={"view": self._view, "min_confidence": self._min_conf},
        )
