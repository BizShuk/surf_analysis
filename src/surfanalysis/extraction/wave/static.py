"""MOG2 background-subtraction wave engine (fixed-camera wave pool)."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import EngineInfo, WaveMetrics
from surfanalysis.extraction.wave.base import WaveEngine, to_wave_metrics
from surfanalysis.extraction.wave.region import region_from_mask

_KERNEL = np.ones((5, 5), np.uint8)


class Mog2WaveEngine(WaveEngine):
    def __init__(self, view: str, min_confidence: float = 0.5, warmup: int = 10) -> None:
        self._view = view
        self._min_conf = min_confidence
        self._warmup = warmup
        self._seen = 0
        self._bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None:  # noqa: ARG002
        fg = self._bg.apply(frame)
        self._seen += 1
        if self._seen <= self._warmup:
            return None
        _thr, binary = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL)
        obs = region_from_mask(binary)
        if obs is None or obs.confidence < self._min_conf:
            return None
        return to_wave_metrics(obs, self._view)

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="wave-static",
            version="0.1.0",
            params={
                "view": self._view,
                "min_confidence": self._min_conf,
                "warmup": self._warmup,
            },
        )
