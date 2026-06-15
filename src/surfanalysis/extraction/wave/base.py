"""WaveEngine Strategy base + raw observation + metric conversion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from surfanalysis.extraction.schema import EngineInfo, WaveMetrics
from surfanalysis.metrics.wave_geometry import angle_vs_horizon_deg, normalized_height

Point = tuple[float, float]
Line = tuple[Point, Point]


@dataclass
class WaveObservation:
    """Raw, view-agnostic pixel geometry extracted from one frame (normalized)."""

    crest: Point
    base: Point
    crest_line: Line
    face_line: Line
    bbox: tuple[float, float, float, float]  # x, y, w, h (normalized)
    confidence: float
    horizon_deg: float | None


def to_wave_metrics(obs: WaveObservation, view: str) -> WaveMetrics:
    if view == "facing":
        angle_line = obs.crest_line
        angle_kind = "crest_tilt"
    else:
        angle_line = obs.face_line
        angle_kind = "face_steepness"
    angle = angle_vs_horizon_deg(angle_line, obs.horizon_deg or 0.0)
    return WaveMetrics(
        view=view,
        height=normalized_height(obs.crest, obs.base),
        angle_deg=angle,
        angle_kind=angle_kind,
        confidence=obs.confidence,
        angle_line=angle_line,
        height_top=obs.crest,
        height_bottom=obs.base,
        horizon_deg=obs.horizon_deg,
    )


class WaveEngine(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None:
        """Return wave metrics for a BGR frame, or None when no wave is found."""

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return engine metadata for the JSON output."""

    def close(self) -> None:
        """Release engine resources. Default no-op."""
        return None


class MockWaveEngine(WaveEngine):
    """Test double: replays a fixed sequence of WaveMetrics | None."""

    def __init__(self, sequence: list[WaveMetrics | None]) -> None:
        self._sequence = sequence
        self._cursor = 0

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None:  # noqa: ARG002
        if self._cursor >= len(self._sequence):
            return None
        out = self._sequence[self._cursor]
        self._cursor += 1
        return out

    def info(self) -> EngineInfo:
        return EngineInfo(name="mock-wave", version="test", params={})
