"""Wave engine package: factory for the Strategy implementations."""

from __future__ import annotations

from surfanalysis.extraction.wave.base import WaveEngine
from surfanalysis.extraction.wave.ocean import HorizonAnchoredWaveEngine
from surfanalysis.extraction.wave.static import Mog2WaveEngine

__all__ = ["WaveEngine", "make_wave_engine"]


def make_wave_engine(name: str, view: str, min_confidence: float = 0.5) -> WaveEngine:
    if name == "ocean":
        return HorizonAnchoredWaveEngine(view, min_confidence)
    if name == "static":
        return Mog2WaveEngine(view, min_confidence)
    raise ValueError(f"unknown wave engine: {name}")
