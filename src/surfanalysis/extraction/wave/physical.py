"""PhysicalWaveComputer — derive wave height in meters from per-frame observations.

Two cooperating paths:
1. Camera geometry (baseline): pinhole projection from crest/base image
   points to world coordinates using `wave_height_meters()`. Requires a
   CameraModel (camera_height_m + focal/sensor metadata).
2. Wavelength / dispersion (sanity check): deep-water gravity wave dispersion
   `L = g T² / (2π)` plus the Miche breaking-wave criterion `H/L ∈ [1/10, 1/7]`
   to bracket the expected height. Independent of camera intrinsics but
   currently a stub: T comes from crest passage timing in the ocean engine,
   which isn't plumbed in yet.

`score_confidence` decides `confidence` from the delta between the two paths.
When path 2 is unavailable, path 1 alone is `medium` (single-source, not
independently verified).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from surfanalysis.extraction.schema import PhysicalWaveFrame, WaveMetrics
from surfanalysis.extraction.wave.camera import CameraModel
from surfanalysis.metrics.wave_geometry import (
    normalized_to_world_height,
    wave_height_meters,
)

G = 9.81  # gravity, m/s²


@dataclass(frozen=True)
class WavelengthEstimate:
    """Wavelength dispersion result for a single observation window."""

    wavelength_m: float
    period_s: float
    h_upper_m: float          # L / 7 (Miche upper bound)
    h_lower_m: float          # L / 10 (Miche lower bound)
    confidence: Literal["high", "medium", "low", "unavailable"] = "medium"


class WavelengthEstimator:
    """Derive wave height bounds from deep-water dispersion (g T²/(2π)).

    `estimate_break_height()` is a stub for now — it would derive T from the
    crest passage timing observed by the ocean engine, which isn't wired in.
    Returns None until that integration lands.
    """

    def from_period_s(self, period_s: float) -> WavelengthEstimate | None:
        if period_s <= 0:
            return None
        L = G * period_s * period_s / (2 * math.pi)
        return WavelengthEstimate(
            wavelength_m=L,
            period_s=period_s,
            h_upper_m=L / 7.0,
            h_lower_m=L / 10.0,
        )

    def estimate_break_height(self) -> float | None:
        """Return an estimated breaking wave height in meters, or None.

        Stub: the real implementation pulls T from the ocean engine's crest
        passage timing (Phase 4 follow-up). Until then we return None, which
        makes `score_confidence` fall back to `medium` (single-source).
        """
        return None


def score_confidence(
    h_camera: float | None, h_wave: float | None
) -> Literal["high", "medium", "low", "unavailable"]:
    """Decide confidence from camera-geometry vs wavelength break-height.

    Rules (per design 2026-06-21):
    - No camera-geometry result at all → `unavailable`
    - Single source (no wavelength) → `medium`
    - Both sources, |delta|/h_wave ≤ 0.20 → `high`
    - Both sources, 0.20 < |delta|/h_wave ≤ 0.50 → `medium`
    - Both sources, |delta|/h_wave > 0.50 → `low`
    """
    if h_camera is None:
        return "unavailable"
    if h_wave is None:
        return "medium"
    if h_wave <= 0:
        return "low"
    delta = abs(h_camera - h_wave) / h_wave
    if delta <= 0.20:
        return "high"
    if delta <= 0.50:
        return "medium"
    return "low"


class PhysicalWaveComputer:
    """Per-frame physical-wave metric computer.

    Stateless; safe to share across frames. Accepts a CameraModel (None means
    "no metadata, skip physics") and an optional WavelengthEstimator.

    The constructor's `horizon_deg` is a *fallback* roll (camera rotation
    about optical axis); the per-frame `metrics.horizon_deg` takes precedence
    when available. Note: `horizon_deg` is **roll**, not pitch — the codebase's
    `horizon.py` returns the slope of the horizon line in the image, which
    is the camera's rotation about the optical axis. Pitch (optical-axis
    tilt below horizontal) is a separate user-supplied quantity on CameraModel.
    """

    def __init__(
        self,
        camera_model: CameraModel | None,
        horizon_deg: float = 0.0,
        wavelength_estimator: WavelengthEstimator | None = None,
    ) -> None:
        self._camera_model = camera_model
        self._horizon_deg = horizon_deg
        self._wavelength_estimator = wavelength_estimator or WavelengthEstimator()

    def compute(self, metrics: WaveMetrics) -> PhysicalWaveFrame:
        """Compute physical height from a per-frame WaveMetrics.

        Pulls `height_top` (crest), `height_bottom` (base), and `horizon_deg`
        from the metrics — these are populated by `to_wave_metrics` and don't
        require re-exposing WaveObservation. Returns a `skipped` frame if no
        CameraModel was provided.
        """
        if self._camera_model is None:
            return PhysicalWaveFrame(
                method="skipped",
                confidence="unavailable",
                reason="insufficient_metadata: provide --camera-height-m",
            )
        # per-frame horizon_deg is more accurate than the constructor fallback
        roll = metrics.horizon_deg if metrics.horizon_deg is not None else self._horizon_deg
        intr = self._camera_model.to_intrinsics(roll_deg=roll)
        try:
            crest_world = normalized_to_world_height(metrics.height_top, intr)
            base_world = normalized_to_world_height(metrics.height_bottom, intr)
            h_camera = wave_height_meters(metrics.height_top, metrics.height_bottom, intr)
        except ValueError as e:
            return PhysicalWaveFrame(
                method="camera_geometry",
                confidence="low",
                reason=f"projection failed: {e}",
            )
        h_wave = self._wavelength_estimator.estimate_break_height()
        confidence = score_confidence(h_camera, h_wave)
        return PhysicalWaveFrame(
            crest_world=crest_world,
            trough_world=base_world,
            height_m=h_camera,
            method="cross_validated" if h_wave is not None else "camera_geometry",
            confidence=confidence,
        )
