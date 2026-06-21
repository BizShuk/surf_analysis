"""Iterate video frames, invoke PoseEngine, assemble SessionRecord."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from surfanalysis.extraction.engine import PoseEngine
from surfanalysis.extraction.schema import (
    SCHEMA_VERSION,
    EngineInfo,
    FrameRecord,
    Keypoints,
    SessionRecord,
    SessionSummary,
    SourceInfo,
    WaveSummary,
)
from surfanalysis.extraction.wave.base import WaveEngine
from surfanalysis.extraction.wave.camera import CameraModel
from surfanalysis.extraction.wave.physical import PhysicalWaveComputer
from surfanalysis.extraction.wave.prescan import prescan_physical
from surfanalysis.metrics import StabilityWindow, compute_frame_metrics
from surfanalysis.metrics.wave_geometry import median_p90

Stance = Literal["regular", "goofy"]

SCHEMA_VERSION_BASE = "1.0"
SCHEMA_VERSION_WAVE = SCHEMA_VERSION


def _kp_to_array(kp: Keypoints) -> np.ndarray:
    return np.array(kp.points, dtype=np.float64)


_CONFIDENCE_RANK = {"high": 3, "medium": 2, "low": 1, "unavailable": 0}


def _aggregate_confidence(
    waves: list,  # list[WaveMetrics] (annotated loose to avoid schema import cycle)
) -> Literal["high", "medium", "low", "unavailable"]:
    """Best confidence across frames; `unavailable` only if every frame is."""
    best = "unavailable"
    best_rank = 0
    for w in waves:
        if w.physical is None:
            continue
        rank = _CONFIDENCE_RANK.get(w.physical.confidence, 0)
        if rank > best_rank:
            best_rank = rank
            best = w.physical.confidence
    return best


class FrameAnalyzer:
    def __init__(self, engine: PoseEngine, stance: Stance, source: SourceInfo,
                 wave_engine: WaveEngine | None = None,
                 camera_model: CameraModel | None = None) -> None:
        self._engine = engine
        self._stance = stance
        self._source = source
        self._wave_engine = wave_engine
        self._camera_model = camera_model
        # Always have a computer so per-frame physical is populated even when
        # metadata is missing (skipped frame with reason). Without this, the
        # schema would have a wave but no physical evidence for the absence.
        self._physical_computer: PhysicalWaveComputer = PhysicalWaveComputer(
            camera_model=camera_model,
        )

    def run(self, frames_iter: Iterable[np.ndarray]) -> SessionRecord:
        frames: list[FrameRecord] = []
        stability = StabilityWindow()
        detections = 0

        for idx, frame in enumerate(frames_iter):
            ts_ms = (idx / self._source.fps) * 1000.0 if self._source.fps > 0 else 0.0
            wave = self._wave_engine.detect(frame, ts_ms) if self._wave_engine else None
            # Populate physical height for every wave frame. When camera
            # metadata is missing, the computer returns a 'skipped' frame
            # with a reason — better than leaving physical=None which would
            # hide the cause from downstream consumers.
            if wave is not None:
                wave.physical = self._physical_computer.compute(wave)
            kp = self._engine.detect(frame, ts_ms)
            if kp is None:
                stability.push(None)
                frames.append(FrameRecord(frame_index=idx, timestamp_ms=ts_ms,
                                          keypoints=None, metrics=None, wave=wave))
                continue
            detections += 1
            metrics = compute_frame_metrics(_kp_to_array(kp), self._stance, stability)
            frames.append(FrameRecord(frame_index=idx, timestamp_ms=ts_ms,
                                      keypoints=kp, metrics=metrics, wave=wave))

        total = len(frames)
        rate = detections / total if total else 0.0
        version = SCHEMA_VERSION_WAVE if self._wave_engine else SCHEMA_VERSION_BASE
        wave_engine_info = self._wave_engine.info() if self._wave_engine else None
        wave_summary = (
            self._aggregate_wave(frames, wave_engine_info, self._camera_model)
            if self._wave_engine else None
        )
        return SessionRecord(
            schema_version=version,
            source=self._source,
            engine=self._engine.info(),
            stance=self._stance,
            frames=frames,
            summary=SessionSummary(
                frames_with_detection=detections,
                frames_total=total,
                detection_rate=rate,
                metrics_aggregate=_aggregate(frames),
            ),
            wave_engine=wave_engine_info,
            wave_summary=wave_summary,
        )

    @staticmethod
    def _aggregate_wave(
        frames: list[FrameRecord], engine_info: EngineInfo | None,
        camera_model: CameraModel | None = None,
    ) -> WaveSummary | None:
        waves = [f.wave for f in frames if f.wave is not None]
        if not waves:
            return None
        # Physical heights: only count frames where confidence is high or
        # medium. low / unavailable frames still get written per-frame for
        # diagnostics but shouldn't bias the session aggregate.
        h_ms: list[float] = [
            w.physical.height_m for w in waves
            if w.physical is not None
            and w.physical.height_m is not None
            and w.physical.confidence in ("high", "medium")
        ]
        h_med_m: float | None = None
        h_p90_m: float | None = None
        if h_ms:
            h_med_m, h_p90_m = median_p90(h_ms)
        a_med, _ = median_p90([w.angle_deg for w in waves])
        views = [w.view for w in waves]
        dom = max(set(views), key=views.count)
        view = dom if all(v == dom for v in views) else "mixed"
        engine_tag = engine_info.name.replace("wave-", "") if engine_info else "unknown"

        # physical_status: derived from prescan_physical (CLI decides the
        # view; for the default facing view, this is "computed" iff camera
        # model was provided). When the CLI doesn't pass camera_model, the
        # analyzer still produces waves (without physical), and status falls
        # back to "insufficient_metadata" since no frame can have physical.
        view = "facing" if view in ("facing", "mixed") else view
        physical_status = prescan_physical(camera_model, view)
        confidence = _aggregate_confidence(waves)

        return WaveSummary(
            frames_detected=len(waves),
            view=view,
            angle_median=a_med,
            engine=engine_tag,
            height_m_median=h_med_m,
            height_m_p90=h_p90_m,
            confidence=confidence,
            physical_status=physical_status,  # type: ignore[arg-type]
            camera=camera_model.schema if camera_model is not None else None,
        )


def _aggregate(frames: list[FrameRecord]) -> dict[str, float]:
    com_x: list[float] = []
    com_y: list[float] = []
    knee_l: list[float] = []
    knee_r: list[float] = []
    weight: list[float] = []
    for f in frames:
        if f.metrics is None:
            continue
        com_x.append(f.metrics.com[0])
        com_y.append(f.metrics.com[1])
        weight.append(f.metrics.weight_dist_front_pct)
        if f.metrics.knee_angle_left is not None:
            knee_l.append(f.metrics.knee_angle_left)
        if f.metrics.knee_angle_right is not None:
            knee_r.append(f.metrics.knee_angle_right)
    agg: dict[str, float] = {}

    def _stats(vals: list[float], prefix: str) -> None:
        if vals:
            arr = np.array(vals)
            agg[f"{prefix}_mean"] = float(arr.mean())
            agg[f"{prefix}_std"] = float(arr.std())

    _stats(com_x, "com_x")
    _stats(com_y, "com_y")
    _stats(knee_l, "knee_angle_left")
    _stats(knee_r, "knee_angle_right")
    _stats(weight, "weight_dist_front_pct")
    return agg
