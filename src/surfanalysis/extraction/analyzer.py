"""Iterate video frames, invoke PoseEngine, assemble SessionRecord."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from surfanalysis.extraction.engine import PoseEngine
from surfanalysis.extraction.schema import (
    EngineInfo,
    FrameRecord,
    Keypoints,
    SessionRecord,
    SessionSummary,
    SourceInfo,
    WaveSummary,
)
from surfanalysis.extraction.wave.base import WaveEngine
from surfanalysis.metrics import StabilityWindow, compute_frame_metrics
from surfanalysis.metrics.wave_geometry import median_p90

Stance = Literal["regular", "goofy"]

SCHEMA_VERSION_BASE = "1.0"
SCHEMA_VERSION_WAVE = "1.1"


def _kp_to_array(kp: Keypoints) -> np.ndarray:
    return np.array(kp.points, dtype=np.float64)


class FrameAnalyzer:
    def __init__(self, engine: PoseEngine, stance: Stance, source: SourceInfo,
                 wave_engine: WaveEngine | None = None) -> None:
        self._engine = engine
        self._stance = stance
        self._source = source
        self._wave_engine = wave_engine

    def run(self, frames_iter: Iterable[np.ndarray]) -> SessionRecord:
        frames: list[FrameRecord] = []
        stability = StabilityWindow()
        detections = 0

        for idx, frame in enumerate(frames_iter):
            ts_ms = (idx / self._source.fps) * 1000.0 if self._source.fps > 0 else 0.0
            wave = self._wave_engine.detect(frame, ts_ms) if self._wave_engine else None
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
            self._aggregate_wave(frames, wave_engine_info)
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
        frames: list[FrameRecord], engine_info: EngineInfo | None
    ) -> WaveSummary | None:
        waves = [f.wave for f in frames if f.wave is not None]
        if not waves:
            return None
        h_med, h_p90 = median_p90([w.height for w in waves])
        a_med, _ = median_p90([w.angle_deg for w in waves])
        views = [w.view for w in waves]
        dom = max(set(views), key=views.count)
        view = dom if all(v == dom for v in views) else "mixed"
        engine_tag = engine_info.name.replace("wave-", "") if engine_info else "unknown"
        return WaveSummary(frames_detected=len(waves), view=view,
                           height_median=h_med, height_p90=h_p90,
                           angle_median=a_med, engine=engine_tag)


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
