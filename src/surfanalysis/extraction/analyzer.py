"""Iterate video frames, invoke PoseEngine, assemble SessionRecord."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np

from surfanalysis.extraction.engine import PoseEngine
from surfanalysis.extraction.schema import (
    FrameRecord,
    Keypoints,
    SessionRecord,
    SessionSummary,
    SourceInfo,
)
from surfanalysis.metrics import StabilityWindow, compute_frame_metrics

Stance = Literal["regular", "goofy"]

SCHEMA_VERSION = "1.0"


def _kp_to_array(kp: Keypoints) -> np.ndarray:
    return np.array(kp.points, dtype=np.float64)


class FrameAnalyzer:
    def __init__(self, engine: PoseEngine, stance: Stance, source: SourceInfo) -> None:
        self._engine = engine
        self._stance = stance
        self._source = source

    def run(self, frames_iter: Iterable[np.ndarray]) -> SessionRecord:
        frames: list[FrameRecord] = []
        stability = StabilityWindow()
        detections = 0

        for idx, frame in enumerate(frames_iter):
            ts_ms = (idx / self._source.fps) * 1000.0 if self._source.fps > 0 else 0.0
            kp = self._engine.detect(frame)
            if kp is None:
                stability.push(None)
                frames.append(FrameRecord(frame_index=idx, timestamp_ms=ts_ms,
                                          keypoints=None, metrics=None))
                continue
            detections += 1
            metrics = compute_frame_metrics(_kp_to_array(kp), self._stance, stability)
            frames.append(FrameRecord(frame_index=idx, timestamp_ms=ts_ms,
                                      keypoints=kp, metrics=metrics))

        total = len(frames)
        rate = detections / total if total else 0.0
        return SessionRecord(
            schema_version=SCHEMA_VERSION,
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
