"""Pydantic models defining the metrics.json contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["regular", "goofy"]
WaveView = Literal["facing", "side"]
AngleKind = Literal["crest_tilt", "face_steepness"]


class SourceInfo(BaseModel):
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: float


class EngineInfo(BaseModel):
    name: str
    version: str
    params: dict[str, float | int | str]


class Keypoints(BaseModel):
    points: list[tuple[float, float, float, float]]
    image_size: tuple[int, int]

    @field_validator("points")
    @classmethod
    def _exactly_33(cls, v: list[tuple[float, float, float, float]]):
        if len(v) != 33:
            raise ValueError(f"expected 33 keypoints, got {len(v)}")
        return v


class FrameMetrics(BaseModel):
    com: tuple[float, float]
    weight_dist_front_pct: float
    knee_angle_left: float | None = None
    knee_angle_right: float | None = None
    elbow_angle_left: float | None = None
    elbow_angle_right: float | None = None
    torso_lean_deg: float | None = None
    shoulder_hip_rotation_deg: float | None = None
    com_stability_score: float | None = None


class WaveMetrics(BaseModel):
    view: WaveView
    height: float
    angle_deg: float
    angle_kind: AngleKind
    confidence: float
    angle_line: tuple[tuple[float, float], tuple[float, float]]
    height_top: tuple[float, float]
    height_bottom: tuple[float, float]
    horizon_deg: float | None = None


class WaveSummary(BaseModel):
    frames_detected: int
    view: WaveView | Literal["mixed"]
    height_median: float
    height_p90: float
    angle_median: float
    engine: str


class FrameRecord(BaseModel):
    frame_index: int
    timestamp_ms: float
    keypoints: Keypoints | None
    metrics: FrameMetrics | None
    wave: WaveMetrics | None = None


class SessionSummary(BaseModel):
    frames_with_detection: int
    frames_total: int
    detection_rate: float
    metrics_aggregate: dict[str, float]


class SessionRecord(BaseModel):
    schema_version: str = Field(pattern=r"^\d+\.\d+$")
    source: SourceInfo
    engine: EngineInfo
    stance: Stance
    frames: list[FrameRecord]
    summary: SessionSummary
    wave_engine: EngineInfo | None = None
    wave_summary: WaveSummary | None = None
