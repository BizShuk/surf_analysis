"""Pydantic models defining the metrics.json contract (schema_version 1.2).

Schema 1.2 changes from 1.1:
- `WaveSummary.height_median` / `height_p90` (screen fraction) REMOVED.
- `WaveSummary.height_m_median` / `height_m_p90` (meters) added — None unless
  CLI was run with `--camera-height-m`.
- `WaveSummary.confidence` / `camera` / `physical_status` added.
- `WaveMetrics.height` (per-frame fraction) REMOVED.
- `WaveMetrics.physical: PhysicalWaveFrame | None` added.
- New `CameraModel` and `PhysicalWaveFrame` types.

Per user decision 2026-06-21, schema 1.1 files are NOT silently downgraded;
the CLI raises `IncompatibleSchemaError` when reading a 1.1 metrics.json.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["regular", "goofy"]
WaveView = Literal["facing", "side"]
AngleKind = Literal["crest_tilt", "face_steepness"]

SCHEMA_VERSION: Literal["1.2"] = "1.2"
SUPPORTED_SCHEMA_VERSIONS: tuple[str, ...] = ("1.2",)
LEGACY_NO_WAVE_SCHEMA_VERSION: str = "1.0"


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


class CameraModel(BaseModel):
    """Camera geometry used by PhysicalWaveComputer.

    `focal_length_mm` and `sensor_height_mm` are alternative ways to derive
    `focal_pixels`; pass one (CLI prefers `focal_length_mm` if EXIF gives it).
    `pitch_deg` is normally inferred from `horizon_deg`; callers may override.
    `source` records provenance so the UI can warn about user-supplied values.
    """
    camera_height_m: float
    focal_length_mm: float | None = None
    sensor_height_mm: float | None = None
    image_height_px: int
    pitch_deg: float | None = None
    roll_deg: float | None = None
    source: Literal["user", "exif", "default"] = "user"


class PhysicalWaveFrame(BaseModel):
    """Per-frame physical-wave metrics in world coordinates (meters).

    `crest_world` / `trough_world` are (X, Y, Z) in meters; X is always 0
    (single-view, lateral depth not modeled). Y is the world height above
    the water plane at the corresponding depth Z.

    `method == "skipped"` means the camera metadata was missing — `height_m`
    is None and downstream code must NOT report a number.
    """
    crest_world: tuple[float, float, float] | None = None
    trough_world: tuple[float, float, float] | None = None
    height_m: float | None = None
    method: Literal["camera_geometry", "wavelength", "cross_validated", "skipped"]
    confidence: Literal["high", "medium", "low", "unavailable"]
    reason: str | None = None


class WaveMetrics(BaseModel):
    view: WaveView
    angle_deg: float
    angle_kind: AngleKind
    confidence: float
    angle_line: tuple[tuple[float, float], tuple[float, float]]
    height_top: tuple[float, float]
    height_bottom: tuple[float, float]
    horizon_deg: float | None = None
    physical: PhysicalWaveFrame | None = None


class WaveSummary(BaseModel):
    frames_detected: int
    view: WaveView | Literal["mixed"]
    angle_median: float
    engine: str
    height_m_median: float | None = None
    height_m_p90: float | None = None
    confidence: Literal["high", "medium", "low", "unavailable"] = "unavailable"
    camera: CameraModel | None = None
    physical_status: Literal[
        "computed", "insufficient_metadata", "unsupported_view"
    ] = "insufficient_metadata"


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
