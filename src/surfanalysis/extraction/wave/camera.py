"""Runtime camera geometry container for PhysicalWaveComputer.

Separated from the Pydantic `CameraModel` in `extraction/schema.py` so that
`metrics/` (mypy strict) can stay Pydantic-free. The Pydantic version is for
JSON serialization (metrics.json output); this runtime version is for live
computation, derives `focal_pixels` from focal_length_mm / sensor_height_mm,
and never touches the schema layer at import time.
"""

from __future__ import annotations

from dataclasses import dataclass

from surfanalysis.extraction.schema import CameraModel as CameraModelSchema
from surfanalysis.metrics.wave_geometry import CameraIntrinsics


@dataclass(frozen=True)
class CameraModel:
    """Camera geometry used by PhysicalWaveComputer.

    Returns `None` from `from_cli` when no camera-height metadata was supplied;
    callers should check and emit a `physical_status="insufficient_metadata"`
    warning rather than crashing.
    """

    schema: CameraModelSchema

    @classmethod
    def from_cli(
        cls,
        camera_height_m: float | None,
        focal_length_mm: float | None,
        sensor_height_mm: float | None,
        image_height_px: int,
        pitch_deg: float | None = None,
        roll_deg: float | None = None,
    ) -> CameraModel | None:
        """Build a CameraModel from CLI arguments.

        Returns None if `camera_height_m` is missing (caller should mark
        `physical_status="insufficient_metadata"`). Raises ValueError when
        neither focal_length_mm nor sensor_height_mm is supplied — we need
        at least one to derive `focal_pixels`.
        """
        if camera_height_m is None:
            return None
        if focal_length_mm is None and sensor_height_mm is None:
            raise ValueError(
                "Need focal_length_mm or sensor_height_mm to derive focal_pixels"
            )
        return cls(
            schema=CameraModelSchema(
                camera_height_m=camera_height_m,
                focal_length_mm=focal_length_mm,
                sensor_height_mm=sensor_height_mm,
                image_height_px=image_height_px,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                source="user",
            )
        )

    def to_intrinsics(
        self, pitch_deg: float | None = None, roll_deg: float = 0.0
    ) -> CameraIntrinsics:
        """Derive pinhole intrinsics for the projection helper.

        `pitch_deg` defaults to the schema's stored value (set by the caller
        from horizon detection), but can be overridden per-call if the scene's
        camera tilt changes mid-clip.
        """
        s = self.schema
        if s.focal_length_mm is None or s.sensor_height_mm is None:
            raise ValueError(
                "CameraModel missing focal_length_mm / sensor_height_mm; "
                "needed to derive focal_pixels"
            )
        focal_pixels = s.image_height_px * s.focal_length_mm / s.sensor_height_mm
        effective_pitch = pitch_deg if pitch_deg is not None else (s.pitch_deg or 0.0)
        effective_roll = roll_deg if roll_deg != 0.0 else (s.roll_deg or 0.0)
        return CameraIntrinsics(
            camera_height_m=s.camera_height_m,
            focal_pixels=focal_pixels,
            image_height_px=s.image_height_px,
            pitch_deg=effective_pitch,
            roll_deg=effective_roll,
        )
