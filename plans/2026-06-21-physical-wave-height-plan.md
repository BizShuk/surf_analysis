# Physical Wave Height (meters) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the screen-fraction `wave_summary.height_*` fields with physical `wave_height_m` in meters, computed by one-view camera geometry (baseline) cross-validated by deep-water wavelength dispersion. Surfer/surfboard are explicitly NOT used as scale references. Schema breaks 1.1 → 1.2 with no deprecation window (user decision 2026-06-21).

**Architecture:** Add a parallel `PhysicalWaveComputer` downstream of the existing `WaveEngine`. It consumes `WaveObservation` (crest/base) + a `CameraModel` (height, focal length, image height, pitch from horizon) and emits `PhysicalWaveFrame` per frame + `WaveSummary.height_m_*` aggregated. A pure helper `normalized_to_world_height(...)` lives in `metrics/wave_geometry.py` (mypy strict, no Pydantic). When camera metadata is missing, `physical_status = "insufficient_metadata"` and the per-frame `physical` field is `None`; the CLI prints a clear warning. Schema 1.1 readers MUST fail loudly via `IncompatibleSchemaError` — no silent degradation.

**Tech Stack:** Python 3.11+, OpenCV (already in deps — only `cv2.phaseCorrelate` if we add motion-based view detection, optional), NumPy, Pydantic v2. No new dependencies.

**Spec:** `plans/2026-06-21-physical-wave-height-design.md`. **Branch:** `feat/physical-wave-height`.

---

## Conventions for the implementer

- Run everything through the project venv: `.venv/bin/python -m pytest ...` (the venv has stale shebangs; always use `python -m`). If a `surf` console script is needed, use `.venv/bin/python -m surfanalysis.cli ...`.
- Tests are plain `pytest` functions with synthetic NumPy frames — no fixtures framework beyond what `tests/conftest.py` provides (currently empty).
- `metrics/` is the ONLY mypy-strict package (`pyproject.toml` `files = ["src/surfanalysis/metrics"]`). `metrics/wave_geometry.py` must be fully type-annotated AND must NOT import `surfanalysis.extraction.schema` (that would drag Pydantic into strict checking). Keep it pure numbers/tuples/str.
- Coordinates are normalized `(0-1)` in image space; world coordinates are meters.
- Lint after each task: `.venv/bin/ruff check .` (selects `E,F,W,I,B,UP`, line-length 100).
- Type-check after each task: `.venv/bin/mypy src/surfanalysis/metrics`.
- Commit message trailer for every commit:

```text
Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
```

---

## File structure

```text
src/surfanalysis/
├── extraction/
│   ├── wave/
│   │   ├── __init__.py         add make_physical_computer() factory
│   │   ├── base.py             MODIFY: WaveObservation gains `timestamp_s`; WaveEngine unchanged
│   │   ├── camera.py           NEW CameraModel runtime class
│   │   ├── horizon.py          unchanged (consumed by camera.py)
│   │   ├── physical.py         NEW PhysicalWaveComputer + WavelengthEstimator
│   │   ├── prescan.py          MODIFY: add prescan_physical()
│   │   └── ...                 (ocean/static/base/region/motion unchanged)
│   ├── analyzer.py             MODIFY: FrameAnalyzer calls PhysicalWaveComputer
│   ├── schema.py               BREAK: remove height_*, add CameraModel/PhysicalWaveFrame; bump version
│   └── exceptions.py           NEW IncompatibleSchemaError
├── metrics/
│   └── wave_geometry.py        MODIFY: add normalized_to_world_height() pure
├── rendering/
│   └── wave_overlay.py         MODIFY: show (m) + confidence, no fraction fallback
└── cli.py                      MODIFY: extract --camera-height-m, --focal-length-mm, --sensor-height-mm

tests/
├── test_wave_geometry.py       MODIFY: add tests for normalized_to_world_height
├── test_wave_camera.py         NEW
├── test_wave_physical.py       NEW (PhysicalWaveComputer + WavelengthEstimator + confidence)
├── test_wave_prescan.py        MODIFY: add prescan_physical tests
├── test_schema.py              BREAK: rewrite for 1.2; remove fraction tests
├── test_analyzer.py            MODIFY: physical pipeline path
├── test_cli_extract.py         MODIFY: --camera-height-m, EXIF fallthrough, schema-1.1 rejection
├── test_cli_render.py          MODIFY: 1.2 acceptance, (m) overlay
└── test_e2e.py                 MODIFY: extract --wave --camera-height-m 3.0 → render
```

---

## Phase 1 — Pure geometry (`metrics/wave_geometry.py`)

### Task 1.1: Add `normalized_to_world_height` and helpers

**Files:**
- Modify: `src/surfanalysis/metrics/wave_geometry.py`
- Test: `tests/test_wave_geometry.py`

`metrics/wave_geometry.py` stays pure (no Pydantic, no I/O). Add the pinhole-camera projection helper that turns a normalized `(x, y)` point into a world `(X, Y, Z)` tuple given a `CameraIntrinsics` NamedTuple (height_m, focal_pixels, image_height_px, pitch_deg, roll_deg).

- [ ] **Step 1: Write failing tests** — append to `tests/test_wave_geometry.py`:

```python
def test_normalized_to_world_height_basic():
    # 3 m above water, looking horizontally, 1080 px tall, focal_pixels = 1080
    # A point at image y = cy (horizon) → depth = ∞
    # A point at image y = cy + 540 (bottom of frame) → tan(alpha) = 1
    # Z = H / tan(theta + alpha) = 3 / tan(0 + 45°) = 3 m
    intr = CameraIntrinsics(
        camera_height_m=3.0,
        focal_pixels=1080.0,
        image_height_px=1080,
        pitch_deg=0.0,
        roll_deg=0.0,
    )
    # y = 1.0 (bottom of normalized frame) = pixel 1080, cy = 540
    # alpha = arctan((1080 - 540) / 1080) = arctan(0.5) ≈ 26.565°
    # Z = 3 / tan(26.565°) ≈ 6.0 m
    # Y = Z * tan(alpha) = 6 * 0.5 = 3.0 m
    X, Y, Z = normalized_to_world_height((0.5, 1.0), intr)
    assert Z == pytest.approx(6.0, rel=1e-3)
    assert Y == pytest.approx(3.0, rel=1e-3)


def test_normalized_to_world_height_with_pitch():
    # 3 m above water, pitch 30° down, 1080 px tall, focal 1080
    intr = CameraIntrinsics(3.0, 1080.0, 1080, pitch_deg=30.0, roll_deg=0.0)
    # Center pixel y=0.5 (horizon-ish but pitched): alpha = arctan(0) = 0
    # Z = 3 / tan(30°) = 3 / 0.577 ≈ 5.196 m
    X, Y, Z = normalized_to_world_height((0.5, 0.5), intr)
    assert Z == pytest.approx(5.196, rel=1e-2)


def test_wave_height_from_observations():
    intr = CameraIntrinsics(3.0, 1080.0, 1080, pitch_deg=0.0, roll_deg=0.0)
    crest = (0.5, 0.3)   # near horizon
    base = (0.5, 0.9)    # near bottom
    h = wave_height_meters(crest, base, intr)
    assert h > 0.0
    # Sanity: should be > 1 m for this geometry
    assert h > 1.0


def test_normalized_to_world_height_invalid_intrinsics():
    intr = CameraIntrinsics(0.0, 1080.0, 1080, 0.0, 0.0)
    with pytest.raises(ValueError, match="camera_height_m"):
        normalized_to_world_height((0.5, 0.5), intr)
```

- [ ] **Step 2: Implement the helpers** in `src/surfanalysis/metrics/wave_geometry.py`:

```python
from typing import NamedTuple


class CameraIntrinsics(NamedTuple):
    """Pinhole camera geometry for normalizing image y to world Z (depth)."""
    camera_height_m: float
    focal_pixels: float          # = image_height_px / (2 * tan(fov_half))
    image_height_px: int
    pitch_deg: float             # positive = camera looking down
    roll_deg: float = 0.0


def _alpha_rad(y_norm: float, intr: CameraIntrinsics) -> float:
    """Angle from optical axis to pixel y (rad)."""
    if intr.focal_pixels <= 0 or intr.image_height_px <= 0:
        raise ValueError("focal_pixels and image_height_px must be positive")
    cy = intr.image_height_px / 2.0
    y_px = y_norm * intr.image_height_px
    return math.atan2(y_px - cy, intr.focal_pixels)


def normalized_to_world_height(
    point_norm: Point, intr: CameraIntrinsics
) -> tuple[float, float, float]:
    """Project a normalized (x, y) point to world (X, Y, Z) in meters.

    X is currently returned as 0.0 (lateral depth not modeled; tracked as
    future-work). Y is height above the reference plane (water), Z is depth.
    """
    if intr.camera_height_m <= 0:
        raise ValueError("camera_height_m must be > 0")
    alpha = _alpha_rad(point_norm[1], intr)
    theta = math.radians(intr.pitch_deg)
    if intr.roll_deg != 0.0:
        # 2D rotation about optical axis (image-space x, y)
        x_norm, y_norm = point_norm
        cy = 0.5  # normalized center
        dx = x_norm - 0.5
        dy = y_norm - cy
        roll = math.radians(intr.roll_deg)
        x_rot = dx * math.cos(roll) - dy * math.sin(roll)
        dy_rot = dx * math.sin(roll) + dy * math.cos(roll)
        y_norm = cy + dy_rot
        point_norm = (0.5 + x_rot, y_norm)
        alpha = _alpha_rad(point_norm[1], intr)
    denom = math.tan(theta + alpha)
    if abs(denom) < 1e-9:
        raise ValueError("pitch + alpha near ±90°, projection undefined")
    z = intr.camera_height_m / denom
    y = z * math.tan(alpha)
    return (0.0, y, z)


def wave_height_meters(
    crest: Point, base: Point, intr: CameraIntrinsics
) -> float:
    """Vertical extent between crest and base in meters, world frame."""
    _, y_crest, _ = normalized_to_world_height(crest, intr)
    _, y_base, _ = normalized_to_world_height(base, intr)
    return abs(y_crest - y_base)
```

- [ ] **Step 3: Run tests, mypy, ruff**:
```bash
.venv/bin/python -m pytest tests/test_wave_geometry.py -v
.venv/bin/mypy src/surfanalysis/metrics
.venv/bin/ruff check .
```

- [ ] **Step 4: Commit**: `feat(metrics): add normalized_to_world_height for pinhole camera projection`

---

## Phase 2 — Schema 1.2 (BREAK)

### Task 2.1: Add `IncompatibleSchemaError` and bump version

**Files:**
- New: `src/surfanalysis/extraction/exceptions.py`
- Modify: `src/surfanalysis/extraction/schema.py` (remove fraction fields, add physical fields)

- [ ] **Step 1: Write the exception** in `src/surfanalysis/extraction/exceptions.py`:

```python
class IncompatibleSchemaError(Exception):
    """Raised when reading a metrics.json with an unsupported schema_version.

    Per user decision 2026-06-21: no silent degradation. Schema 1.1 files
    must be re-extracted with the current CLI to be read by 1.2 readers.
    """
```

- [ ] **Step 2: Write failing schema tests** in `tests/test_schema.py` (rewrite the wave-related section):

```python
import pytest
from pydantic import ValidationError
from surfanalysis.extraction.exceptions import IncompatibleSchemaError
from surfanalysis.extraction.schema import (
    CameraModel, PhysicalWaveFrame, WaveMetrics, WaveSummary,
    SessionRecord, SCHEMA_VERSION,
)


def test_schema_version_is_1_2():
    assert SCHEMA_VERSION == "1.2"


def test_wave_summary_no_fraction_fields():
    # 1.2 must NOT have height_median / height_p90
    summary = WaveSummary(
        frames_detected=100,
        view="facing",
        angle_median=0.0,
        engine="ocean",
        height_m_median=0.85,
        height_m_p90=1.10,
        confidence="high",
        physical_status="computed",
    )
    with pytest.raises(AttributeError):
        _ = summary.height_median  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        _ = summary.height_p90  # type: ignore[attr-defined]


def test_wave_metrics_has_physical():
    phys = PhysicalWaveFrame(
        crest_world=(0.0, 0.5, 3.0),
        trough_world=(0.0, -0.3, 2.5),
        height_m=0.8,
        method="camera_geometry",
        confidence="high",
    )
    wm = WaveMetrics(
        view="facing", height=0.0,  # sentinel; field kept but unused
        angle_deg=0.0, angle_kind="crest_tilt",
        confidence=0.0,
        angle_line=((0.0, 0.0), (1.0, 0.0)),
        height_top=(0.5, 0.3),
        height_bottom=(0.5, 0.9),
        physical=phys,
    )
    assert wm.physical is phys
    assert wm.physical.height_m == pytest.approx(0.8)


def test_camera_model_required_fields():
    cam = CameraModel(
        camera_height_m=3.0,
        focal_length_mm=16.0,
        sensor_height_mm=7.0,
        image_height_px=1080,
        pitch_deg=15.0,
        roll_deg=0.0,
        source="user",
    )
    assert cam.camera_height_m == 3.0
```

- [ ] **Step 3: Modify `src/surfanalysis/extraction/schema.py`**:

```python
# At the top
SCHEMA_VERSION: Literal["1.2"] = "1.2"
SUPPORTED_VERSIONS: tuple[str, ...] = ("1.2",)

# Remove `WaveMetrics.height` references in serialization; keep field for
# back-compat in *typing* but mark `exclude=True`:
class WaveMetrics(BaseModel):
    view: WaveView
    height: float = 0.0  # deprecated; excluded from dump
    angle_deg: float
    angle_kind: AngleKind
    confidence: float
    angle_line: tuple[tuple[float, float], tuple[float, float]]
    height_top: tuple[float, float]
    height_bottom: tuple[float, float]
    horizon_deg: float | None = None
    physical: PhysicalWaveFrame | None = None

    model_config = ConfigDict(  # type: ignore[assignment]
        json_schema_extra={"exclude": ["height"]}  # Pydantic v2: use model_dump(exclude=...)
    )


# New
class CameraModel(BaseModel):
    camera_height_m: float
    focal_length_mm: float | None = None
    sensor_height_mm: float | None = None
    image_height_px: int
    pitch_deg: float | None = None
    roll_deg: float | None = None
    source: Literal["user", "exif", "default"]


class PhysicalWaveFrame(BaseModel):
    crest_world: tuple[float, float, float] | None = None
    trough_world: tuple[float, float, float] | None = None
    height_m: float | None = None
    method: Literal["camera_geometry", "wavelength", "cross_validated", "skipped"]
    confidence: Literal["high", "medium", "low", "unavailable"]
    reason: str | None = None


class WaveSummary(BaseModel):
    frames_detected: int
    view: WaveView | Literal["mixed"]
    angle_median: float
    engine: str
    height_m_median: float | None = None
    height_m_p90: float | None = None
    confidence: Literal["high", "medium", "low", "unavailable"] = "unavailable"
    camera: CameraModel | None = None
    physical_status: Literal["computed", "insufficient_metadata", "unsupported_view"] = "insufficient_metadata"
```

Also delete any `SessionRecord.wave_summary.height_median` / `height_p90` references in tests and call sites.

- [ ] **Step 4: Update `wave_summary` aggregation in `FrameAnalyzer`** (and any other call site). Search:
```bash
grep -rn "height_median\|height_p90" src/ tests/
```
For each hit: if it's a *write* to `WaveSummary`, replace with `height_m_median` / `height_m_p90` (compute via `PhysicalWaveComputer`). If it's a *read*, update to the new field. If it's a test asserting the old fields exist, delete those assertions (they belong to 1.1).

- [ ] **Step 5: Run**:
```bash
.venv/bin/python -m pytest tests/test_schema.py -v
.venv/bin/ruff check .
```

- [ ] **Step 6: Commit**: `feat(schema): bump to 1.2 — remove fraction height_*, add physical_*`

---

## Phase 3 — `CameraModel` runtime + `prescan_physical`

### Task 3.1: New `extraction/wave/camera.py`

**Files:**
- New: `src/surfanalysis/wave/camera.py`
- Test: `tests/test_wave_camera.py`

- [ ] **Step 1: Write failing tests**:

```python
def test_camera_model_from_user_input():
    cam = CameraModel.from_cli(
        camera_height_m=3.0,
        focal_length_mm=16.0,
        sensor_height_mm=7.0,
        image_height_px=1080,
    )
    assert cam.source == "user"
    assert cam.camera_height_m == 3.0


def test_camera_model_focal_pixels_computed():
    cam = CameraModel.from_cli(
        camera_height_m=3.0,
        focal_length_mm=16.0,
        sensor_height_mm=7.0,
        image_height_px=1080,
    )
    # focal_pixels = image_height_px * focal_length_mm / sensor_height_mm
    expected = 1080 * 16.0 / 7.0
    intr = cam.to_intrinsics(pitch_deg=0.0)
    assert intr.focal_pixels == pytest.approx(expected, rel=1e-4)


def test_camera_model_requires_height():
    with pytest.raises(ValueError):
        CameraModel.from_cli(
            camera_height_m=None, focal_length_mm=16.0, sensor_height_mm=7.0,
            image_height_px=1080,
        )
```

- [ ] **Step 2: Implement** in `src/surfanalysis/extraction/wave/camera.py`:

```python
from dataclasses import dataclass
from surfanalysis.extraction.schema import CameraModel as CameraModelSchema
from surfanalysis.metrics.wave_geometry import CameraIntrinsics


@dataclass
class CameraModel:
    """Runtime camera geometry container (separate from Pydantic schema)."""
    schema: CameraModelSchema

    @classmethod
    def from_cli(
        cls,
        camera_height_m: float | None,
        focal_length_mm: float | None,
        sensor_height_mm: float | None,
        image_height_px: int,
    ) -> "CameraModel | None":
        if camera_height_m is None:
            return None
        if focal_length_mm is None and sensor_height_mm is None:
            raise ValueError("Need focal_length_mm or sensor_height_mm")
        return cls(
            schema=CameraModelSchema(
                camera_height_m=camera_height_m,
                focal_length_mm=focal_length_mm,
                sensor_height_mm=sensor_height_mm,
                image_height_px=image_height_px,
                source="user",
            )
        )

    def to_intrinsics(self, pitch_deg: float, roll_deg: float = 0.0) -> CameraIntrinsics:
        s = self.schema
        if s.focal_length_mm is None or s.sensor_height_mm is None:
            raise ValueError("CameraModel missing focal/sensor for intrinsics")
        focal_pixels = s.image_height_px * s.focal_length_mm / s.sensor_height_mm
        return CameraIntrinsics(
            camera_height_m=s.camera_height_m,
            focal_pixels=focal_pixels,
            image_height_px=s.image_height_px,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
```

- [ ] **Step 3: Run + lint + commit**:
```bash
.venv/bin/python -m pytest tests/test_wave_camera.py -v
.venv/bin/ruff check .
git add -A && git commit -m "feat(wave): add CameraModel runtime with focal_pixels derivation"
```

### Task 3.2: Add `prescan_physical`

**Files:**
- Modify: `src/surfanalysis/extraction/wave/prescan.py`
- Test: `tests/test_wave_prescan.py`

- [ ] **Step 1: Tests**:

```python
def test_prescan_physical_with_camera():
    from surfanalysis.extraction.wave.prescan import prescan_physical
    assert prescan_physical(camera_model=mock_camera()) == "computed"


def test_prescan_physical_without_camera():
    from surfanalysis.extraction.wave.prescan import prescan_physical
    assert prescan_physical(camera_model=None) == "insufficient_metadata"
```

- [ ] **Step 2: Implement**:

```python
def prescan_physical(
    camera_model: CameraModel | None,
    view: str = "facing",
) -> Literal["computed", "insufficient_metadata", "unsupported_view"]:
    if view not in ("facing", "side"):
        return "unsupported_view"
    if camera_model is None:
        return "insufficient_metadata"
    return "computed"
```

- [ ] **Step 3: Run + commit**: `feat(wave): prescan_physical decides computed/missing/unsupported`

---

## Phase 4 — `PhysicalWaveComputer`

### Task 4.1: Core computation

**Files:**
- New: `src/surfanalysis/extraction/wave/physical.py`
- Test: `tests/test_wave_physical.py`

- [ ] **Step 1: Tests** (in `tests/test_wave_physical.py`):

```python
def test_compute_height_with_camera_geometry():
    from surfanalysis.extraction.wave.physical import PhysicalWaveComputer
    cam = CameraModel.from_cli(3.0, 16.0, 7.0, 1080)
    pc = PhysicalWaveComputer(camera_model=cam, horizon_deg=0.0)
    obs = WaveObservation(crest=(0.5, 0.3), base=(0.5, 0.9))
    frame = pc.compute(obs)
    assert frame.method == "camera_geometry"
    assert frame.height_m is not None
    assert frame.height_m > 0.0
    assert frame.confidence in ("high", "medium", "low", "unavailable")


def test_compute_height_skipped_without_camera():
    from surfanalysis.extraction.wave.physical import PhysicalWaveComputer
    pc = PhysicalWaveComputer(camera_model=None, horizon_deg=0.0)
    obs = WaveObservation(crest=(0.5, 0.3), base=(0.5, 0.9))
    frame = pc.compute(obs)
    assert frame.method == "skipped"
    assert frame.height_m is None
    assert frame.confidence == "unavailable"
    assert frame.reason is not None
    assert "camera_height_m" in frame.reason
```

- [ ] **Step 2: Implement** the `PhysicalWaveComputer` class:

```python
class PhysicalWaveComputer:
    def __init__(
        self,
        camera_model: CameraModel | None,
        horizon_deg: float,
        wavelength_estimator: WavelengthEstimator | None = None,
    ) -> None:
        self.camera_model = camera_model
        self.horizon_deg = horizon_deg
        self.wavelength_estimator = wavelength_estimator or WavelengthEstimator()

    def compute(self, obs: "WaveObservation") -> PhysicalWaveFrame:
        if self.camera_model is None:
            return PhysicalWaveFrame(
                method="skipped",
                confidence="unavailable",
                reason="insufficient_metadata: provide --camera-height-m",
            )
        intr = self.camera_model.to_intrinsics(pitch_deg=self.horizon_deg)
        try:
            crest_world = normalized_to_world_height(obs.crest, intr)
            base_world = normalized_to_world_height(obs.base, intr)
            h_camera = abs(crest_world[1] - base_world[1])
        except ValueError as e:
            return PhysicalWaveFrame(
                method="camera_geometry",
                confidence="low",
                reason=f"projection failed: {e}",
            )
        # Cross-validate with wavelength if available
        h_wave = self.wavelength_estimator.estimate_break_height()
        confidence = score_confidence(h_camera, h_wave)
        return PhysicalWaveFrame(
            crest_world=crest_world,
            trough_world=base_world,
            height_m=h_camera,
            method="cross_validated" if h_wave is not None else "camera_geometry",
            confidence=confidence,
        )
```

Also implement `WavelengthEstimator` (initial stub returning `None`; real implementation in Task 4.2) and `score_confidence(h_camera, h_wave)`:

```python
def score_confidence(
    h_camera: float | None, h_wave: float | None
) -> Literal["high", "medium", "low", "unavailable"]:
    if h_camera is None:
        return "unavailable"
    if h_wave is None:
        return "medium"  # single-path, treat as medium unless n_frames small
    if h_wave <= 0:
        return "low"
    delta = abs(h_camera - h_wave) / h_wave
    if delta <= 0.20:
        return "high"
    if delta <= 0.50:
        return "medium"
    return "low"
```

- [ ] **Step 3: Run + commit**: `feat(wave): PhysicalWaveComputer with camera_geometry + confidence`

### Task 4.2: Wavelength estimator

- [ ] **Step 1: Tests**:

```python
def test_wavelength_from_period():
    from surfanalysis.extraction.wave.physical import WavelengthEstimator
    est = WavelengthEstimator()
    # T = 8 s, deep water: L = g*T²/(2π) = 9.81*64/6.283 ≈ 100 m
    result = est.from_period_s(8.0)
    assert result is not None
    assert result.wavelength_m == pytest.approx(100.0, rel=1e-2)
    assert 10.0 < result.h_upper_m < 14.5  # L/7
    assert 10.0 < result.h_lower_m < 11.0  # L/10


def test_wavelength_unavailable_for_short_period():
    from surfanalysis.extraction.wave.physical import WavelengthEstimator
    est = WavelengthEstimator()
    result = est.from_period_s(0.0)
    assert result is None
```

- [ ] **Step 2: Implement**:

```python
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WavelengthEstimate:
    wavelength_m: float
    period_s: float
    h_upper_m: float
    h_lower_m: float
    confidence: Literal["high", "medium", "low", "unavailable"] = "medium"


class WavelengthEstimator:
    G = 9.81

    def from_period_s(self, period_s: float) -> WavelengthEstimate | None:
        if period_s <= 0:
            return None
        L = self.G * period_s * period_s / (2 * math.pi)
        return WavelengthEstimate(
            wavelength_m=L,
            period_s=period_s,
            h_upper_m=L / 7.0,
            h_lower_m=L / 10.0,
        )

    def estimate_break_height(self) -> float | None:
        # Stub: real implementation derives T from crest passage timing
        # in the ocean engine. Returns None until that hooks in.
        return None
```

- [ ] **Step 3: Run + commit**: `feat(wave): WavelengthEstimator (deep-water dispersion)`

### Task 4.3: Wire `PhysicalWaveComputer` into `FrameAnalyzer`

**Files:**
- Modify: `src/surfanalysis/extraction/analyzer.py`
- Test: `tests/test_analyzer.py`

- [ ] **Step 1: Tests** — extend `test_analyzer.py`:

```python
def test_frame_analyzer_calls_physical_computer():
    from surfanalysis.extraction.analyzer import FrameAnalyzer
    from surfanalysis.extraction.wave.camera import CameraModel
    cam = CameraModel.from_cli(3.0, 16.0, 7.0, 1080)
    analyzer = FrameAnalyzer(
        pose_engine=MockPoseEngine(),
        wave_engine=MockWaveEngine(),
        camera_model=cam,
    )
    rec = analyzer.analyze(frame=mock_frame(), frame_index=0, timestamp_ms=0)
    assert rec.wave is not None
    assert rec.wave.physical is not None
    assert rec.wave.physical.method in ("camera_geometry", "cross_validated", "skipped")


def test_frame_analyzer_aggregates_height_m():
    analyzer = FrameAnalyzer(
        pose_engine=MockPoseEngine(),
        wave_engine=MockWaveEngine(),
        camera_model=cam_with_metadata(),
    )
    session = analyzer.run(video=mock_video(n_frames=10))
    ws = session.wave_summary
    assert ws.height_m_median is not None or ws.physical_status == "insufficient_metadata"
    if ws.physical_status == "computed":
        assert ws.height_m_median > 0.0
        assert ws.confidence in ("high", "medium", "low", "unavailable")
```

- [ ] **Step 2: Modify `FrameAnalyzer.__init__` to accept `camera_model` and instantiate `PhysicalWaveComputer` lazily**. Update `aggregate_session` to compute `height_m_median` / `height_m_p90` from `frame.wave.physical.height_m` (only for `confidence ∈ {"high", "medium"}` frames).

- [ ] **Step 3: Run + commit**: `feat(extraction): FrameAnalyzer integrates PhysicalWaveComputer`

---

## Phase 5 — CLI flags

### Task 5.1: Add `--camera-height-m` etc. to `extract`

**Files:**
- Modify: `src/surfanalysis/cli.py`
- Test: `tests/test_cli_extract.py`

- [ ] **Step 1: Tests**:

```python
def test_extract_with_camera_height_creates_physical_metrics(tmp_path):
    out = run_cli(["extract", str(VIDEO), "--wave", "--camera-height-m", "3.0",
                   "--focal-length-mm", "16.0", "--sensor-height-mm", "7.0"])
    metrics = json.loads(out.read_text())
    assert metrics["schema_version"] == "1.2"
    assert metrics["wave_summary"]["physical_status"] == "computed"
    assert metrics["wave_summary"]["camera"]["camera_height_m"] == 3.0


def test_extract_without_camera_height_warns(tmp_path, capsys):
    out = run_cli(["extract", str(VIDEO), "--wave"])
    metrics = json.loads(out.read_text())
    assert metrics["wave_summary"]["physical_status"] == "insufficient_metadata"
    assert "camera_height_m" in capsys.readouterr().err


def test_extract_rejects_schema_1_1_metrics(tmp_path):
    legacy = tmp_path / "legacy.metrics.json"
    legacy.write_text(LEGACY_1_1_PAYLOAD)
    result = run_cli(["render", str(VIDEO), str(legacy)], expect_exit=2)
    assert "schema" in result.stderr.lower()
```

- [ ] **Step 2: Modify `cli.py`** — add argparse flags and pass to `FrameAnalyzer`. After successful extract, if `physical_status == "insufficient_metadata"`, print to stderr:

```text
warning: --camera-height-m not provided; wave height in metrics.json is null
         pass --camera-height-m <meters> for physical wave height in meters
```

Add the schema-version gate in `render`:

```python
def _validate_metrics_schema(metrics_path: Path) -> None:
    raw = json.loads(metrics_path.read_text())
    if raw.get("schema_version") not in ("1.2",):
        raise IncompatibleSchemaError(
            f"schema_version={raw.get('schema_version')!r} not supported; "
            f"re-extract with current CLI (requires 1.2)"
        )
```

- [ ] **Step 3: Run + commit**: `feat(cli): --camera-height-m + 1.2 schema gate`

---

## Phase 6 — Overlay

### Task 6.1: Show `(m)` + confidence badge, no fraction fallback

**Files:**
- Modify: `src/surfanalysis/rendering/wave_overlay.py`
- Test: `tests/test_wave_overlay.py`

- [ ] **Step 1: Tests**:

```python
def test_overlay_shows_meters_when_computed():
    metrics = sample_1_2_metrics(physical_status="computed", height_m_median=0.85, confidence="high")
    frame = render_overlay(blank_frame, metrics)
    text = extract_text_from_frame(frame)
    assert "0.85" in text and "m" in text
    assert "high" in text.lower()


def test_overlay_shows_metadata_warning():
    metrics = sample_1_2_metrics(physical_status="insufficient_metadata")
    frame = render_overlay(blank_frame, metrics)
    text = extract_text_from_frame(frame)
    assert "camera-height-m" in text
    # NO fraction number should appear
    assert "fraction" not in text.lower()
```

- [ ] **Step 2: Update `wave_overlay.py`**:

```python
def _draw_hud(summary: WaveSummary) -> None:
    if summary.physical_status == "computed" and summary.height_m_median is not None:
        h = summary.height_m_median
        conf = summary.confidence
        draw_text(f"H = {h:.2f} m  ({conf})", color=_conf_color(conf))
    elif summary.physical_status == "insufficient_metadata":
        draw_text("H: needs --camera-height-m", color=GRAY)
    elif summary.physical_status == "unsupported_view":
        draw_text("H: unsupported view (POV/handheld)", color=GRAY)
```

Delete any code path that draws the old fraction number.

- [ ] **Step 3: Run + commit**: `feat(render): overlay shows (m) + confidence, no fraction fallback`

---

## Phase 7 — E2E on `sample.MOV`

### Task 7.1: Re-extract sample with camera metadata

- [ ] **Step 1: Estimate camera setup**. The sample was shot handheld above a wave pool; assume `camera_height_m = 2.5` (eye level standing on pool deck), `focal_length_mm = 26.0` (iPhone wide), `sensor_height_mm = 4.0` (typical phone sensor).

- [ ] **Step 2: Run**:
```bash
.venv/bin/python -m surfanalysis.cli extract sample/sample.MOV \
    --wave --wave-engine ocean --view facing \
    --camera-height-m 2.5 \
    --focal-length-mm 26.0 --sensor-height-mm 4.0 \
    -o sample/sample.metrics.json
```

- [ ] **Step 3: Verify**:
```bash
.venv/bin/python -c "
import json
d = json.load(open('sample/sample.metrics.json'))
print('schema:', d['schema_version'])
ws = d['wave_summary']
print('physical_status:', ws['physical_status'])
print('camera:', ws.get('camera'))
print('height_m_median:', ws.get('height_m_median'))
print('height_m_p90:', ws.get('height_m_p90'))
print('confidence:', ws.get('confidence'))
"
```

- [ ] **Step 4: Render and visually inspect**:
```bash
.venv/bin/python -m surfanalysis.cli render \
    sample/sample.MOV sample/sample.metrics.json \
    -o sample/sample.annotated.mp4
```
Open the video; confirm `H = X.XX m (high)` appears in the HUD.

- [ ] **Step 5: Commit**: `docs: refresh sample.metrics.json (1.2) with physical wave height`

---

## Phase 8 — Final verification

- [ ] **Step 1: Full test suite**:
```bash
.venv/bin/python -m pytest -v
```

- [ ] **Step 2: Lint + type-check**:
```bash
.venv/bin/ruff check .
.venv/bin/mypy src/surfanalysis/metrics
```

- [ ] **Step 3: Sanity-check schema migration** — confirm no `height_median` / `height_p90` references remain in `src/` or `tests/`:
```bash
grep -rn "height_median\|height_p90" src/ tests/  # must return nothing
```

- [ ] **Step 4: Update `CLAUDE.md`** — change the `⚠` warning under `## Run` from "is fraction" to "is null unless --camera-height-m provided"; update `## Wave height semantics` to mark "design landed" and link the design+plan files.

- [ ] **Step 5: Update README** output-format section to show `height_m_median` / `confidence` instead of `height_median`.

- [ ] **Step 6: Commit**: `docs: update CLAUDE.md + README for schema 1.2 (physical wave height)`

---

## Definition of Done

- [ ] All 8 phases committed.
- [ ] `pytest` green.
- [ ] `ruff` + `mypy` clean.
- [ ] `sample/sample.metrics.json` regenerated under schema 1.2 with `height_m_median > 0`.
- [ ] `sample.annotated.mp4` shows `H = X.XX m (confidence)` in HUD.
- [ ] No `height_median` / `height_p90` references remain in code or tests.
- [ ] `CLAUDE.md` + `README.md` updated to reflect 1.2 contract.
