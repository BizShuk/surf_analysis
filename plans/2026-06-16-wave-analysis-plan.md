# Wave Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add optional wave-face analysis (normalized wave height + wave angle) to the existing two-stage CLI — `extract` emits view-aware `WaveMetrics` per frame plus a session `WaveSummary`; `render` draws an angle line + height bracket + HUD, fully decoupled from the pose overlay.

**Architecture:** A new `WaveEngine` Strategy (mirrors the existing `PoseEngine`) with two OpenCV implementations — `ocean` (horizon-anchored, per-frame, camera-motion-agnostic) and `static` (MOG2 background subtraction for fixed-camera wave pools). A single pre-scan picks both the engine (global motion) and the camera view (`facing` vs `side`). Raw pixel geometry (`WaveObservation`) is extracted by the engines; pure numeric helpers in `metrics/wave_geometry.py` convert it to normalized metrics. Pose and wave run in parallel and never share state.

**Tech Stack:** Python 3.11+, OpenCV (`cv2.BackgroundSubtractorMOG2`, `phaseCorrelate`, `HoughLinesP`, `fitLine`), NumPy, Pydantic v2. No new dependencies — all CV primitives are in the already-installed `opencv-python`.

**Spec:** `plans/2026-06-15-wave-analysis-design.md`. **Branch:** `feat/wave-analysis`.

---

## Conventions for the implementer

- Run everything through the project venv: `.venv/bin/python -m pytest ...` (the venv has stale shebangs; always use `python -m`). If a `surf` console script is needed, use `.venv/bin/python -m surfanalysis.cli ...`.
- Tests are plain `pytest` functions with synthetic NumPy frames — no fixtures framework beyond what `tests/conftest.py` provides (currently empty).
- `metrics/` is the ONLY mypy-strict package (`pyproject.toml` `files = ["src/surfanalysis/metrics"]`). `metrics/wave_geometry.py` must be fully type-annotated AND must NOT import `surfanalysis.extraction.schema` (that would drag Pydantic into strict checking). Keep it pure numbers/tuples/str.
- Coordinates are normalized `(0-1)` throughout, matching the rest of the codebase.
- Lint after each task: `.venv/bin/ruff check .` (selects `E,F,W,I,B,UP`, line-length 100).
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
│   │   ├── __init__.py        make_wave_engine() factory (avoids base↔engine import cycle)
│   │   ├── base.py            WaveObservation, WaveEngine(ABC), MockWaveEngine, to_wave_metrics()
│   │   ├── horizon.py         detect_horizon(frame) -> float | None  (tilt degrees)
│   │   ├── motion.py          global_motion(prev_gray, cur_gray) -> float
│   │   ├── region.py          region_from_mask(mask) -> WaveObservation | None
│   │   ├── ocean.py           HorizonAnchoredWaveEngine + wave_mask()
│   │   ├── static.py          Mog2WaveEngine
│   │   └── prescan.py         prescan(frames) -> (engine_name, view)
│   ├── analyzer.py            MODIFY: FrameAnalyzer accepts wave_engine
│   └── schema.py              MODIFY: WaveMetrics, WaveSummary, FrameRecord.wave, SessionRecord.*
├── metrics/
│   └── wave_geometry.py       NEW pure funcs (mypy strict, no Pydantic import)
├── rendering/
│   └── wave_overlay.py        NEW WaveOverlay (decoupled from pose overlay)
└── cli.py                     MODIFY: extract + render flags

tests/
├── test_wave_geometry.py      NEW
├── test_wave_base.py          NEW
├── test_wave_horizon.py       NEW
├── test_wave_motion.py        NEW
├── test_wave_region.py        NEW
├── test_wave_engines.py       NEW (ocean + static)
├── test_wave_prescan.py       NEW
├── test_wave_overlay.py       NEW
├── test_schema.py             MODIFY (wave round-trip + 1.0 back-compat)
├── test_analyzer.py           MODIFY (wave_engine path)
├── test_cli_extract.py        MODIFY (--wave)
├── test_cli_render.py         MODIFY (1.1 accepted, --show-wave)
└── test_e2e.py                MODIFY (extract --wave -> render)
```

---

## Phase 1 — Contract & pure geometry

### Task 1: Schema additions (`WaveMetrics`, `WaveSummary`, record fields)

**Files:**
- Modify: `src/surfanalysis/extraction/schema.py`
- Test: `tests/test_schema.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_schema.py`:

```python
from surfanalysis.extraction.schema import WaveMetrics, WaveSummary


def _wave():
    return WaveMetrics(
        view="facing", height=0.42, angle_deg=8.3, angle_kind="crest_tilt",
        confidence=0.74, angle_line=((0.18, 0.31), (0.86, 0.27)),
        height_top=(0.52, 0.29), height_bottom=(0.52, 0.71), horizon_deg=-1.2,
    )


def test_wave_metrics_round_trip():
    w = _wave()
    restored = WaveMetrics.model_validate_json(w.model_dump_json())
    assert restored.view == "facing"
    assert restored.angle_kind == "crest_tilt"
    assert restored.angle_line[0] == (0.18, 0.31)


def test_wave_view_must_be_valid():
    with pytest.raises(ValidationError):
        WaveMetrics(
            view="overhead", height=0.4, angle_deg=0.0, angle_kind="crest_tilt",
            confidence=0.5, angle_line=((0.0, 0.0), (1.0, 0.0)),
            height_top=(0.5, 0.1), height_bottom=(0.5, 0.9),
        )


def test_frame_record_wave_defaults_none():
    fr = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None, metrics=None)
    assert fr.wave is None


def test_session_record_back_compat_v1_0_without_wave():
    src = SourceInfo(path="x.mp4", width=1, height=1, fps=1.0,
                     total_frames=0, duration_ms=0.0)
    eng = EngineInfo(name="mediapipe", version="x", params={})
    summary = SessionSummary(frames_with_detection=0, frames_total=0,
                             detection_rate=0.0, metrics_aggregate={})
    s = SessionRecord(schema_version="1.0", source=src, engine=eng,
                      stance="regular", frames=[], summary=summary)
    assert s.wave_engine is None
    assert s.wave_summary is None
    # a 1.0 JSON blob (no wave keys) still validates:
    restored = SessionRecord.model_validate_json(s.model_dump_json())
    assert restored.wave_summary is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_schema.py -q`
Expected: FAIL with `ImportError: cannot import name 'WaveMetrics'`.

- [ ] **Step 3: Implement the schema changes** — edit `src/surfanalysis/extraction/schema.py`. Add after the `Stance` alias:

```python
WaveView = Literal["facing", "side"]
AngleKind = Literal["crest_tilt", "face_steepness"]
```

Add these models before `FrameRecord`:

```python
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
```

Add `wave` to `FrameRecord`:

```python
class FrameRecord(BaseModel):
    frame_index: int
    timestamp_ms: float
    keypoints: Keypoints | None
    metrics: FrameMetrics | None
    wave: WaveMetrics | None = None
```

Add two fields to `SessionRecord` (after `summary`):

```python
    wave_engine: EngineInfo | None = None
    wave_summary: WaveSummary | None = None
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_schema.py -q`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/schema.py tests/test_schema.py
git commit -m "feat(schema): add WaveMetrics/WaveSummary, schema 1.1 (back-compatible)"
```

---

### Task 2: Pure wave geometry (`metrics/wave_geometry.py`)

**Files:**
- Create: `src/surfanalysis/metrics/wave_geometry.py`
- Test: `tests/test_wave_geometry.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_wave_geometry.py`:

```python
import pytest

from surfanalysis.metrics.wave_geometry import (
    angle_vs_horizon_deg,
    classify_view,
    line_angle_deg,
    median_p90,
    normalized_height,
)


def test_line_angle_horizontal_is_zero():
    assert line_angle_deg((0.0, 0.5), (1.0, 0.5)) == pytest.approx(0.0)


def test_line_angle_normalizes_to_pm90():
    # a near-vertical line (image y grows downward) -> close to +90
    assert line_angle_deg((0.5, 0.1), (0.5, 0.9)) == pytest.approx(90.0)


def test_angle_vs_horizon_subtracts_roll():
    line = ((0.0, 0.50), (1.0, 0.40))  # tilts up to the right
    bare = angle_vs_horizon_deg(line, 0.0)
    rolled = angle_vs_horizon_deg(line, -5.0)
    assert rolled == pytest.approx(bare + 5.0)


def test_normalized_height_is_vertical_extent():
    assert normalized_height((0.5, 0.20), (0.5, 0.75)) == pytest.approx(0.55)


def test_classify_view_facing_when_wide_and_flat():
    # wide region, near-horizontal crest -> facing
    assert classify_view(0.8, 0.4, ((0.1, 0.3), (0.9, 0.28))) == "facing"


def test_classify_view_side_when_steep():
    # steep crest/face line -> side regardless of aspect
    assert classify_view(0.6, 0.6, ((0.2, 0.8), (0.6, 0.3))) == "side"


def test_median_p90():
    med, p90 = median_p90([0.1, 0.2, 0.3, 0.4, 0.5])
    assert med == pytest.approx(0.3)
    assert p90 == pytest.approx(0.5)


def test_median_p90_empty():
    assert median_p90([]) == (0.0, 0.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_geometry.py -q`
Expected: FAIL with `ModuleNotFoundError: surfanalysis.metrics.wave_geometry`.

- [ ] **Step 3: Implement** — create `src/surfanalysis/metrics/wave_geometry.py`:

```python
"""Pure geometry + aggregation for wave metrics.

mypy-strict module: fully annotated, no Pydantic / no I/O.
"""

from __future__ import annotations

import math
from statistics import median

from surfanalysis.metrics.geometry import wrap_to_180

Point = tuple[float, float]
Line = tuple[Point, Point]


def line_angle_deg(p1: Point, p2: Point) -> float:
    """Angle of the line p1->p2 vs image-horizontal, normalized to (-90, 90].

    Image y grows downward, so a line going down-to-the-right is positive.
    """
    ang = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
    while ang > 90.0:
        ang -= 180.0
    while ang <= -90.0:
        ang += 180.0
    return ang


def angle_vs_horizon_deg(line: Line, horizon_deg: float) -> float:
    """Line tilt relative to the detected horizon (absorbs camera roll)."""
    return wrap_to_180(line_angle_deg(line[0], line[1]) - horizon_deg)


def normalized_height(top: Point, bottom: Point) -> float:
    """Vertical extent between two normalized points (already 0-1)."""
    return abs(top[1] - bottom[1])


def classify_view(bbox_w: float, bbox_h: float, crest_line: Line) -> str:
    """Return "facing" or "side" from wave-region shape + top-edge tilt.

    Heuristic (tunable): a facing wave fills the frame width with a gently
    tilted top edge; a side/profile wave shows a steeply tilted face line.
    """
    tilt = abs(line_angle_deg(crest_line[0], crest_line[1]))
    aspect = bbox_w / bbox_h if bbox_h > 0.0 else 0.0
    if tilt < 30.0 and aspect >= 1.0:
        return "facing"
    return "side"


def median_p90(values: list[float]) -> tuple[float, float]:
    """Return (median, 90th-percentile) of values; (0.0, 0.0) if empty."""
    if not values:
        return (0.0, 0.0)
    ordered = sorted(values)
    med = float(median(ordered))
    idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return (med, float(ordered[idx]))
```

- [ ] **Step 4: Run tests + mypy strict**

Run: `.venv/bin/python -m pytest tests/test_wave_geometry.py -q`
Expected: PASS.
Run: `.venv/bin/python -m mypy src/surfanalysis/metrics`
Expected: `Success: no issues found`.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/metrics/wave_geometry.py tests/test_wave_geometry.py
git commit -m "feat(metrics): pure wave geometry helpers (angle/height/view/aggregate)"
```

---

## Phase 2 — Wave detection primitives

### Task 3: `WaveEngine` base, `WaveObservation`, `to_wave_metrics`, `MockWaveEngine`

**Files:**
- Create: `src/surfanalysis/extraction/wave/__init__.py` (empty for now)
- Create: `src/surfanalysis/extraction/wave/base.py`
- Test: `tests/test_wave_base.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_wave_base.py`:

```python
import numpy as np

from surfanalysis.extraction.wave.base import (
    MockWaveEngine,
    WaveObservation,
    to_wave_metrics,
)


def _obs():
    return WaveObservation(
        crest=(0.52, 0.29),
        base=(0.52, 0.71),
        crest_line=((0.1, 0.30), (0.9, 0.27)),
        face_line=((0.52, 0.71), (0.52, 0.29)),
        bbox=(0.1, 0.27, 0.8, 0.44),
        confidence=0.8,
        horizon_deg=0.0,
    )


def test_to_wave_metrics_facing_uses_crest_line():
    m = to_wave_metrics(_obs(), "facing")
    assert m.view == "facing"
    assert m.angle_kind == "crest_tilt"
    assert m.angle_line == ((0.1, 0.30), (0.9, 0.27))
    assert m.height == _obs().base[1] - _obs().crest[1]


def test_to_wave_metrics_side_uses_face_line():
    m = to_wave_metrics(_obs(), "side")
    assert m.angle_kind == "face_steepness"
    assert m.angle_line == ((0.52, 0.71), (0.52, 0.29))


def test_mock_wave_engine_replays_sequence():
    seq = [to_wave_metrics(_obs(), "facing"), None]
    eng = MockWaveEngine(seq)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert eng.detect(frame, 0.0) is not None
    assert eng.detect(frame, 1.0) is None
    assert eng.detect(frame, 2.0) is None  # past end
    assert eng.info().name == "mock-wave"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_base.py -q`
Expected: FAIL with `ModuleNotFoundError: surfanalysis.extraction.wave`.

- [ ] **Step 3: Implement** — create `src/surfanalysis/extraction/wave/__init__.py` (empty), then `src/surfanalysis/extraction/wave/base.py`:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_base.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/wave/__init__.py src/surfanalysis/extraction/wave/base.py tests/test_wave_base.py
git commit -m "feat(wave): WaveEngine base, WaveObservation, to_wave_metrics, mock"
```

---

### Task 4: Horizon detection (`extraction/wave/horizon.py`)

**Files:**
- Create: `src/surfanalysis/extraction/wave/horizon.py`
- Test: `tests/test_wave_horizon.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_wave_horizon.py`:

```python
import numpy as np

from surfanalysis.extraction.wave.horizon import detect_horizon


def _sky_sea(tilt_rows: int = 0) -> np.ndarray:
    """320x240 BGR: bright sky on top, dark sea below a sharp boundary."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in range(320):
        cut = 120 + int(tilt_rows * (x / 320.0))
        img[:cut, x] = (235, 235, 235)
        img[cut:, x] = (40, 30, 20)
    return img


def test_detect_horizon_flat_is_near_zero():
    ang = detect_horizon(_sky_sea(tilt_rows=0))
    assert ang is not None
    assert abs(ang) < 3.0


def test_detect_horizon_none_on_uniform_frame():
    assert detect_horizon(np.full((240, 320, 3), 80, dtype=np.uint8)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_horizon.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — create `src/surfanalysis/extraction/wave/horizon.py`:

```python
"""Sea-sky / dominant horizontal line detection -> tilt degrees."""

from __future__ import annotations

import math

import cv2
import numpy as np

from surfanalysis.metrics.geometry import wrap_to_180

_MAX_HORIZON_TILT = 25.0  # only accept near-horizontal lines as a horizon


def detect_horizon(frame: np.ndarray) -> float | None:
    """Return the horizon tilt in degrees (vs image-horizontal), or None.

    Picks the longest near-horizontal Hough segment. None means no horizon
    found; callers should then assume image-horizontal (0 deg).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    min_len = frame.shape[1] // 3
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=120, minLineLength=min_len, maxLineGap=20
    )
    if lines is None:
        return None
    best_angle: float | None = None
    best_len = 0.0
    for x1, y1, x2, y2 in lines[:, 0, :]:
        angle = wrap_to_180(math.degrees(math.atan2(float(y2 - y1), float(x2 - x1))))
        if abs(angle) > _MAX_HORIZON_TILT:
            continue
        length = math.hypot(float(x2 - x1), float(y2 - y1))
        if length > best_len:
            best_len = length
            best_angle = angle
    return best_angle
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_horizon.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/wave/horizon.py tests/test_wave_horizon.py
git commit -m "feat(wave): horizon detection (longest near-horizontal Hough line)"
```

---

### Task 5: Global camera motion (`extraction/wave/motion.py`)

**Files:**
- Create: `src/surfanalysis/extraction/wave/motion.py`
- Test: `tests/test_wave_motion.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_wave_motion.py`:

```python
import numpy as np

from surfanalysis.extraction.wave.motion import global_motion


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160), dtype=np.uint8)


def test_global_motion_identical_is_near_zero():
    g = _texture(1)
    assert global_motion(g, g) < 0.5


def test_global_motion_detects_shift():
    g = _texture(2)
    shifted = np.roll(g, 8, axis=1)  # shift 8 px horizontally
    assert global_motion(g, shifted) > 3.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_motion.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — create `src/surfanalysis/extraction/wave/motion.py`:

```python
"""Global camera-motion magnitude between two grayscale frames."""

from __future__ import annotations

import cv2
import numpy as np


def global_motion(prev_gray: np.ndarray, cur_gray: np.ndarray) -> float:
    """Return the global translation magnitude (px) via phase correlation."""
    a = np.float32(prev_gray)
    b = np.float32(cur_gray)
    (dx, dy), _response = cv2.phaseCorrelate(a, b)
    return float((dx * dx + dy * dy) ** 0.5)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_motion.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/wave/motion.py tests/test_wave_motion.py
git commit -m "feat(wave): global camera-motion estimate via phaseCorrelate"
```

---

### Task 6: Geometry from a foreground mask (`extraction/wave/region.py`)

**Files:**
- Create: `src/surfanalysis/extraction/wave/region.py`
- Test: `tests/test_wave_region.py`

- [ ] **Step 1: Write the failing test** — create `tests/test_wave_region.py`:

```python
import numpy as np

from surfanalysis.extraction.wave.region import region_from_mask


def test_region_from_mask_extracts_crest_and_base():
    mask = np.zeros((240, 320), dtype=np.uint8)
    # a wide band from row 60 (top/crest) to row 180 (bottom/base)
    mask[60:180, 40:280] = 255
    obs = region_from_mask(mask, horizon_deg=0.0)
    assert obs is not None
    assert obs.crest[1] < obs.base[1]                  # crest above base
    assert obs.base[1] - obs.crest[1] > 0.3            # spans a real height
    assert 0.0 <= obs.confidence <= 1.0


def test_region_from_mask_none_when_too_small():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[10:14, 10:14] = 255  # tiny speck below min area
    assert region_from_mask(mask) is None


def test_region_from_mask_none_when_empty():
    assert region_from_mask(np.zeros((240, 320), dtype=np.uint8)) is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_region.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — create `src/surfanalysis/extraction/wave/region.py`:

```python
"""Extract wave geometry (crest/base/lines/bbox) from a binary mask."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.wave.base import Line, Point, WaveObservation


def _fit_line(pts: np.ndarray, w: int, h: int) -> Line:
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    xs = pts[:, 0]
    xmin, xmax = float(xs.min()), float(xs.max())

    def at(xq: float) -> Point:
        t = (xq - float(x0)) / float(vx) if abs(float(vx)) > 1e-6 else 0.0
        return (xq / w, (float(y0) + t * float(vy)) / h)

    return (at(xmin), at(xmax))


def region_from_mask(
    mask: np.ndarray, min_area_frac: float = 0.02, horizon_deg: float | None = None
) -> WaveObservation | None:
    """Largest contour -> crest/base/crest_line/face_line/bbox/confidence.

    Returns None when no contour passes the minimum-area gate (this also
    filters out the surfer, who is a much smaller blob than the wave body).
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    if area < min_area_frac * w * h:
        return None
    x, y, bw, bh = cv2.boundingRect(c)
    pts = c.reshape(-1, 2)
    crest = (float(pts[pts[:, 1].argmin(), 0]) / w, float(pts[:, 1].min()) / h)
    base = (float(pts[pts[:, 1].argmax(), 0]) / w, float(pts[:, 1].max()) / h)
    top_band = pts[pts[:, 1] <= y + 0.2 * bh]
    crest_line = (
        _fit_line(top_band, w, h)
        if len(top_band) >= 2
        else ((x / w, y / h), ((x + bw) / w, y / h))
    )
    face_line = (base, crest)
    bbox = (x / w, y / h, bw / w, bh / h)
    confidence = min(1.0, area / (0.25 * w * h))
    return WaveObservation(crest, base, crest_line, face_line, bbox, confidence, horizon_deg)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_region.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/wave/region.py tests/test_wave_region.py
git commit -m "feat(wave): extract wave geometry from a foreground mask"
```

---

## Phase 3 — Engines & pre-scan

### Task 7: Ocean & static engines (`ocean.py`, `static.py`)

**Files:**
- Create: `src/surfanalysis/extraction/wave/ocean.py`
- Create: `src/surfanalysis/extraction/wave/static.py`
- Test: `tests/test_wave_engines.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_wave_engines.py`:

```python
import numpy as np

from surfanalysis.extraction.wave.ocean import HorizonAnchoredWaveEngine, wave_mask
from surfanalysis.extraction.wave.static import Mog2WaveEngine


def _ocean_frame() -> np.ndarray:
    """Bright sky on top, blue-green water band, white foam crest."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:90, :] = (235, 235, 235)            # sky
    img[90:200, :] = (160, 140, 40)          # blue-green water (BGR)
    img[95:115, :] = (250, 250, 250)         # foam crest band
    return img


def test_wave_mask_marks_foam_and_water():
    mask = wave_mask(_ocean_frame(), horizon_deg=0.0)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask[100, 160] == 255             # foam pixel is in the mask


def test_ocean_engine_detects_on_synthetic_wave():
    eng = HorizonAnchoredWaveEngine(view="facing", min_confidence=0.0)
    m = eng.detect(_ocean_frame(), 0.0)
    assert m is not None
    assert m.view == "facing"
    assert 0.0 < m.height <= 1.0
    assert eng.info().name == "wave-ocean"


def test_ocean_engine_none_on_uniform_frame():
    eng = HorizonAnchoredWaveEngine(view="facing", min_confidence=0.5)
    assert eng.detect(np.full((240, 320, 3), 80, dtype=np.uint8), 0.0) is None


def test_static_engine_detects_moving_blob_after_warmup():
    eng = Mog2WaveEngine(view="facing", min_confidence=0.0, warmup=5)
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, size=(240, 320, 3), dtype=np.uint8)  # dark static venue
    result = None
    for i in range(12):
        frame = bg.copy()
        # a bright moving water band sweeping downward each frame
        top = 40 + i * 6
        frame[top:top + 80, 30:290] = 240
        result = eng.detect(frame, float(i))
    assert result is not None          # detects after warmup frames
    assert eng.info().name == "wave-static"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_engines.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3a: Implement** — create `src/surfanalysis/extraction/wave/ocean.py`:

```python
"""Horizon-anchored per-frame wave engine (ocean / moving camera)."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import EngineInfo, WaveMetrics
from surfanalysis.extraction.wave.base import WaveEngine, to_wave_metrics
from surfanalysis.extraction.wave.horizon import detect_horizon
from surfanalysis.extraction.wave.region import region_from_mask

_KERNEL = np.ones((5, 5), np.uint8)


def wave_mask(frame: np.ndarray, horizon_deg: float | None) -> np.ndarray:  # noqa: ARG001
    """Binary mask of foam (bright, low-sat) + blue-green water face."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    foam = (vv > 200) & (ss < 60)
    bluegreen = (hh > 80) & (hh < 140) & (ss > 40)
    mask = ((foam | bluegreen).astype(np.uint8)) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, _KERNEL)


class HorizonAnchoredWaveEngine(WaveEngine):
    def __init__(self, view: str, min_confidence: float = 0.5) -> None:
        self._view = view
        self._min_conf = min_confidence

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None:  # noqa: ARG002
        horizon_deg = detect_horizon(frame)
        mask = wave_mask(frame, horizon_deg)
        obs = region_from_mask(mask, horizon_deg=horizon_deg)
        if obs is None or obs.confidence < self._min_conf:
            return None
        return to_wave_metrics(obs, self._view)

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="wave-ocean",
            version="0.1.0",
            params={"view": self._view, "min_confidence": self._min_conf},
        )
```

- [ ] **Step 3b: Implement** — create `src/surfanalysis/extraction/wave/static.py`:

```python
"""MOG2 background-subtraction wave engine (fixed-camera wave pool)."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import EngineInfo, WaveMetrics
from surfanalysis.extraction.wave.base import WaveEngine, to_wave_metrics
from surfanalysis.extraction.wave.region import region_from_mask

_KERNEL = np.ones((5, 5), np.uint8)


class Mog2WaveEngine(WaveEngine):
    def __init__(self, view: str, min_confidence: float = 0.5, warmup: int = 10) -> None:
        self._view = view
        self._min_conf = min_confidence
        self._warmup = warmup
        self._seen = 0
        self._bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None:  # noqa: ARG002
        fg = self._bg.apply(frame)
        self._seen += 1
        if self._seen <= self._warmup:
            return None
        _thr, binary = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _KERNEL)
        obs = region_from_mask(binary)
        if obs is None or obs.confidence < self._min_conf:
            return None
        return to_wave_metrics(obs, self._view)

    def info(self) -> EngineInfo:
        return EngineInfo(
            name="wave-static",
            version="0.1.0",
            params={
                "view": self._view,
                "min_confidence": self._min_conf,
                "warmup": self._warmup,
            },
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_engines.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/wave/ocean.py src/surfanalysis/extraction/wave/static.py tests/test_wave_engines.py
git commit -m "feat(wave): ocean (horizon-anchored) and static (MOG2) engines"
```

---

### Task 8: Pre-scan + engine factory (`prescan.py`, `__init__.py`)

**Files:**
- Create: `src/surfanalysis/extraction/wave/prescan.py`
- Modify: `src/surfanalysis/extraction/wave/__init__.py`
- Test: `tests/test_wave_prescan.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_wave_prescan.py`:

```python
import numpy as np

from surfanalysis.extraction.wave import make_wave_engine
from surfanalysis.extraction.wave.ocean import HorizonAnchoredWaveEngine
from surfanalysis.extraction.wave.prescan import prescan
from surfanalysis.extraction.wave.static import Mog2WaveEngine


def _ocean_frame(shift: int) -> np.ndarray:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:90, :] = (235, 235, 235)
    img[90:200, :] = (160, 140, 40)
    img[95:115, :] = (250, 250, 250)
    return np.roll(img, shift, axis=1)


def test_prescan_static_when_camera_fixed():
    frames = [_ocean_frame(0) for _ in range(8)]
    engine_name, _view = prescan(frames)
    assert engine_name == "static"


def test_prescan_ocean_when_camera_pans():
    frames = [_ocean_frame(i * 10) for i in range(8)]  # panning
    engine_name, _view = prescan(frames)
    assert engine_name == "ocean"


def test_make_wave_engine_returns_right_type():
    assert isinstance(make_wave_engine("ocean", "facing"), HorizonAnchoredWaveEngine)
    assert isinstance(make_wave_engine("static", "side"), Mog2WaveEngine)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_prescan.py -q`
Expected: FAIL with `ImportError`/`ModuleNotFoundError`.

- [ ] **Step 3a: Implement** — create `src/surfanalysis/extraction/wave/prescan.py`:

```python
"""One pre-scan pass: pick the engine (motion) and the view (geometry)."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.wave.motion import global_motion
from surfanalysis.extraction.wave.ocean import wave_mask
from surfanalysis.extraction.wave.region import region_from_mask
from surfanalysis.metrics.wave_geometry import classify_view

_STATIC_MOTION_PX = 1.5  # median global motion below this => fixed camera


def prescan(frames: list[np.ndarray], n: int = 15) -> tuple[str, str]:
    """Return (engine_name, view) from the first n frames; locked for the clip."""
    sample = frames[:n]
    if len(sample) < 2:
        return ("ocean", "facing")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sample]
    motions = [global_motion(grays[i - 1], grays[i]) for i in range(1, len(grays))]
    engine_name = "static" if float(np.median(motions)) < _STATIC_MOTION_PX else "ocean"

    votes: list[str] = []
    for f in sample:
        obs = region_from_mask(wave_mask(f, None))
        if obs is not None:
            votes.append(classify_view(obs.bbox[2], obs.bbox[3], obs.crest_line))
    view = max(set(votes), key=votes.count) if votes else "facing"
    return (engine_name, view)
```

- [ ] **Step 3b: Implement the factory** — replace `src/surfanalysis/extraction/wave/__init__.py` with:

```python
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
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_prescan.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/wave/prescan.py src/surfanalysis/extraction/wave/__init__.py tests/test_wave_prescan.py
git commit -m "feat(wave): pre-scan engine/view selection + engine factory"
```

---

## Phase 4 — Pipeline integration

### Task 9: `FrameAnalyzer` runs the wave engine

**Files:**
- Modify: `src/surfanalysis/extraction/analyzer.py`
- Test: `tests/test_analyzer.py`

- [ ] **Step 1: Write the failing test** — append to `tests/test_analyzer.py`:

```python
from surfanalysis.extraction.wave.base import MockWaveEngine, WaveObservation, to_wave_metrics


def _wave_metrics():
    obs = WaveObservation(
        crest=(0.5, 0.3), base=(0.5, 0.7),
        crest_line=((0.1, 0.30), (0.9, 0.28)),
        face_line=((0.5, 0.7), (0.5, 0.3)),
        bbox=(0.1, 0.28, 0.8, 0.42), confidence=0.8, horizon_deg=0.0,
    )
    return to_wave_metrics(obs, "facing")


def test_analyzer_populates_wave_and_summary():
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=2, duration_ms=66.0)
    engine = MockEngine(sequence=[_placed_kp(), _placed_kp()])
    wave = MockWaveEngine([_wave_metrics(), None])
    analyzer = FrameAnalyzer(engine=engine, stance="regular", source=src, wave_engine=wave)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
    session = analyzer.run(frames_iter=iter(frames))

    assert session.schema_version == "1.1"
    assert session.frames[0].wave is not None
    assert session.frames[1].wave is None
    assert session.wave_summary is not None
    assert session.wave_summary.frames_detected == 1
    assert session.wave_engine.name == "mock-wave"


def test_analyzer_without_wave_engine_stays_v1_0():
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=1, duration_ms=33.0)
    analyzer = FrameAnalyzer(engine=MockEngine([_placed_kp()]), stance="regular", source=src)
    session = analyzer.run(frames_iter=iter([np.zeros((480, 640, 3), dtype=np.uint8)]))
    assert session.schema_version == "1.0"
    assert session.wave_summary is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_analyzer.py -q`
Expected: FAIL (`FrameAnalyzer` has no `wave_engine` kwarg; `schema_version` is `"1.0"`).

- [ ] **Step 3: Implement** — edit `src/surfanalysis/extraction/analyzer.py`.

Add imports near the top:

```python
from surfanalysis.extraction.wave.base import WaveEngine
from surfanalysis.metrics.wave_geometry import median_p90
```

Replace the module constant block:

```python
SCHEMA_VERSION = "1.0"
```

with:

```python
SCHEMA_VERSION_BASE = "1.0"
SCHEMA_VERSION_WAVE = "1.1"
```

Change `FrameAnalyzer.__init__` to accept the optional engine:

```python
    def __init__(self, engine: PoseEngine, stance: Stance, source: SourceInfo,
                 wave_engine: WaveEngine | None = None) -> None:
        self._engine = engine
        self._stance = stance
        self._source = source
        self._wave_engine = wave_engine
```

In `run()`, compute the wave per frame. Replace the detection/append block inside the loop so BOTH the detected and the `None` branch carry a wave value:

```python
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
```

Replace the `return SessionRecord(...)` block with one that fills the wave fields:

```python
        total = len(frames)
        rate = detections / total if total else 0.0
        version = SCHEMA_VERSION_WAVE if self._wave_engine else SCHEMA_VERSION_BASE
        wave_engine_info = self._wave_engine.info() if self._wave_engine else None
        wave_summary = self._aggregate_wave(frames, wave_engine_info) if self._wave_engine else None
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
    def _aggregate_wave(frames: list[FrameRecord], engine_info) -> "WaveSummary | None":
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
```

Add `WaveSummary` to the schema import block at the top of the file:

```python
from surfanalysis.extraction.schema import (
    FrameRecord,
    Keypoints,
    SessionRecord,
    SessionSummary,
    SourceInfo,
    WaveSummary,
)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_analyzer.py -q`
Expected: PASS (both new tests + the original two).

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/analyzer.py tests/test_analyzer.py
git commit -m "feat(extract): FrameAnalyzer runs optional wave engine, emits 1.1 + summary"
```

---

### Task 10: `extract` CLI flags + pre-scan wiring

**Files:**
- Modify: `src/surfanalysis/cli.py`
- Test: `tests/test_cli_extract.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cli_extract.py`:

```python
def test_extract_wave_adds_wave_fields(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--wave",
         "--wave-engine", "static", "--view", "facing", "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.1"
    assert "wave" in data["frames"][0]          # key present (value may be null)
    assert data["wave_engine"]["name"] == "wave-static"


def test_extract_without_wave_unchanged(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.0"
    assert data["wave_summary"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_extract.py -q`
Expected: FAIL (unrecognized `--wave` argument; exit code 2).

- [ ] **Step 3: Implement** — edit `src/surfanalysis/cli.py`.

Add a helper above `cmd_extract`:

```python
def _build_wave_engine(args: argparse.Namespace, cap: "cv2.VideoCapture"):
    from surfanalysis.extraction.wave import make_wave_engine
    from surfanalysis.extraction.wave.prescan import prescan

    engine_name, view = args.wave_engine, args.view
    if engine_name == "auto" or view == "auto":
        sample: list[np.ndarray] = []
        for _ in range(15):
            ok, fr = cap.read()
            if not ok:
                break
            sample.append(fr)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pe, pv = prescan(sample)
        if engine_name == "auto":
            engine_name = pe
        if view == "auto":
            view = pv
    return make_wave_engine(engine_name, view, args.min_confidence)
```

In `cmd_extract`, after the pose `engine` is constructed and before building the analyzer, add:

```python
    wave_engine = None
    if args.wave:
        try:
            wave_engine = _build_wave_engine(args, cap)
        except Exception as e:
            print(f"error: wave engine init failed: {e}", file=sys.stderr)
            cap.release()
            engine.close()
            return EXIT_ENGINE
```

Pass it to the analyzer:

```python
    analyzer = FrameAnalyzer(engine=engine, stance=args.stance, source=source,
                             wave_engine=wave_engine)
```

In the `finally` block, also close the wave engine:

```python
    finally:
        if progress is not None:
            progress.close()
        cap.release()
        engine.close()
        if wave_engine is not None:
            wave_engine.close()
```

Add the new arguments to the `extract` sub-parser in `_build_parser` (after `--min-confidence`):

```python
    e.add_argument("--wave", action="store_true")
    e.add_argument("--wave-engine", choices=["auto", "ocean", "static"], default="auto")
    e.add_argument("--view", choices=["auto", "facing", "side"], default="auto")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_extract.py -q`
Expected: PASS (new + original tests).

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/cli.py tests/test_cli_extract.py
git commit -m "feat(cli): extract --wave/--wave-engine/--view with pre-scan auto-select"
```

---

## Phase 5 — Rendering

### Task 11: Wave overlay (`rendering/wave_overlay.py`)

**Files:**
- Create: `src/surfanalysis/rendering/wave_overlay.py`
- Test: `tests/test_wave_overlay.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_wave_overlay.py`:

```python
import numpy as np

from surfanalysis.extraction.schema import FrameRecord, WaveMetrics
from surfanalysis.rendering.wave_overlay import WaveOverlay


def _record_with_wave():
    wave = WaveMetrics(
        view="facing", height=0.42, angle_deg=8.0, angle_kind="crest_tilt",
        confidence=0.8, angle_line=((0.1, 0.30), (0.9, 0.27)),
        height_top=(0.5, 0.29), height_bottom=(0.5, 0.71), horizon_deg=0.0,
    )
    return FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None,
                       metrics=None, wave=wave)


def test_wave_overlay_draws_when_wave_present():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = WaveOverlay().draw(blank, _record_with_wave())
    assert out.sum() > 0                       # drew lines + text even with keypoints=None


def test_wave_overlay_noop_without_wave():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    record = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None,
                         metrics=None, wave=None)
    out = WaveOverlay().draw(blank, record)
    assert out.sum() == 0


def test_wave_overlay_pct_changes_label_pixels():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out_norm = WaveOverlay(height_pct=False).draw(blank.copy(), _record_with_wave())
    out_pct = WaveOverlay(height_pct=True).draw(blank.copy(), _record_with_wave())
    assert not np.array_equal(out_norm, out_pct)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_wave_overlay.py -q`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — create `src/surfanalysis/rendering/wave_overlay.py`:

```python
"""Draw wave angle line + height bracket + HUD, decoupled from pose overlay."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import FrameRecord
from surfanalysis.rendering.overlay import hex_to_bgr

_EMA_ALPHA = 0.3


class WaveOverlay:
    def __init__(self, color: str = "#00E5FF", font_scale: float = 0.6,
                 height_pct: bool = False) -> None:
        self._c = hex_to_bgr(color)
        self._fs = font_scale
        self._pct = height_pct
        self._ema_h: float | None = None
        self._ema_a: float | None = None

    def _text(self, frame: np.ndarray, text: str, org: tuple[int, int]) -> None:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._fs, self._c, 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
        w = record.wave
        if w is None:
            return frame
        h, ww = frame.shape[:2]

        def px(p: tuple[float, float]) -> tuple[int, int]:
            return (int(p[0] * ww), int(p[1] * h))

        # angle line (crest tilt for facing / face steepness for side)
        cv2.line(frame, px(w.angle_line[0]), px(w.angle_line[1]), self._c, 2)
        # vertical height bracket
        cv2.line(frame, px(w.height_top), px(w.height_bottom), self._c, 2)

        # EMA-smoothed display values (JSON keeps the raw per-frame numbers)
        self._ema_h = w.height if self._ema_h is None else \
            _EMA_ALPHA * w.height + (1 - _EMA_ALPHA) * self._ema_h
        self._ema_a = w.angle_deg if self._ema_a is None else \
            _EMA_ALPHA * w.angle_deg + (1 - _EMA_ALPHA) * self._ema_a

        label = "tilt" if w.angle_kind == "crest_tilt" else "face"
        htxt = f"{self._ema_h * 100:.0f}%" if self._pct else f"{self._ema_h:.2f}"
        x0 = max(0, ww - 230)
        self._text(frame, f"wave H {htxt}", (x0, 30))
        self._text(frame, f"{label} {self._ema_a:+.0f} deg", (x0, 54))
        return frame
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_wave_overlay.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/rendering/wave_overlay.py tests/test_wave_overlay.py
git commit -m "feat(render): WaveOverlay (angle line + height bracket + HUD)"
```

---

### Task 12: `render` CLI — accept schema 1.x, wire the wave overlay

**Files:**
- Modify: `src/surfanalysis/cli.py`
- Test: `tests/test_cli_render.py`

- [ ] **Step 1: Write the failing tests** — append to `tests/test_cli_render.py`:

```python
def test_render_accepts_schema_1_1_with_wave(tmp_path: Path):
    video = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video), fourcc, 15.0, (320, 240))
    for _ in range(15):
        writer.write(np.full((240, 320, 3), 80, dtype=np.uint8))
    writer.release()

    jpath = tmp_path / "tiny.metrics.json"
    subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract", str(video),
         "-o", str(jpath), "--wave", "--wave-engine", "static", "--quiet"],
        check=True,
    )
    out = tmp_path / "out.mp4"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render", str(video),
         str(jpath), "-o", str(out), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cli_render.py::test_render_accepts_schema_1_1_with_wave -q`
Expected: FAIL — current `cmd_render` rejects any version `!= "1.0"`, returning exit 4.

- [ ] **Step 3: Implement** — edit `src/surfanalysis/cli.py`, function `cmd_render`.

Replace the version check:

```python
    if session.schema_version != "1.0":
        print(f"error: unsupported schema_version {session.schema_version}",
              file=sys.stderr)
        return EXIT_SCHEMA
```

with a major-version check (accepts 1.0 and 1.1, still rejects 99.9):

```python
    if session.schema_version.split(".")[0] != "1":
        print(f"error: unsupported schema_version {session.schema_version}",
              file=sys.stderr)
        return EXIT_SCHEMA
```

Add the wave-overlay import alongside the existing render imports at the top of `cmd_render`:

```python
    from surfanalysis.rendering.overlay import OverlayRenderer
    from surfanalysis.rendering.wave_overlay import WaveOverlay
    from surfanalysis.rendering.writer import VideoSink
```

After the `renderer = OverlayRenderer(...)` construction, build the wave overlay:

```python
    wave_overlay = WaveOverlay(
        color=args.wave_color,
        font_scale=args.font_scale,
        height_pct=args.wave_height_pct,
    ) if args.show_wave else None
```

In the per-frame loop, draw the wave layer after the pose layer:

```python
        for record in session.frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame = renderer.draw(frame, record)
            if wave_overlay is not None:
                frame = wave_overlay.draw(frame, record)
            sink.write(frame)
            if progress is not None:
                progress.update(1)
```

Add the new arguments to the `render` sub-parser in `_build_parser` (after `--weight-color`):

```python
    r.add_argument("--show-wave", action="store_true", default=True)
    r.add_argument("--no-wave", dest="show_wave", action="store_false")
    r.add_argument("--wave-color", type=str, default="#00E5FF")
    r.add_argument("--wave-height-pct", action="store_true")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cli_render.py -q`
Expected: PASS (new test + the original three, including the 99.9 rejection).

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/cli.py tests/test_cli_render.py
git commit -m "feat(cli): render accepts schema 1.x and draws wave overlay"
```

---

## Phase 6 — End-to-end & docs

### Task 13: End-to-end wave test

**Files:**
- Modify: `tests/test_e2e.py`

- [ ] **Step 1: Read the current e2e test** to match its fixture pattern.

Run: `.venv/bin/python -c "print(open('tests/test_e2e.py').read())"`
Expected: see how it builds/locates a clip and asserts on the JSON.

- [ ] **Step 2: Write the failing test** — append to `tests/test_e2e.py` (uses a self-contained synthetic clip so it never depends on `sample/`):

```python
def test_e2e_extract_wave_then_render(tmp_path):
    import json
    import subprocess
    import sys

    import cv2
    import numpy as np

    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, size=(240, 320, 3), dtype=np.uint8)
    for i in range(15):
        frame = bg.copy()
        top = 40 + i * 4
        frame[top:top + 80, 30:290] = 240        # moving bright water band
        writer.write(frame)
    writer.release()

    jpath = tmp_path / "clip.metrics.json"
    r1 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract", str(video),
         "-o", str(jpath), "--wave", "--wave-engine", "static", "--quiet"],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    data = json.loads(jpath.read_text())
    assert data["schema_version"] == "1.1"

    out = tmp_path / "annotated.mp4"
    r2 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render", str(video), str(jpath),
         "-o", str(out), "--quiet"],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 3: Run the full suite + lint + mypy**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (entire suite).
Run: `.venv/bin/ruff check .`
Expected: no errors.
Run: `.venv/bin/python -m mypy src/surfanalysis/metrics`
Expected: `Success: no issues found`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(e2e): extract --wave then render on a synthetic clip"
```

---

### Task 14: Manual validation on the real sample

**Files:** none (manual verification — produces evidence, no code).

- [ ] **Step 1: Run extract with wave on the real sample**

Run:

```bash
.venv/bin/python -m surfanalysis.cli extract sample/sample.MOV \
  -o /tmp/sample.wave.json --wave --wave-engine auto --view auto
```

Expected: exit 0; prints detection rate. Inspect `/tmp/sample.wave.json`:

```bash
.venv/bin/python -c "import json;d=json.load(open('/tmp/sample.wave.json'));print(d['schema_version'], d['wave_engine']['name'], d['wave_summary'])"
```

Expected: `1.1`, an engine name, and a `wave_summary` with `frames_detected > 0`. The sample is a fixed-camera standing wave, so pre-scan should pick `static`; record the actual engine/view chosen.

- [ ] **Step 2: Render and eyeball the overlay**

Run:

```bash
.venv/bin/python -m surfanalysis.cli render sample/sample.MOV /tmp/sample.wave.json -o /tmp/sample.wave.mp4
```

Expected: exit 0, output video exists. Open `/tmp/sample.wave.mp4` and confirm the angle line + height bracket + HUD sit on the wave plausibly. This is the empirical-tuning checkpoint: if the mask/thresholds in `ocean.wave_mask`, `Mog2WaveEngine`, or `classify_view` clearly mis-track, note it (do NOT silently accept a bad result) and adjust thresholds, then re-run.

- [ ] **Step 3: Commit any threshold adjustments** (only if Step 2 required them)

```bash
git add -A
git commit -m "tune(wave): adjust detection thresholds after sample validation"
```

---

### Task 15: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `plans/2026-06-15-wave-analysis-design.md` (lint: add languages to bare code fences)

- [ ] **Step 1: Update `README.md`** — under the Usage section, add a wave-analysis note and extend the usage example:

````markdown
Wave analysis (optional) detects the wave face and emits normalized wave height
(`0-1`) + wave angle (deg) per frame, plus a session median. Enable it with
`--wave`; the engine (`ocean`/`static`) and camera view (`facing`/`side`) auto-select.

```bash
surf extract surf_session.mp4 --wave            # adds wave fields to metrics.json
surf render  surf_session.mp4 surf_session.metrics.json   # draws wave overlay
```
````

- [ ] **Step 2: Update `CLAUDE.md`** — in the Structure section add the new module, and in the Run section add the wave flags:

```markdown
- `extraction/wave/` — WaveEngine (ABC + ocean/static impls), horizon/motion/region helpers, pre-scan
- `metrics/wave_geometry.py` — pure wave geometry (mypy strict)
- `rendering/wave_overlay.py` — wave overlay, decoupled from pose overlay
```

```markdown
- Wave analysis: `surf extract <video> --wave [--wave-engine auto|ocean|static] [--view auto|facing|side]`; `surf render` draws it unless `--no-wave`. Wave metrics are normalized and schema_version becomes "1.1" (render still reads 1.0).
```

- [ ] **Step 3: Fix the spec's bare code fences (markdownlint MD040)** — in `plans/2026-06-15-wave-analysis-design.md`, add `text` after the opening ``` of every fence that has no language (the ASCII diagrams in sections 2, 5, 6, 7, 8, 9 and the formula blocks). Leave ```python / ```json / ```tree / ```mermaid fences as-is.

Run to confirm clean: `npx markdownlint-cli plans/2026-06-15-wave-analysis-design.md` (or your configured linter). Expected: no MD040 warnings. Do NOT run `--fix` (it can rewrite content beyond fences).

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md plans/2026-06-15-wave-analysis-design.md
git commit -m "docs: document wave analysis (README, CLAUDE.md) + lint spec fences"
```

---

## Self-review (completed during planning)

**Spec coverage** — every spec section maps to a task:

| Spec section | Task(s) |
| --- | --- |
| §4 schema v1.1, back-compat | Task 1 |
| §5 view量法 (crest_tilt/face_steepness), classify_view, aggregate | Task 2, Task 3 (`to_wave_metrics`), Task 9 (`_aggregate_wave`) |
| §6 WaveEngine ABC + ocean/static, auto pre-scan, None semantics | Tasks 3, 7, 8; pipeline Task 9 |
| §6 horizon anchor, motion, region | Tasks 4, 5, 6 |
| §7 wave overlay (angle line + bracket + HUD, deg text, EMA, decoupled) | Task 11 |
| §8 CLI flags (`--wave`/`--wave-engine`/`--view`/`--show-wave`/`--wave-color`/`--wave-height-pct`); render accepts major v1; default off | Tasks 10, 12 |
| §9 tests (geometry, engines, prescan, overlay, schema, cli, e2e) | Tasks 1-13 |
| §10 known limits | documented in spec; manual checkpoint Task 14 |

**Placeholder scan** — no `TBD`/`TODO`/"add error handling"; every code step shows complete code.

**Type consistency** — `WaveObservation` (Task 3) field names (`crest`, `base`, `crest_line`, `face_line`, `bbox`, `confidence`, `horizon_deg`) are used identically in `region.py` (Task 6), engines (Task 7), and `to_wave_metrics` (Task 3). `WaveMetrics` field names match between schema (Task 1), `to_wave_metrics` (Task 3), overlay (Task 11), and aggregation (Task 9). `make_wave_engine(name, view, min_confidence)` signature matches its callers in Task 10. `prescan(frames) -> (engine_name, view)` return shape matches its use in Task 10.

**YAGNI note** — the spec listed `crest_tilt_deg` and `face_steepness_deg` as separate functions; they share one formula, so this plan implements a single `angle_vs_horizon_deg` and distinguishes meaning via `angle_kind` (set in `to_wave_metrics`). No behavior lost.
