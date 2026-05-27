# Surfing Movement Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python CLI tool `surf` that ingests a surfing video, extracts MediaPipe Pose keypoints into a JSON metrics file, then renders an annotated video showing skeleton, center of mass, and biomechanical numbers.

**Architecture:** Two-stage pipeline (`extract` → JSON → `render`). Inside extract, a Strategy-pattern `PoseEngine` (ABC) lets future swap of pose models without touching rendering. `metrics/` is a pure-function package, fully unit-testable without video or model dependencies.

**Tech Stack:** Python 3.11+, MediaPipe Pose, OpenCV (cv2), NumPy, Pydantic v2, tqdm. Package manager `uv`. Tests with pytest. Lint with ruff, type-check with mypy --strict.

**Spec reference:** [plans/2026-05-26-surfing-analysis-design.md](2026-05-26-surfing-analysis-design.md)

---

## Task 0: Project scaffolding & dependencies

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/surfanalysis/__init__.py`
- Create: `src/surfanalysis/extraction/__init__.py`
- Create: `src/surfanalysis/metrics/__init__.py`
- Create: `src/surfanalysis/rendering/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Initialize git**

Run from `/Users/shuk/projects/surfing_analysis`:
```bash
git init
git branch -m main
```
Expected: `Initialized empty Git repository...`

- [ ] **Step 2: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.mp4
!tests/fixtures/*.mp4
*.metrics.json
.DS_Store
```

- [ ] **Step 3: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "surfanalysis"
version = "0.1.0"
description = "Biomechanical analysis of surfing video"
requires-python = ">=3.11"
dependencies = [
    "mediapipe>=0.10.9",
    "opencv-python>=4.9.0",
    "numpy>=1.26",
    "pydantic>=2.5",
    "tqdm>=4.66",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=4.1",
    "ruff>=0.3",
    "mypy>=1.8",
]

[project.scripts]
surf = "surfanalysis.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["src/surfanalysis"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "B", "UP"]

[tool.mypy]
python_version = "3.11"
strict = true
files = ["src/surfanalysis/metrics"]
```

- [ ] **Step 4: Create empty package `__init__.py` files**

Each of the four `__init__.py` files (`src/surfanalysis/__init__.py`, the three subpackages, and `tests/__init__.py`) is empty.

- [ ] **Step 5: Create `tests/conftest.py` (empty for now, expanded later)**

```python
"""Shared pytest fixtures for surfanalysis tests."""
```

- [ ] **Step 6: Install dev environment**

Run:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```
Expected: `Successfully installed surfanalysis-0.1.0 ...`

- [ ] **Step 7: Verify pytest runs (collects 0 tests)**

Run: `pytest`
Expected: `no tests ran in 0.xxs`

- [ ] **Step 8: Commit**

```bash
git add .gitignore pyproject.toml src tests plans
git commit -m "feat: project scaffolding and dependency manifest"
```

---

## Task 1: Geometry primitives (pure functions)

**Files:**
- Create: `src/surfanalysis/metrics/geometry.py`
- Create: `tests/test_geometry.py`

- [ ] **Step 1: Write failing tests**

`tests/test_geometry.py`:
```python
import math
import pytest
import numpy as np
from surfanalysis.metrics.geometry import (
    midpoint,
    distance,
    angle_at_vertex,
    project_onto_segment,
    wrap_to_180,
)


def test_midpoint_basic():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 4.0])
    assert np.allclose(midpoint(a, b), [1.0, 2.0])


def test_distance_pythagorean():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert distance(a, b) == pytest.approx(5.0)


def test_angle_at_vertex_straight_line():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert angle_at_vertex(a, b, c) == pytest.approx(180.0)


def test_angle_at_vertex_right_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angle_at_vertex(a, b, c) == pytest.approx(90.0)


def test_angle_at_vertex_degenerate_returns_nan_or_zero():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    result = angle_at_vertex(a, b, c)
    assert math.isnan(result)


def test_project_onto_segment_midpoint():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    p = np.array([5.0, 3.0])
    assert project_onto_segment(p, a, b) == pytest.approx(0.5)


def test_project_onto_segment_clamps_below():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    p = np.array([-5.0, 0.0])
    assert project_onto_segment(p, a, b) == pytest.approx(0.0)


def test_project_onto_segment_clamps_above():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    p = np.array([20.0, 0.0])
    assert project_onto_segment(p, a, b) == pytest.approx(1.0)


def test_wrap_to_180_positive_overflow():
    assert wrap_to_180(190.0) == pytest.approx(-170.0)


def test_wrap_to_180_negative_overflow():
    assert wrap_to_180(-190.0) == pytest.approx(170.0)


def test_wrap_to_180_in_range_unchanged():
    assert wrap_to_180(45.0) == pytest.approx(45.0)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_geometry.py -v`
Expected: `ImportError: cannot import name 'midpoint' from 'surfanalysis.metrics.geometry'`

- [ ] **Step 3: Implement `geometry.py`**

`src/surfanalysis/metrics/geometry.py`:
```python
"""Pure geometric primitives operating on 2D numpy points."""

from __future__ import annotations

import math

import numpy as np


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def angle_at_vertex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    return math.degrees(math.acos(cos_theta))


def project_onto_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    denom = float(np.dot(v, v))
    if denom == 0.0:
        return 0.0
    t = float(np.dot(p - a, v) / denom)
    return max(0.0, min(1.0, t))


def wrap_to_180(angle_deg: float) -> float:
    a = (angle_deg + 180.0) % 360.0 - 180.0
    return a if a != -180.0 else 180.0
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_geometry.py -v`
Expected: All 10 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/metrics/geometry.py tests/test_geometry.py
git commit -m "feat(metrics): geometry primitives (midpoint, distance, angle, projection)"
```

---

## Task 2: MediaPipe landmark constants

**Files:**
- Create: `src/surfanalysis/extraction/landmarks.py`

- [ ] **Step 1: Implement `landmarks.py` (no test — it is constants)**

`src/surfanalysis/extraction/landmarks.py`:
```python
"""MediaPipe Pose 33-keypoint named indices.

Reference: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
"""

NOSE = 0
L_EYE_INNER, L_EYE, L_EYE_OUTER = 1, 2, 3
R_EYE_INNER, R_EYE, R_EYE_OUTER = 4, 5, 6
L_EAR, R_EAR = 7, 8
L_MOUTH, R_MOUTH = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32  # foot_index (toe)

TOTAL_LANDMARKS = 33
VISIBILITY_THRESHOLD = 0.5
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from surfanalysis.extraction.landmarks import L_HIP; print(L_HIP)"`
Expected: `23`

- [ ] **Step 3: Commit**

```bash
git add src/surfanalysis/extraction/landmarks.py
git commit -m "feat(extraction): MediaPipe 33-keypoint named constants"
```

---

## Task 3: Pydantic schema models

**Files:**
- Create: `src/surfanalysis/extraction/schema.py`
- Create: `tests/test_schema.py`

- [ ] **Step 1: Write failing tests**

`tests/test_schema.py`:
```python
import pytest
from pydantic import ValidationError

from surfanalysis.extraction.schema import (
    Keypoints,
    FrameMetrics,
    FrameRecord,
    SessionRecord,
    SourceInfo,
    EngineInfo,
    SessionSummary,
)


def _kp_33():
    return [(0.5, 0.5, 0.0, 0.9)] * 33


def test_keypoints_requires_33_points():
    with pytest.raises(ValidationError):
        Keypoints(points=[(0.5, 0.5, 0.0, 0.9)] * 32, image_size=(1920, 1080))


def test_keypoints_accepts_exactly_33():
    kp = Keypoints(points=_kp_33(), image_size=(1920, 1080))
    assert len(kp.points) == 33


def test_frame_record_allows_none_keypoints():
    fr = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None, metrics=None)
    assert fr.keypoints is None


def test_session_record_round_trip_json():
    src = SourceInfo(path="x.mp4", width=1920, height=1080, fps=30.0,
                     total_frames=900, duration_ms=30000.0)
    eng = EngineInfo(name="mediapipe", version="0.10.x",
                     params={"model_complexity": 1, "min_detection_confidence": 0.5})
    summary = SessionSummary(frames_with_detection=0, frames_total=0,
                             detection_rate=0.0, metrics_aggregate={})
    session = SessionRecord(schema_version="1.0", source=src, engine=eng,
                            stance="regular", frames=[], summary=summary)
    json_str = session.model_dump_json()
    restored = SessionRecord.model_validate_json(json_str)
    assert restored.stance == "regular"
    assert restored.source.fps == 30.0


def test_frame_metrics_all_optional_except_com_and_weight():
    fm = FrameMetrics(com=(0.5, 0.5), weight_dist_front_pct=50.0)
    assert fm.knee_angle_left is None


def test_stance_must_be_regular_or_goofy():
    src = SourceInfo(path="x.mp4", width=1, height=1, fps=1.0,
                     total_frames=0, duration_ms=0.0)
    eng = EngineInfo(name="mediapipe", version="x", params={})
    summary = SessionSummary(frames_with_detection=0, frames_total=0,
                             detection_rate=0.0, metrics_aggregate={})
    with pytest.raises(ValidationError):
        SessionRecord(schema_version="1.0", source=src, engine=eng,
                      stance="sideways", frames=[], summary=summary)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_schema.py -v`
Expected: `ImportError` from `surfanalysis.extraction.schema`.

- [ ] **Step 3: Implement `schema.py`**

`src/surfanalysis/extraction/schema.py`:
```python
"""Pydantic models defining the metrics.json contract."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Stance = Literal["regular", "goofy"]


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


class FrameRecord(BaseModel):
    frame_index: int
    timestamp_ms: float
    keypoints: Keypoints | None
    metrics: FrameMetrics | None


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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_schema.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/schema.py tests/test_schema.py
git commit -m "feat(extraction): Pydantic schema for metrics.json contract"
```

---

## Task 4: Center of Mass metric

**Files:**
- Create: `src/surfanalysis/metrics/com.py`
- Create: `tests/test_com.py`

- [ ] **Step 1: Write failing tests**

`tests/test_com.py`:
```python
import numpy as np
import pytest

from surfanalysis.metrics.com import compute_com


def _kp(positions: dict[int, tuple[float, float]], default_vis: float = 0.9):
    """Build a (33, 4) keypoint array; unset indices get visibility 0."""
    arr = np.zeros((33, 4), dtype=np.float64)
    for i, (x, y) in positions.items():
        arr[i] = (x, y, 0.0, default_vis)
    return arr


def test_com_symmetric_tpose_centered():
    """T-pose with all segments present: CoM should be near body centroid."""
    positions = {
        0: (0.5, 0.10),                                # NOSE
        11: (0.4, 0.30), 12: (0.6, 0.30),              # shoulders
        13: (0.3, 0.40), 14: (0.7, 0.40),              # elbows
        15: (0.2, 0.50), 16: (0.8, 0.50),              # wrists
        23: (0.45, 0.55), 24: (0.55, 0.55),            # hips
        25: (0.45, 0.75), 26: (0.55, 0.75),            # knees
        27: (0.45, 0.95), 28: (0.55, 0.95),            # ankles
        31: (0.45, 1.00), 32: (0.55, 1.00),            # feet
    }
    kp = _kp(positions)
    com = compute_com(kp)
    assert com is not None
    assert com[0] == pytest.approx(0.5, abs=0.01)
    assert 0.4 < com[1] < 0.7  # below shoulders, above feet


def test_com_returns_none_when_too_many_segments_missing():
    """Only NOSE visible → not enough mass to compute CoM."""
    kp = _kp({0: (0.5, 0.5)})
    assert compute_com(kp) is None


def test_com_skips_low_visibility_points():
    positions = {
        0: (0.5, 0.10),
        11: (0.4, 0.30), 12: (0.6, 0.30),
        23: (0.45, 0.55), 24: (0.55, 0.55),
        25: (0.45, 0.75), 26: (0.55, 0.75),
        27: (0.45, 0.95), 28: (0.55, 0.95),
        31: (0.45, 1.00), 32: (0.55, 1.00),
    }
    kp = _kp(positions)
    # mark left shoulder as low visibility
    kp[11, 3] = 0.2
    com = compute_com(kp)
    assert com is not None  # still enough mass present
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_com.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `com.py`**

`src/surfanalysis/metrics/com.py`:
```python
"""Center of Mass via Plagenhoef segmental mass approximation."""

from __future__ import annotations

import numpy as np

from surfanalysis.extraction.landmarks import (
    L_ANKLE,
    L_ELBOW,
    L_FOOT,
    L_HIP,
    L_KNEE,
    L_SHOULDER,
    L_WRIST,
    NOSE,
    R_ANKLE,
    R_ELBOW,
    R_FOOT,
    R_HIP,
    R_KNEE,
    R_SHOULDER,
    R_WRIST,
    VISIBILITY_THRESHOLD,
)

_MIN_PRESENT_MASS = 0.8


def _point_if_visible(kp: np.ndarray, i: int) -> np.ndarray | None:
    if kp[i, 3] < VISIBILITY_THRESHOLD:
        return None
    return kp[i, :2].copy()


def _midpoint_if_both_visible(kp: np.ndarray, a: int, b: int) -> np.ndarray | None:
    if kp[a, 3] < VISIBILITY_THRESHOLD or kp[b, 3] < VISIBILITY_THRESHOLD:
        return None
    return (kp[a, :2] + kp[b, :2]) / 2.0


def _trunk_centroid(kp: np.ndarray) -> np.ndarray | None:
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in needed):
        return None
    return (kp[L_SHOULDER, :2] + kp[R_SHOULDER, :2]
            + kp[L_HIP, :2] + kp[R_HIP, :2]) / 4.0


def compute_com(kp: np.ndarray) -> tuple[float, float] | None:
    """Return (com_x, com_y) in normalized image coords, or None if insufficient data.

    kp shape: (33, 4) with columns x, y, z, visibility.
    """
    segments: list[tuple[np.ndarray | None, float]] = [
        (_point_if_visible(kp, NOSE), 0.081),
        (_trunk_centroid(kp), 0.497),
        (_midpoint_if_both_visible(kp, L_SHOULDER, L_ELBOW), 0.028),
        (_midpoint_if_both_visible(kp, R_SHOULDER, R_ELBOW), 0.028),
        (_midpoint_if_both_visible(kp, L_ELBOW, L_WRIST), 0.016),
        (_midpoint_if_both_visible(kp, R_ELBOW, R_WRIST), 0.016),
        (_midpoint_if_both_visible(kp, L_HIP, L_KNEE), 0.100),
        (_midpoint_if_both_visible(kp, R_HIP, R_KNEE), 0.100),
        (_midpoint_if_both_visible(kp, L_KNEE, L_ANKLE), 0.047),
        (_midpoint_if_both_visible(kp, R_KNEE, R_ANKLE), 0.047),
        (_midpoint_if_both_visible(kp, L_ANKLE, L_FOOT), 0.014),
        (_midpoint_if_both_visible(kp, R_ANKLE, R_FOOT), 0.014),
    ]

    total_mass = 0.0
    weighted = np.zeros(2, dtype=np.float64)
    for pos, w in segments:
        if pos is None:
            continue
        weighted += pos * w
        total_mass += w

    if total_mass < _MIN_PRESENT_MASS:
        return None
    com = weighted / total_mass
    return float(com[0]), float(com[1])
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_com.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/metrics/com.py tests/test_com.py
git commit -m "feat(metrics): Plagenhoef segmental center-of-mass estimator"
```

---

## Task 5: Weight distribution metric

**Files:**
- Create: `src/surfanalysis/metrics/weight_dist.py`
- Create: `tests/test_weight_dist.py`

- [ ] **Step 1: Write failing tests**

`tests/test_weight_dist.py`:
```python
import numpy as np
import pytest

from surfanalysis.metrics.weight_dist import compute_weight_dist_front_pct


def _kp_with_feet(l_foot: tuple[float, float], r_foot: tuple[float, float]):
    arr = np.zeros((33, 4), dtype=np.float64)
    arr[31] = (*l_foot, 0.0, 0.9)
    arr[32] = (*r_foot, 0.0, 0.9)
    return arr


def test_com_at_midpoint_returns_50pct():
    kp = _kp_with_feet((0.0, 1.0), (1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(0.5, 1.0), stance="regular")
    assert pct == pytest.approx(50.0, abs=0.5)


def test_com_at_front_foot_returns_100pct_regular():
    """Regular stance: L_FOOT (idx 31) is front."""
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(0.0, 1.0), stance="regular")
    assert pct == pytest.approx(100.0)


def test_com_at_back_foot_returns_0pct_regular():
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(1.0, 1.0), stance="regular")
    assert pct == pytest.approx(0.0)


def test_goofy_swaps_front_back():
    """Goofy stance: R_FOOT (idx 32) is front."""
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct = compute_weight_dist_front_pct(kp, com=(1.0, 1.0), stance="goofy")
    assert pct == pytest.approx(100.0)


def test_com_beyond_segment_clamps():
    kp = _kp_with_feet(l_foot=(0.0, 1.0), r_foot=(1.0, 1.0))
    pct_far = compute_weight_dist_front_pct(kp, com=(-2.0, 1.0), stance="regular")
    assert pct_far == pytest.approx(100.0)


def test_returns_none_when_foot_missing():
    arr = np.zeros((33, 4))  # all zeros, no visibility
    pct = compute_weight_dist_front_pct(arr, com=(0.5, 1.0), stance="regular")
    assert pct is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_weight_dist.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `weight_dist.py`**

`src/surfanalysis/metrics/weight_dist.py`:
```python
"""Estimate front/back foot weight distribution from CoM projection."""

from __future__ import annotations

from typing import Literal

import numpy as np

from surfanalysis.extraction.landmarks import L_FOOT, R_FOOT, VISIBILITY_THRESHOLD
from surfanalysis.metrics.geometry import project_onto_segment

Stance = Literal["regular", "goofy"]


def compute_weight_dist_front_pct(
    kp: np.ndarray,
    com: tuple[float, float],
    stance: Stance,
) -> float | None:
    if kp[L_FOOT, 3] < VISIBILITY_THRESHOLD or kp[R_FOOT, 3] < VISIBILITY_THRESHOLD:
        return None
    if stance == "regular":
        front_idx, back_idx = L_FOOT, R_FOOT
    else:
        front_idx, back_idx = R_FOOT, L_FOOT
    front = kp[front_idx, :2]
    back = kp[back_idx, :2]
    com_np = np.asarray(com, dtype=np.float64)
    # Project: t=0 means at back, t=1 means at front. front_pct = t * 100.
    t = project_onto_segment(com_np, back, front)
    return float(t * 100.0)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_weight_dist.py -v`
Expected: All 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/metrics/weight_dist.py tests/test_weight_dist.py
git commit -m "feat(metrics): front/back weight distribution from CoM projection"
```

---

## Task 6: Joint angles (knee, elbow, torso lean, shoulder-hip differential)

**Files:**
- Create: `src/surfanalysis/metrics/angles.py`
- Create: `tests/test_angles.py`

- [ ] **Step 1: Write failing tests**

`tests/test_angles.py`:
```python
import math
import numpy as np
import pytest

from surfanalysis.metrics.angles import (
    compute_knee_angles,
    compute_elbow_angles,
    compute_torso_lean,
    compute_shoulder_hip_diff,
)


def _kp():
    return np.zeros((33, 4), dtype=np.float64)


def test_knee_straight_returns_180():
    kp = _kp()
    # left leg straight vertical: hip (0.4, 0.5), knee (0.4, 0.7), ankle (0.4, 0.9)
    kp[23] = (0.4, 0.5, 0, 0.9)  # L_HIP
    kp[25] = (0.4, 0.7, 0, 0.9)  # L_KNEE
    kp[27] = (0.4, 0.9, 0, 0.9)  # L_ANKLE
    # right leg straight too
    kp[24] = (0.6, 0.5, 0, 0.9)
    kp[26] = (0.6, 0.7, 0, 0.9)
    kp[28] = (0.6, 0.9, 0, 0.9)
    left, right = compute_knee_angles(kp)
    assert left == pytest.approx(180.0)
    assert right == pytest.approx(180.0)


def test_knee_right_angle():
    kp = _kp()
    kp[23] = (0.4, 0.5, 0, 0.9)  # L_HIP
    kp[25] = (0.4, 0.7, 0, 0.9)  # L_KNEE  (corner)
    kp[27] = (0.6, 0.7, 0, 0.9)  # L_ANKLE (90° bend toward right)
    left, _ = compute_knee_angles(kp)
    assert left == pytest.approx(90.0)


def test_knee_returns_none_when_visibility_low():
    kp = _kp()
    # left leg full, right leg missing
    kp[23] = (0.4, 0.5, 0, 0.9)
    kp[25] = (0.4, 0.7, 0, 0.9)
    kp[27] = (0.4, 0.9, 0, 0.9)
    left, right = compute_knee_angles(kp)
    assert left is not None
    assert right is None


def test_elbow_straight_returns_180():
    kp = _kp()
    kp[11] = (0.4, 0.3, 0, 0.9)  # L_SHOULDER
    kp[13] = (0.3, 0.4, 0, 0.9)  # L_ELBOW
    kp[15] = (0.2, 0.5, 0, 0.9)  # L_WRIST (collinear extension)
    left, _ = compute_elbow_angles(kp)
    assert left == pytest.approx(180.0)


def test_torso_lean_upright_zero():
    kp = _kp()
    kp[11] = (0.4, 0.3, 0, 0.9)  # L_SHOULDER
    kp[12] = (0.6, 0.3, 0, 0.9)  # R_SHOULDER
    kp[23] = (0.4, 0.6, 0, 0.9)  # L_HIP
    kp[24] = (0.6, 0.6, 0, 0.9)  # R_HIP
    lean = compute_torso_lean(kp)
    assert lean == pytest.approx(0.0, abs=0.1)


def test_torso_lean_forward_positive():
    kp = _kp()
    # shoulders shifted forward (smaller x) from hips
    kp[11] = (0.3, 0.3, 0, 0.9)
    kp[12] = (0.5, 0.3, 0, 0.9)
    kp[23] = (0.4, 0.6, 0, 0.9)
    kp[24] = (0.6, 0.6, 0, 0.9)
    lean = compute_torso_lean(kp)
    # shoulder midpoint=(0.4,0.3), hip midpoint=(0.5,0.6), trunk_vec=(-0.1, -0.3)
    # atan2(-0.1, 0.3) ~ -18 deg → negative in our convention
    assert lean < 0


def test_shoulder_hip_diff_zero_when_aligned():
    kp = _kp()
    kp[11] = (0.4, 0.3, 0, 0.9)
    kp[12] = (0.6, 0.3, 0, 0.9)
    kp[23] = (0.4, 0.6, 0, 0.9)
    kp[24] = (0.6, 0.6, 0, 0.9)
    diff = compute_shoulder_hip_diff(kp)
    assert diff == pytest.approx(0.0, abs=0.5)


def test_shoulder_hip_diff_nonzero_when_twisted():
    kp = _kp()
    # shoulders tilted, hips level
    kp[11] = (0.4, 0.28, 0, 0.9)
    kp[12] = (0.6, 0.32, 0, 0.9)
    kp[23] = (0.4, 0.60, 0, 0.9)
    kp[24] = (0.6, 0.60, 0, 0.9)
    diff = compute_shoulder_hip_diff(kp)
    assert abs(diff) > 1.0
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_angles.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `angles.py`**

`src/surfanalysis/metrics/angles.py`:
```python
"""Joint and trunk angle computations."""

from __future__ import annotations

import math

import numpy as np

from surfanalysis.extraction.landmarks import (
    L_ANKLE, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST,
    R_ANKLE, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
    VISIBILITY_THRESHOLD,
)
from surfanalysis.metrics.geometry import angle_at_vertex, midpoint, wrap_to_180


def _angle3(kp: np.ndarray, a: int, b: int, c: int) -> float | None:
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in (a, b, c)):
        return None
    val = angle_at_vertex(kp[a, :2], kp[b, :2], kp[c, :2])
    if math.isnan(val):
        return None
    return val


def compute_knee_angles(kp: np.ndarray) -> tuple[float | None, float | None]:
    left = _angle3(kp, L_HIP, L_KNEE, L_ANKLE)
    right = _angle3(kp, R_HIP, R_KNEE, R_ANKLE)
    return left, right


def compute_elbow_angles(kp: np.ndarray) -> tuple[float | None, float | None]:
    left = _angle3(kp, L_SHOULDER, L_ELBOW, L_WRIST)
    right = _angle3(kp, R_SHOULDER, R_ELBOW, R_WRIST)
    return left, right


def compute_torso_lean(kp: np.ndarray) -> float | None:
    """Angle of trunk vector (mid_hip → mid_shoulder) relative to image-up.

    Positive: shoulders shifted in +x relative to hips (rightward lean in image).
    Negative: leftward.
    """
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in needed):
        return None
    mid_sh = midpoint(kp[L_SHOULDER, :2], kp[R_SHOULDER, :2])
    mid_hp = midpoint(kp[L_HIP, :2], kp[R_HIP, :2])
    trunk = mid_sh - mid_hp
    # image y axis grows downward; upright shoulder is "above" hip, so trunk.y < 0
    return float(math.degrees(math.atan2(trunk[0], -trunk[1])))


def compute_shoulder_hip_diff(kp: np.ndarray) -> float | None:
    needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
    if any(kp[i, 3] < VISIBILITY_THRESHOLD for i in needed):
        return None
    sh_angle = math.degrees(math.atan2(
        kp[R_SHOULDER, 1] - kp[L_SHOULDER, 1],
        kp[R_SHOULDER, 0] - kp[L_SHOULDER, 0],
    ))
    hp_angle = math.degrees(math.atan2(
        kp[R_HIP, 1] - kp[L_HIP, 1],
        kp[R_HIP, 0] - kp[L_HIP, 0],
    ))
    return wrap_to_180(sh_angle - hp_angle)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_angles.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/metrics/angles.py tests/test_angles.py
git commit -m "feat(metrics): knee/elbow/torso/shoulder-hip angle computations"
```

---

## Task 7: CoM stability (rolling-window variance)

**Files:**
- Create: `src/surfanalysis/metrics/stability.py`
- Create: `tests/test_stability.py`

- [ ] **Step 1: Write failing tests**

`tests/test_stability.py`:
```python
import pytest

from surfanalysis.metrics.stability import StabilityWindow


def test_constant_com_yields_score_one():
    win = StabilityWindow(size=15, alpha=100.0)
    for _ in range(15):
        win.push((0.5, 0.5))
    score = win.score()
    assert score == pytest.approx(1.0)


def test_oscillating_com_yields_lower_score():
    win = StabilityWindow(size=15, alpha=100.0)
    for i in range(15):
        x = 0.5 + (0.1 if i % 2 == 0 else -0.1)
        win.push((x, 0.5))
    score = win.score()
    assert 0.0 < score < 0.5


def test_score_none_when_too_few_samples():
    win = StabilityWindow(size=15, alpha=100.0)
    for _ in range(5):
        win.push((0.5, 0.5))
    assert win.score() is None


def test_none_com_does_not_increment_count():
    win = StabilityWindow(size=15, alpha=100.0)
    for _ in range(5):
        win.push((0.5, 0.5))
    for _ in range(20):
        win.push(None)  # missed detections
    # only 5 valid samples remain in the rolling window; below threshold
    assert win.score() is None


def test_window_evicts_oldest():
    win = StabilityWindow(size=3, alpha=100.0)
    win.push((0.0, 0.0))
    win.push((0.0, 0.0))
    win.push((0.0, 0.0))
    win.push((10.0, 10.0))  # evicts the first
    score = win.score()
    assert score is not None
    assert score < 0.99
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_stability.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `stability.py`**

`src/surfanalysis/metrics/stability.py`:
```python
"""Rolling-window CoM stability scoring."""

from __future__ import annotations

from collections import deque

import numpy as np

_MIN_VALID_SAMPLES = 8


class StabilityWindow:
    """Track recent CoM positions and report a stability score in (0, 1]."""

    def __init__(self, size: int = 15, alpha: float = 100.0) -> None:
        self._buf: deque[tuple[float, float] | None] = deque(maxlen=size)
        self._alpha = alpha

    def push(self, com: tuple[float, float] | None) -> None:
        self._buf.append(com)

    def score(self) -> float | None:
        valid = [c for c in self._buf if c is not None]
        if len(valid) < _MIN_VALID_SAMPLES:
            return None
        arr = np.array(valid, dtype=np.float64)
        var = float(np.var(arr[:, 0]) + np.var(arr[:, 1]))
        return 1.0 / (1.0 + self._alpha * var)
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_stability.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/metrics/stability.py tests/test_stability.py
git commit -m "feat(metrics): CoM stability score via rolling-window variance"
```

---

## Task 8: Per-frame metrics aggregator

**Files:**
- Create: `src/surfanalysis/metrics/__init__.py` (overwrite empty stub)
- Create: `tests/test_compute_frame.py`

- [ ] **Step 1: Write failing tests**

`tests/test_compute_frame.py`:
```python
import numpy as np

from surfanalysis.metrics import compute_frame_metrics
from surfanalysis.metrics.stability import StabilityWindow


def _full_kp():
    arr = np.zeros((33, 4), dtype=np.float64)
    placements = {
        0: (0.5, 0.10),
        11: (0.45, 0.30), 12: (0.55, 0.30),
        13: (0.40, 0.40), 14: (0.60, 0.40),
        15: (0.35, 0.50), 16: (0.65, 0.50),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.45, 0.92), 28: (0.55, 0.92),
        31: (0.45, 0.95), 32: (0.55, 0.95),
    }
    for i, (x, y) in placements.items():
        arr[i] = (x, y, 0.0, 0.9)
    return arr


def test_compute_frame_metrics_returns_complete_struct():
    kp = _full_kp()
    win = StabilityWindow()
    fm = compute_frame_metrics(kp, stance="regular", stability_window=win)
    assert fm is not None
    assert 0.0 <= fm.com[0] <= 1.0
    assert 0.0 <= fm.weight_dist_front_pct <= 100.0
    assert fm.knee_angle_left is not None
    assert fm.elbow_angle_left is not None
    assert fm.torso_lean_deg is not None
    assert fm.shoulder_hip_rotation_deg is not None
    # stability needs more samples first
    assert fm.com_stability_score is None


def test_compute_frame_metrics_returns_none_when_com_unavailable():
    kp = np.zeros((33, 4), dtype=np.float64)  # everything invisible
    win = StabilityWindow()
    fm = compute_frame_metrics(kp, stance="regular", stability_window=win)
    assert fm is None
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_compute_frame.py -v`
Expected: `ImportError: cannot import name 'compute_frame_metrics'`.

- [ ] **Step 3: Implement `metrics/__init__.py`**

`src/surfanalysis/metrics/__init__.py`:
```python
"""Public entry point for per-frame metric computation."""

from __future__ import annotations

from typing import Literal

import numpy as np

from surfanalysis.extraction.schema import FrameMetrics
from surfanalysis.metrics.angles import (
    compute_elbow_angles,
    compute_knee_angles,
    compute_shoulder_hip_diff,
    compute_torso_lean,
)
from surfanalysis.metrics.com import compute_com
from surfanalysis.metrics.stability import StabilityWindow
from surfanalysis.metrics.weight_dist import compute_weight_dist_front_pct

Stance = Literal["regular", "goofy"]


def compute_frame_metrics(
    kp: np.ndarray,
    stance: Stance,
    stability_window: StabilityWindow,
) -> FrameMetrics | None:
    com = compute_com(kp)
    if com is None:
        stability_window.push(None)
        return None
    weight_pct = compute_weight_dist_front_pct(kp, com, stance)
    if weight_pct is None:
        stability_window.push(None)
        return None

    stability_window.push(com)
    knee_l, knee_r = compute_knee_angles(kp)
    elbow_l, elbow_r = compute_elbow_angles(kp)
    return FrameMetrics(
        com=com,
        weight_dist_front_pct=weight_pct,
        knee_angle_left=knee_l,
        knee_angle_right=knee_r,
        elbow_angle_left=elbow_l,
        elbow_angle_right=elbow_r,
        torso_lean_deg=compute_torso_lean(kp),
        shoulder_hip_rotation_deg=compute_shoulder_hip_diff(kp),
        com_stability_score=stability_window.score(),
    )


__all__ = ["compute_frame_metrics", "StabilityWindow"]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_compute_frame.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Run full metrics test suite**

Run: `pytest tests/ -v -k "metrics or geometry or com or weight_dist or angles or stability or compute_frame or schema"`
Expected: All tests so far PASS.

- [ ] **Step 6: Type-check metrics package**

Run: `mypy src/surfanalysis/metrics`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add src/surfanalysis/metrics/__init__.py tests/test_compute_frame.py
git commit -m "feat(metrics): per-frame metrics aggregator + mypy strict pass"
```

---

## Task 9: PoseEngine ABC

**Files:**
- Create: `src/surfanalysis/extraction/engine.py`
- Create: `tests/test_engine.py`

- [ ] **Step 1: Write failing tests**

`tests/test_engine.py`:
```python
import numpy as np
import pytest

from surfanalysis.extraction.engine import PoseEngine, MockEngine
from surfanalysis.extraction.schema import Keypoints


def test_pose_engine_is_abstract():
    with pytest.raises(TypeError):
        PoseEngine()  # type: ignore[abstract]


def test_mock_engine_returns_supplied_keypoints():
    kp = Keypoints(points=[(0.5, 0.5, 0.0, 0.9)] * 33, image_size=(640, 480))
    engine = MockEngine(sequence=[kp, None, kp])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert engine.detect(frame) == kp
    assert engine.detect(frame) is None
    assert engine.detect(frame) == kp


def test_mock_engine_reports_metadata():
    engine = MockEngine(sequence=[])
    info = engine.info()
    assert info.name == "mock"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_engine.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `engine.py`**

`src/surfanalysis/extraction/engine.py`:
```python
"""Strategy-pattern pose engine abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from surfanalysis.extraction.schema import EngineInfo, Keypoints


class PoseEngine(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Keypoints | None:
        """Return keypoints for a BGR frame, or None if no person detected."""

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return engine metadata for the JSON output."""


class MockEngine(PoseEngine):
    """Test double: replays a fixed sequence of detections."""

    def __init__(self, sequence: list[Keypoints | None]) -> None:
        self._sequence = sequence
        self._cursor = 0

    def detect(self, frame: np.ndarray) -> Keypoints | None:  # noqa: ARG002
        if self._cursor >= len(self._sequence):
            return None
        out = self._sequence[self._cursor]
        self._cursor += 1
        return out

    def info(self) -> EngineInfo:
        return EngineInfo(name="mock", version="test", params={})
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_engine.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/engine.py tests/test_engine.py
git commit -m "feat(extraction): PoseEngine ABC and MockEngine for tests"
```

---

## Task 10: FrameAnalyzer (orchestration)

**Files:**
- Create: `src/surfanalysis/extraction/analyzer.py`
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_analyzer.py`:
```python
from pathlib import Path

import numpy as np
import pytest

from surfanalysis.extraction.analyzer import FrameAnalyzer
from surfanalysis.extraction.engine import MockEngine
from surfanalysis.extraction.schema import Keypoints, SourceInfo


def _placed_kp():
    pts = [(0.0, 0.0, 0.0, 0.0)] * 33
    placements = {
        0: (0.5, 0.10), 11: (0.45, 0.30), 12: (0.55, 0.30),
        13: (0.40, 0.40), 14: (0.60, 0.40),
        15: (0.35, 0.50), 16: (0.65, 0.50),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.45, 0.92), 28: (0.55, 0.92),
        31: (0.45, 0.95), 32: (0.55, 0.95),
    }
    for i, (x, y) in placements.items():
        pts[i] = (x, y, 0.0, 0.9)
    return Keypoints(points=pts, image_size=(640, 480))


def test_analyzer_assembles_session_record(tmp_path):
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=3, duration_ms=100.0)
    engine = MockEngine(sequence=[_placed_kp(), None, _placed_kp()])
    analyzer = FrameAnalyzer(engine=engine, stance="regular", source=src)
    fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    session = analyzer.run(frames_iter=iter(fake_frames))

    assert session.source.total_frames == 3
    assert len(session.frames) == 3
    assert session.summary.frames_with_detection == 2
    assert session.summary.detection_rate == pytest.approx(2 / 3)
    assert session.frames[0].metrics is not None
    assert session.frames[1].metrics is None
    assert session.engine.name == "mock"


def test_analyzer_writes_json(tmp_path: Path):
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=1, duration_ms=33.0)
    engine = MockEngine(sequence=[_placed_kp()])
    analyzer = FrameAnalyzer(engine=engine, stance="regular", source=src)
    out = tmp_path / "metrics.json"
    session = analyzer.run(frames_iter=iter([np.zeros((480, 640, 3), dtype=np.uint8)]))
    out.write_text(session.model_dump_json(indent=2))
    assert out.read_text().startswith('{')
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_analyzer.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `analyzer.py`**

`src/surfanalysis/extraction/analyzer.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_analyzer.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/analyzer.py tests/test_analyzer.py
git commit -m "feat(extraction): FrameAnalyzer assembling SessionRecord from engine"
```

---

## Task 11: MediaPipe engine implementation

**Files:**
- Create: `src/surfanalysis/extraction/mediapipe_engine.py`
- Create: `tests/test_mediapipe_engine.py`

- [ ] **Step 1: Write failing test (lightweight smoke test)**

`tests/test_mediapipe_engine.py`:
```python
import numpy as np
import pytest

mediapipe = pytest.importorskip("mediapipe")

from surfanalysis.extraction.mediapipe_engine import MediaPipeEngine


def test_mediapipe_engine_returns_none_on_blank_frame():
    engine = MediaPipeEngine(model_complexity=0, min_detection_confidence=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    kp = engine.detect(blank)
    assert kp is None
    engine.close()


def test_mediapipe_engine_info_includes_params():
    engine = MediaPipeEngine(model_complexity=1, min_detection_confidence=0.5)
    info = engine.info()
    assert info.name == "mediapipe"
    assert info.params["model_complexity"] == 1
    engine.close()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_mediapipe_engine.py -v`
Expected: `ImportError` from `surfanalysis.extraction.mediapipe_engine`.

- [ ] **Step 3: Implement `mediapipe_engine.py`**

`src/surfanalysis/extraction/mediapipe_engine.py`:
```python
"""PoseEngine implementation using Google MediaPipe Pose."""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

from surfanalysis.extraction.engine import PoseEngine
from surfanalysis.extraction.schema import EngineInfo, Keypoints


class MediaPipeEngine(PoseEngine):
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._params = {
            "model_complexity": model_complexity,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }
        self._pose = mp.solutions.pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def detect(self, frame: np.ndarray) -> Keypoints | None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return None
        h, w = frame.shape[:2]
        points = [
            (lm.x, lm.y, lm.z, lm.visibility)
            for lm in result.pose_landmarks.landmark
        ]
        return Keypoints(points=points, image_size=(w, h))

    def info(self) -> EngineInfo:
        return EngineInfo(name="mediapipe", version=mp.__version__, params=self._params)

    def close(self) -> None:
        self._pose.close()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_mediapipe_engine.py -v`
Expected: Both tests PASS (may take ~5s on first run due to model load).

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/extraction/mediapipe_engine.py tests/test_mediapipe_engine.py
git commit -m "feat(extraction): MediaPipe Pose engine implementation"
```

---

## Task 12: Skeleton topology

**Files:**
- Create: `src/surfanalysis/rendering/skeleton.py`
- Create: `tests/test_skeleton.py`

- [ ] **Step 1: Write failing tests**

`tests/test_skeleton.py`:
```python
from surfanalysis.rendering.skeleton import SKELETON_EDGES, valid_indices


def test_edges_are_index_pairs():
    for a, b in SKELETON_EDGES:
        assert 0 <= a < 33
        assert 0 <= b < 33
        assert a != b


def test_includes_arm_chain():
    assert (11, 13) in SKELETON_EDGES or (13, 11) in SKELETON_EDGES
    assert (13, 15) in SKELETON_EDGES or (15, 13) in SKELETON_EDGES


def test_includes_leg_chain():
    assert (23, 25) in SKELETON_EDGES or (25, 23) in SKELETON_EDGES
    assert (25, 27) in SKELETON_EDGES or (27, 25) in SKELETON_EDGES


def test_valid_indices_complete():
    assert set(valid_indices()) == set(range(33))
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_skeleton.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `skeleton.py`**

`src/surfanalysis/rendering/skeleton.py`:
```python
"""MediaPipe Pose skeleton connectivity (which keypoints connect to which)."""

from __future__ import annotations

from surfanalysis.extraction.landmarks import (
    L_ANKLE, L_EAR, L_ELBOW, L_EYE, L_EYE_INNER, L_EYE_OUTER,
    L_FOOT, L_HEEL, L_HIP, L_INDEX, L_KNEE, L_MOUTH, L_PINKY,
    L_SHOULDER, L_THUMB, L_WRIST, NOSE,
    R_ANKLE, R_EAR, R_ELBOW, R_EYE, R_EYE_INNER, R_EYE_OUTER,
    R_FOOT, R_HEEL, R_HIP, R_INDEX, R_KNEE, R_MOUTH, R_PINKY,
    R_SHOULDER, R_THUMB, R_WRIST,
    TOTAL_LANDMARKS,
)

SKELETON_EDGES: list[tuple[int, int]] = [
    # face
    (NOSE, L_EYE_INNER), (L_EYE_INNER, L_EYE), (L_EYE, L_EYE_OUTER), (L_EYE_OUTER, L_EAR),
    (NOSE, R_EYE_INNER), (R_EYE_INNER, R_EYE), (R_EYE, R_EYE_OUTER), (R_EYE_OUTER, R_EAR),
    (L_MOUTH, R_MOUTH),
    # arms
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (L_WRIST, L_PINKY), (L_WRIST, L_INDEX), (L_WRIST, L_THUMB), (L_PINKY, L_INDEX),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (R_WRIST, R_PINKY), (R_WRIST, R_INDEX), (R_WRIST, R_THUMB), (R_PINKY, R_INDEX),
    # torso
    (L_SHOULDER, R_SHOULDER),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_HIP, R_HIP),
    # legs
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE), (L_ANKLE, L_HEEL), (L_HEEL, L_FOOT), (L_ANKLE, L_FOOT),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE), (R_ANKLE, R_HEEL), (R_HEEL, R_FOOT), (R_ANKLE, R_FOOT),
]


def valid_indices() -> list[int]:
    return list(range(TOTAL_LANDMARKS))
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_skeleton.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/rendering/skeleton.py tests/test_skeleton.py
git commit -m "feat(rendering): MediaPipe 33-keypoint skeleton edge list"
```

---

## Task 13: Overlay renderer

**Files:**
- Create: `src/surfanalysis/rendering/overlay.py`
- Create: `tests/test_overlay.py`

- [ ] **Step 1: Write failing tests**

`tests/test_overlay.py`:
```python
import numpy as np

from surfanalysis.extraction.schema import FrameMetrics, FrameRecord, Keypoints
from surfanalysis.rendering.overlay import OverlayRenderer, hex_to_bgr


def _frame_record():
    pts = [(0.0, 0.0, 0.0, 0.0)] * 33
    placements = {
        11: (0.45, 0.30), 12: (0.55, 0.30),
        13: (0.40, 0.40), 14: (0.60, 0.40),
        15: (0.35, 0.50), 16: (0.65, 0.50),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.45, 0.92), 28: (0.55, 0.92),
        31: (0.45, 0.95), 32: (0.55, 0.95),
    }
    for i, (x, y) in placements.items():
        pts[i] = (x, y, 0.0, 0.9)
    kp = Keypoints(points=pts, image_size=(640, 480))
    metrics = FrameMetrics(
        com=(0.5, 0.6),
        weight_dist_front_pct=58.2,
        knee_angle_left=112.0,
        knee_angle_right=110.0,
        elbow_angle_left=140.0, elbow_angle_right=138.0,
        torso_lean_deg=-8.5,
        shoulder_hip_rotation_deg=12.3,
        com_stability_score=0.91,
    )
    return FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=kp, metrics=metrics)


def test_hex_to_bgr_basic():
    assert hex_to_bgr("#00FF00") == (0, 255, 0)
    assert hex_to_bgr("FF0000") == (0, 0, 255)


def test_overlay_returns_modified_frame():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = OverlayRenderer()
    out = renderer.draw(blank, _frame_record())
    assert out.shape == blank.shape
    assert out.sum() > 0  # something was drawn


def test_overlay_no_metrics_returns_unchanged():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    record = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None, metrics=None)
    renderer = OverlayRenderer()
    out = renderer.draw(blank, record)
    assert out.sum() == 0


def test_overlay_show_secondary_adds_more_text():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    record = _frame_record()
    r_main = OverlayRenderer(show_secondary=False)
    r_full = OverlayRenderer(show_secondary=True)
    out_main = r_main.draw(blank.copy(), record)
    out_full = r_full.draw(blank.copy(), record)
    assert out_full.sum() > out_main.sum()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_overlay.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `overlay.py`**

`src/surfanalysis/rendering/overlay.py`:
```python
"""Draw skeleton, center-of-mass marker, and metric text onto frames."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import FrameRecord
from surfanalysis.rendering.skeleton import SKELETON_EDGES

VISIBILITY_DRAW = 0.5


def hex_to_bgr(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (b, g, r)


class OverlayRenderer:
    def __init__(
        self,
        skeleton_color: str = "#00FF00",
        com_color: str = "#FFFF00",
        text_color: str = "#FFFFFF",
        font_scale: float = 0.6,
        show_secondary: bool = False,
    ) -> None:
        self._sk = hex_to_bgr(skeleton_color)
        self._com = hex_to_bgr(com_color)
        self._tx = hex_to_bgr(text_color)
        self._font_scale = font_scale
        self._show_secondary = show_secondary

    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
        if record.keypoints is None or record.metrics is None:
            return frame
        h, w = frame.shape[:2]
        pts = [(int(p[0] * w), int(p[1] * h), p[3]) for p in record.keypoints.points]

        # skeleton
        for a, b in SKELETON_EDGES:
            xa, ya, va = pts[a]
            xb, yb, vb = pts[b]
            if va < VISIBILITY_DRAW or vb < VISIBILITY_DRAW:
                continue
            cv2.line(frame, (xa, ya), (xb, yb), self._sk, 2)

        # com marker
        cx = int(record.metrics.com[0] * w)
        cy = int(record.metrics.com[1] * h)
        cv2.circle(frame, (cx, cy), 8, self._com, -1)
        cv2.circle(frame, (cx, cy), 9, (0, 0, 0), 1)

        # primary text block (top-left)
        lines = [
            f"F:{record.metrics.weight_dist_front_pct:.0f}%  B:{100 - record.metrics.weight_dist_front_pct:.0f}%",
        ]
        if record.metrics.knee_angle_left is not None:
            lines.append(f"L knee: {record.metrics.knee_angle_left:.0f} deg")
        if record.metrics.knee_angle_right is not None:
            lines.append(f"R knee: {record.metrics.knee_angle_right:.0f} deg")

        if self._show_secondary:
            if record.metrics.elbow_angle_left is not None:
                lines.append(f"L elbow: {record.metrics.elbow_angle_left:.0f} deg")
            if record.metrics.elbow_angle_right is not None:
                lines.append(f"R elbow: {record.metrics.elbow_angle_right:.0f} deg")
            if record.metrics.torso_lean_deg is not None:
                lines.append(f"torso lean: {record.metrics.torso_lean_deg:+.1f} deg")
            if record.metrics.shoulder_hip_rotation_deg is not None:
                lines.append(f"sh-hip rot: {record.metrics.shoulder_hip_rotation_deg:+.1f} deg")
            if record.metrics.com_stability_score is not None:
                lines.append(f"stability: {record.metrics.com_stability_score:.2f}")

        for i, text in enumerate(lines):
            y = 20 + i * int(20 * self._font_scale * 1.6)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        self._font_scale, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        self._font_scale, self._tx, 1, cv2.LINE_AA)
        return frame
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_overlay.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/rendering/overlay.py tests/test_overlay.py
git commit -m "feat(rendering): overlay renderer with skeleton, CoM, metric text"
```

---

## Task 14: VideoWriter wrapper

**Files:**
- Create: `src/surfanalysis/rendering/writer.py`
- Create: `tests/test_writer.py`

- [ ] **Step 1: Write failing tests**

`tests/test_writer.py`:
```python
from pathlib import Path
import cv2
import numpy as np

from surfanalysis.rendering.writer import VideoSink


def test_video_sink_writes_file(tmp_path: Path):
    out = tmp_path / "out.mp4"
    sink = VideoSink(out, width=320, height=240, fps=15.0, codec="mp4v")
    for _ in range(10):
        sink.write(np.zeros((240, 320, 3), dtype=np.uint8))
    sink.close()
    assert out.exists()
    assert out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_writer.py -v`
Expected: `ImportError`.

- [ ] **Step 3: Implement `writer.py`**

`src/surfanalysis/rendering/writer.py`:
```python
"""Thin wrapper around cv2.VideoWriter with FourCC handling."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoSink:
    def __init__(
        self,
        path: Path | str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise OSError(f"failed to open writer for {path} ({codec})")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_writer.py -v`
Expected: PASS (file is created).

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/rendering/writer.py tests/test_writer.py
git commit -m "feat(rendering): VideoSink wrapping cv2.VideoWriter"
```

---

## Task 15: CLI — extract subcommand

**Files:**
- Create: `src/surfanalysis/cli.py`
- Create: `tests/test_cli_extract.py`

- [ ] **Step 1: Write failing test**

`tests/test_cli_extract.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """Generate a 1-second synthetic 320x240 video at 15 fps."""
    out = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (320, 240))
    for _ in range(15):
        writer.write(np.full((240, 320, 3), 80, dtype=np.uint8))
    writer.release()
    return out


def test_extract_writes_valid_json(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.0"
    assert data["source"]["fps"] == 15.0
    assert len(data["frames"]) == 15
    # blank frames → no detections expected
    assert data["summary"]["frames_total"] == 15


def test_extract_missing_file_returns_exit_1(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tmp_path / "nope.mp4"), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_cli_extract.py -v`
Expected: `ModuleNotFoundError` or similar.

- [ ] **Step 3: Implement `cli.py` (extract path; render placeholder added in next task)**

`src/surfanalysis/cli.py`:
```python
"""Command-line entry point: surf extract / surf render."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from surfanalysis.extraction.analyzer import FrameAnalyzer
from surfanalysis.extraction.mediapipe_engine import MediaPipeEngine
from surfanalysis.extraction.schema import SessionRecord, SourceInfo

EXIT_OK = 0
EXIT_IO = 1
EXIT_ENGINE = 2
EXIT_DECODE = 3
EXIT_SCHEMA = 4


def _open_video(path: Path) -> tuple[cv2.VideoCapture, SourceInfo]:
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cv2 could not open {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = (total / fps) * 1000.0 if fps else 0.0
    return cap, SourceInfo(path=str(path), width=width, height=height,
                            fps=fps, total_frames=total, duration_ms=duration_ms)


def _iter_frames(cap: cv2.VideoCapture, max_frames: int | None,
                 progress: tqdm | None) -> Iterator[np.ndarray]:
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        n += 1
        if progress is not None:
            progress.update(1)
        if max_frames is not None and n >= max_frames:
            break


def cmd_extract(args: argparse.Namespace) -> int:
    video = Path(args.video)
    out_path = Path(args.output) if args.output else video.with_suffix(".metrics.json")
    try:
        cap, source = _open_video(video)
    except FileNotFoundError as e:
        print(f"error: video not found: {e}", file=sys.stderr)
        return EXIT_IO
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_DECODE

    try:
        engine = MediaPipeEngine(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_confidence,
        )
    except Exception as e:  # pragma: no cover
        print(f"error: engine init failed: {e}", file=sys.stderr)
        cap.release()
        return EXIT_ENGINE

    if not args.quiet:
        print(f"[INFO] {source.width}x{source.height}, {source.fps:.2f} fps, "
              f"{source.total_frames} frames")

    analyzer = FrameAnalyzer(engine=engine, stance=args.stance, source=source)
    progress = None if args.quiet else tqdm(total=source.total_frames, unit="frame")
    try:
        session = analyzer.run(_iter_frames(cap, args.max_frames, progress))
    finally:
        if progress is not None:
            progress.close()
        cap.release()
        engine.close()

    out_path.write_text(session.model_dump_json(indent=2))
    if not args.quiet:
        print(f"[INFO] Detection rate: {session.summary.detection_rate:.1%}")
        print(f"[INFO] Wrote {out_path}")
    return EXIT_OK


def cmd_render(args: argparse.Namespace) -> int:
    # Implemented in Task 16
    print("render: not implemented yet", file=sys.stderr)
    return EXIT_OK


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="surf",
                                description="Surfing biomechanical analysis CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="Run pose extraction on a video")
    e.add_argument("video", type=str)
    e.add_argument("-o", "--output", type=str, default=None)
    e.add_argument("--engine", choices=["mediapipe"], default="mediapipe")
    e.add_argument("--stance", choices=["regular", "goofy"], default="regular")
    e.add_argument("--model-complexity", type=int, choices=[0, 1, 2], default=1)
    e.add_argument("--min-confidence", type=float, default=0.5)
    e.add_argument("--max-frames", type=int, default=None)
    e.add_argument("--quiet", action="store_true")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("render", help="Render annotated video from metrics JSON")
    r.add_argument("video", type=str)
    r.add_argument("metrics_json", type=str)
    r.add_argument("-o", "--output", type=str, default=None)
    r.add_argument("--show-secondary", action="store_true")
    r.add_argument("--codec", choices=["mp4v", "avc1"], default="mp4v")
    r.add_argument("--font-scale", type=float, default=0.6)
    r.add_argument("--skeleton-color", type=str, default="#00FF00")
    r.add_argument("--com-color", type=str, default="#FFFF00")
    r.add_argument("--quiet", action="store_true")
    r.set_defaults(func=cmd_render)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_cli_extract.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/cli.py tests/test_cli_extract.py
git commit -m "feat(cli): surf extract subcommand wiring engine + analyzer"
```

---

## Task 16: CLI — render subcommand

**Files:**
- Modify: `src/surfanalysis/cli.py` (replace `cmd_render` stub)
- Create: `tests/test_cli_render.py`

- [ ] **Step 1: Write failing test**

`tests/test_cli_render.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tiny_video_and_json(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video), fourcc, 15.0, (320, 240))
    for _ in range(15):
        writer.write(np.full((240, 320, 3), 80, dtype=np.uint8))
    writer.release()

    json_path = tmp_path / "tiny.metrics.json"
    # Run extract first (cheaper than building JSON by hand)
    subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(video), "-o", str(json_path), "--quiet"],
        check=True,
    )
    return video, json_path


def test_render_produces_output_video(tiny_video_and_json, tmp_path: Path):
    video, jpath = tiny_video_and_json
    out = tmp_path / "out.mp4"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(video), str(jpath), "-o", str(out), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_rejects_unknown_schema_version(tiny_video_and_json, tmp_path: Path):
    video, jpath = tiny_video_and_json
    data = json.loads(jpath.read_text())
    data["schema_version"] = "99.9"
    jpath.write_text(json.dumps(data))
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(video), str(jpath), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 4
```

- [ ] **Step 2: Run tests to verify failure**

Run: `pytest tests/test_cli_render.py -v`
Expected: render returns 0 but produces no file, or schema test fails.

- [ ] **Step 3: Replace `cmd_render` in `cli.py`**

Find `def cmd_render(args: argparse.Namespace) -> int:` in `src/surfanalysis/cli.py` and replace its body with:

```python
def cmd_render(args: argparse.Namespace) -> int:
    from surfanalysis.rendering.overlay import OverlayRenderer
    from surfanalysis.rendering.writer import VideoSink

    video = Path(args.video)
    json_path = Path(args.metrics_json)
    out_path = (
        Path(args.output) if args.output
        else video.with_name(f"{video.stem}_annotated.mp4")
    )

    if not video.exists():
        print(f"error: video not found: {video}", file=sys.stderr)
        return EXIT_IO
    if not json_path.exists():
        print(f"error: metrics json not found: {json_path}", file=sys.stderr)
        return EXIT_IO

    try:
        session = SessionRecord.model_validate_json(json_path.read_text())
    except Exception as e:
        print(f"error: invalid metrics json: {e}", file=sys.stderr)
        return EXIT_SCHEMA

    if session.schema_version != "1.0":
        print(f"error: unsupported schema_version {session.schema_version}",
              file=sys.stderr)
        return EXIT_SCHEMA

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"error: cannot open video {video}", file=sys.stderr)
        return EXIT_DECODE

    sink = VideoSink(out_path, width=session.source.width,
                     height=session.source.height, fps=session.source.fps,
                     codec=args.codec)
    renderer = OverlayRenderer(
        skeleton_color=args.skeleton_color,
        com_color=args.com_color,
        font_scale=args.font_scale,
        show_secondary=args.show_secondary,
    )

    progress = None if args.quiet else tqdm(total=len(session.frames), unit="frame")
    try:
        for record in session.frames:
            ok, frame = cap.read()
            if not ok:
                break
            sink.write(renderer.draw(frame, record))
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        cap.release()
        sink.close()

    if not args.quiet:
        print(f"[INFO] Wrote {out_path}")
    return EXIT_OK
```

- [ ] **Step 4: Run tests to verify pass**

Run: `pytest tests/test_cli_render.py -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/surfanalysis/cli.py tests/test_cli_render.py
git commit -m "feat(cli): surf render subcommand with schema version gate"
```

---

## Task 17: E2E smoke + README + CLAUDE.md

**Files:**
- Create: `tests/test_e2e.py`
- Create: `README.md`
- Create: `CLAUDE.md`

- [ ] **Step 1: Write E2E test**

`tests/test_e2e.py`:
```python
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """Create a 2-second synthetic clip with a moving rectangle (no real pose)."""
    out = tmp_path / "synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(out), fourcc, 15.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        x = 50 + i * 4
        cv2.rectangle(frame, (x, 80), (x + 40, 160), (200, 200, 200), -1)
        w.write(frame)
    w.release()
    return out


def test_e2e_pipeline(synthetic_clip: Path, tmp_path: Path):
    metrics_json = tmp_path / "synth.metrics.json"
    annotated = tmp_path / "synth_annotated.mp4"

    r1 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(synthetic_clip), "-o", str(metrics_json), "--quiet"],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    data = json.loads(metrics_json.read_text())
    assert data["summary"]["frames_total"] == 30

    r2 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(synthetic_clip), str(metrics_json), "-o", str(annotated),
         "--show-secondary", "--quiet"],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert annotated.exists()
    assert annotated.stat().st_size > 0
```

- [ ] **Step 2: Run E2E test**

Run: `pytest tests/test_e2e.py -v`
Expected: PASS (full pipeline executes; synthetic frames yield no detections but pipeline completes).

- [ ] **Step 3: Run full test suite**

Run: `pytest --cov=surfanalysis`
Expected: All tests pass; coverage report prints. Aim for >85% on `metrics/`.

- [ ] **Step 4: Run lint and type-check**

Run: `ruff check . && mypy src/surfanalysis/metrics`
Expected: No issues.

- [ ] **Step 5: Create `README.md`**

```markdown
# surfanalysis

Biomechanical analysis of surfing video. Loads an mp4, runs MediaPipe Pose to extract 33 keypoints per frame, computes center-of-mass / weight distribution / joint angles, and produces an annotated video plus a metrics JSON.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
surf extract surf_session.mp4 --stance regular
surf render surf_session.mp4 surf_session.metrics.json --show-secondary
```

See [plans/2026-05-26-surfing-analysis-design.md](plans/2026-05-26-surfing-analysis-design.md) for the spec and [plans/2026-05-26-surfing-analysis-plan.md](plans/2026-05-26-surfing-analysis-plan.md) for the implementation plan.

## Layout

- `src/surfanalysis/metrics/` — pure-function biomechanics calculations
- `src/surfanalysis/extraction/` — pose engine + analyzer pipeline
- `src/surfanalysis/rendering/` — overlay + video writer
- `src/surfanalysis/cli.py` — `surf extract` / `surf render` entry points

## Tests

```bash
pytest --cov=surfanalysis
```
```

- [ ] **Step 6: Create `CLAUDE.md`**

```markdown
# CLAUDE.md — surfanalysis

## Project overview
Two-stage CLI for surfing video biomechanical analysis. `extract` runs MediaPipe Pose and emits `metrics.json`; `render` overlays skeleton + CoM + numbers onto the video.

## Structure

- `metrics/` — pure functions, mypy strict required, no I/O
- `extraction/` — PoseEngine (ABC + MediaPipe impl), FrameAnalyzer, Pydantic schema
- `rendering/` — overlay + video writer; depends only on schema
- `cli.py` — argparse subcommands; no business logic

## Tech decisions
- Python 3.11+, MediaPipe Pose (default `model_complexity=1`), OpenCV for I/O and drawing, Pydantic v2 for schema
- Two stages decoupled by JSON contract (`schema_version="1.0"`)
- Strategy pattern on `PoseEngine` for future RTMPose / YOLO-pose swap

## Build / test
- Install: `pip install -e ".[dev]"`
- Run tests: `pytest`
- Lint: `ruff check .`
- Type-check (metrics only, strict): `mypy src/surfanalysis/metrics`

## Conventions
- All metrics return `None` when required keypoints have `visibility < 0.5`
- Keypoint coordinates stored normalized (0-1) throughout
- No magic numbers: indices live in `extraction/landmarks.py`, mass coefficients in `metrics/com.py`

## Out of scope (see Phase 2/3 in spec)
- Real-time webcam input
- Web UI
- Multi-person tracking
- Maneuver classification (cutback, bottom turn, aerial)
- Surfboard detection
```

- [ ] **Step 7: Commit**

```bash
git add tests/test_e2e.py README.md CLAUDE.md
git commit -m "feat: E2E smoke test + README + CLAUDE.md"
```

- [ ] **Step 8: Tag v0.1.0**

```bash
git tag -a v0.1.0 -m "Phase 1: surfing analysis CLI (extract + render)"
```

---

## Manual Verification Checklist (after Task 17)

Beyond automated tests, verify the tool actually works on a real surfing clip:

- [ ] Place a real surfing mp4 at `/tmp/surf_sample.mp4`
- [ ] Run: `surf extract /tmp/surf_sample.mp4 --stance regular`
- [ ] Open `/tmp/surf_sample.metrics.json` — inspect a non-None metrics frame, confirm CoM in (0, 1)² and weight_dist_front_pct in [0, 100]
- [ ] Run: `surf render /tmp/surf_sample.mp4 /tmp/surf_sample.metrics.json --show-secondary`
- [ ] Open `/tmp/surf_sample_annotated.mp4` and watch — skeleton tracks the surfer, CoM dot sits near torso/hip area, text legible
- [ ] Detection rate above 50%? If significantly lower, surfer may be too small in frame or occluded — note in follow-up issue
