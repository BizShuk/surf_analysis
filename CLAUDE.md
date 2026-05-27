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
