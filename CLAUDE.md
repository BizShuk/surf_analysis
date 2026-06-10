# CLAUDE.md — surfanalysis

## Project overview

Two-stage CLI for surfing video biomechanical analysis. `extract` runs MediaPipe Pose and emits `metrics.json`; `render` overlays skeleton + CoM + torso-lean angle line + foot-to-foot weight-distribution line + numbers onto the video.

## Structure

- `metrics/` — pure functions, mypy strict required, no I/O
- `extraction/` — PoseEngine (ABC + MediaPipe impl), FrameAnalyzer, Pydantic schema
- `rendering/` — overlay + video writer; depends only on schema
- `cli.py` — argparse subcommands; no business logic

## Tech decisions

- Python 3.11+, MediaPipe Pose (default `model_complexity=1`), OpenCV for I/O and drawing, Pydantic v2 for schema
- Two stages decoupled by JSON contract (`schema_version="1.0"`)
- Strategy pattern on `PoseEngine` for future RTMPose / YOLO-pose swap

## Run

- `surf extract <video> [--stance regular|goofy]` → `<file_name>.metrics.json` next to the video
- `surf render <video> <metrics.json>` → `<file_name>.annotated.<file_extension>` next to the video
- Default output naming is a contract covered by tests in `test_cli_extract.py` / `test_cli_render.py`; `-o` overrides
- Small/distant subjects: `--model-complexity 2 --min-confidence 0.3`
- Without activated venv: `.venv/bin/python -m surfanalysis.cli <subcommand> ...`

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
