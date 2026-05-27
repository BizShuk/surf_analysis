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