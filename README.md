# surfanalysis

Biomechanical analysis of surfing video. Loads an mp4, runs MediaPipe Pose to extract 33 keypoints per frame, computes center-of-mass / weight distribution / joint angles, and produces an annotated video plus a metrics JSON.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

Two stages: `extract` runs pose estimation and writes metrics; `render` draws the overlay (skeleton, CoM, torso-lean angle line, foot-to-foot weight-distribution line, numbers).

```bash
# Stage 1: pose extraction
surf extract surf_session.mp4 --stance regular

# Stage 2: annotated video
surf render surf_session.mp4 surf_session.metrics.json --show-secondary
```

Outputs are written next to the input video:

| Command                                  | Default output       |
| ---------------------------------------- | -------------------- |
| `surf extract clip.MOV`                  | `clip.metrics.json`  |
| `surf render clip.MOV clip.metrics.json` | `clip.annotated.MOV` |

Override either path with `-o/--output`. For small or distant subjects (e.g. wave-pool footage shot from the deck), `--model-complexity 2 --min-confidence 0.3` improves detection. Running without an activated venv also works: `.venv/bin/python -m surfanalysis.cli extract ...`.

### Wave analysis (optional)

Detect the wave face and emit normalized wave height (`0-1`) + wave angle (deg) per frame, plus a session median. Enable with `--wave`; the engine (`ocean`/`static`) and camera view (`facing`/`side`) auto-select, or set them explicitly. Adds wave fields to `metrics.json` and bumps `schema_version` to `1.1` (render still reads `1.0`).

```bash
surf extract surf_session.mp4 --wave                      # auto engine + view
surf render  surf_session.mp4 surf_session.metrics.json   # draws wave overlay (use --no-wave to skip)
```

For standing-wave pools, prefer `--wave-engine ocean` (the color/foam detector tracks the steady face well; the MOG2 `static` engine learns steady water into its background and under-detects). The `static` engine suits fixed-camera ocean footage with transient whitewater.

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
