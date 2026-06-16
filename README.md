# surfanalysis

Biomechanical analysis of surfing video. Loads an mp4, runs MediaPipe Pose to extract 33 keypoints per frame, computes center-of-mass / weight distribution / joint angles, and produces an annotated video plus a metrics JSON.

## Status & TODO

Current state (`2026-06-16`): both stages work end-to-end on the `sample/` wave-pool clip. Quality gates are green — `107` tests pass, `ruff` clean, `mypy --strict` clean on `metrics/`.

Working:

- Two-stage pipeline (`extract` → `metrics.json` → `render` → annotated `.mp4`), decoupled by the JSON contract.
- Pose in MediaPipe `VIDEO` mode (temporal tracking; `49.6%` → `94.5%` detection on the sample vs `IMAGE`).
- Metrics: CoM, foot-to-foot weight distribution, torso-lean angle, joint angles, stability score — all `None`-gated on `visibility < 0.5`.
- Wave analysis (`schema_version 1.1`): `ocean` color/foam engine reaches `99.5%` detection on the standing-wave sample.

Known issues / to improve:

- `Wave engine auto-select is mis-tuned`. `prescan` keys on camera motion, so fixed-camera pools get `static` (MOG2), which learns the steady wave into its background (`~5.7%` detection). Workaround is explicit `--wave-engine ocean`; the real fix is a foreground-stability heuristic, not a camera-motion one.
- `Type coverage is partial`. `mypy --strict` runs on `metrics/` only; `extraction/` and `rendering/` are unchecked.
- `Engine swap is blocked on keypoint format`. RTMPose / YOLO-pose output `COCO-17` (no heel / foot_index), which the weight-distribution and CoM metrics depend on. See [docs/pose-engines.md](docs/pose-engines.md) for the swap checklist.
- `No real-ocean footage validated`. Only the wave-pool sample and a synthetic clip are exercised; ocean shore-break behavior of the `static` engine is untested on real data.
- `Render reads the 1.0 contract loosely`. It tolerates `1.x` but does not strictly validate the wave-specific `1.1` fields.
- `2D only`. No 3D joint angles or true torso rotation; the MediaPipe `z` is pseudo-3D and unused.

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
| `surf render clip.MOV clip.metrics.json` | `clip.annotated.mp4` |

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
