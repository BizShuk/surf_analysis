# surfanalysis

Biomechanical analysis of surfing video. Loads an mp4, runs MediaPipe Pose to extract 33 keypoints per frame, computes center-of-mass / weight distribution / joint angles, and produces an annotated video plus a metrics JSON.

## Status & TODO

Current state (`2026-06-21`): both stages work end-to-end on the `sample/` wave-pool clip. Quality gates are green — `142` tests pass, `ruff` clean, `mypy --strict` clean on `metrics/`.

Working:

- Two-stage pipeline (`extract` → `metrics.json` → `render` → annotated `.mp4`), decoupled by the JSON contract.
- Pose in MediaPipe `VIDEO` mode (temporal tracking; `49.6%` → `94.5%` detection on the sample vs `IMAGE`).
- Metrics: CoM, foot-to-foot weight distribution, torso-lean angle, joint angles, stability score — all `None`-gated on `visibility < 0.5`.
- Wave analysis (`schema_version 1.2`): `ocean` color/foam engine reaches `99.5%` detection on the standing-wave sample. **Schema 1.1 is rejected** with `IncompatibleSchemaError` — see `CLAUDE.md` § "Wave height semantics".
- **Physical wave height in meters** (optional, schema 1.2): pass `--camera-height-m` + `--focal-length-mm` (or `--sensor-height-mm`) to derive per-frame crest/base world coordinates via pinhole projection. **No surfer reference scale is used.** The CLI prints a stderr warning when wave is on but metadata is missing.

Known issues / to improve:

- `Wave engine auto-select is mis-tuned`. `prescan` keys on camera motion, so fixed-camera pools get `static` (MOG2), which learns the steady wave into its background (`~5.7%` detection). Workaround is explicit `--wave-engine ocean`; the real fix is a foreground-stability heuristic, not a camera-motion one.
- **`--camera-pitch-deg` works as a boolean gate, not a knob**. The height formula `h = H × (1 - tan(α_crest)/tan(α_base))` cancels pitch, so any pitch > ~0.5° produces the same answer. For top-down shots where the wave's crest image points are above the horizon at pitch=0, pass `--camera-pitch-deg 45` (or similar) to unlock the projection.
- `Type coverage is partial`. `mypy --strict` runs on `metrics/` only; `extraction/` and `rendering/` are unchecked.
- `Engine swap is blocked on keypoint format`. RTMPose / YOLO-pose output `COCO-17` (no heel / foot_index), which the weight-distribution and CoM metrics depend on. See [docs/pose-engines.md](docs/pose-engines.md) for the swap checklist.
- `No real-ocean footage validated`. Only the wave-pool sample and a synthetic clip are exercised; ocean shore-break behavior of the `static` engine is untested on real data.
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

Detect the wave face and emit wave angle (deg) per frame + physical wave height in meters (when camera metadata is supplied), plus a session summary. Enable with `--wave`; the engine (`ocean`/`static`) and camera view (`facing`/`side`) auto-select, or set them explicitly. Adds wave fields to `metrics.json` and bumps `schema_version` to `1.2`.

```bash
surf extract surf_session.mp4 --wave                                                    # auto engine + view
surf extract surf_session.mp4 --wave --camera-height-m 2.5 --camera-pitch-deg 45 \
                              --focal-length-mm 26 --sensor-height-mm 4                # adds physical wave height (m)
surf render  surf_session.mp4 surf_session.metrics.json                                # draws wave overlay (use --no-wave to skip)
```

For standing-wave pools, prefer `--wave-engine ocean` (the color/foam detector tracks the steady face well; the MOG2 `static` engine learns steady water into its background and under-detects). The `static` engine suits fixed-camera ocean footage with transient whitewater.

For physical wave height, pass `--camera-pitch-deg` when the camera is not roughly horizontal (top-down pool shots, drone footage, etc.). The flag is a boolean gate — any pitch > ~0.5° unlocks the projection; the height value is independent of pitch in the 15-90° range.

See [plans/2026-05-26-surfing-analysis-design.md](plans/2026-05-26-surfing-analysis-design.md) for the spec and [plans/2026-05-26-surfing-analysis-plan.md](plans/2026-05-26-surfing-analysis-plan.md) for the implementation plan. Physical wave height: [plans/2026-06-21-physical-wave-height-design.md](plans/2026-06-21-physical-wave-height-design.md).

## Layout

- `src/surfanalysis/metrics/` — pure-function biomechanics calculations (mypy strict)
- `src/surfanalysis/extraction/` — pose engine + analyzer pipeline (incl. `wave/` subpackage: ocean/static engines, `CameraModel`, `PhysicalWaveComputer`, `WavelengthEstimator`)
- `src/surfanalysis/rendering/` — overlay + video writer
- `cli.py` — `surf extract` / `surf render` entry points

## Tests

```bash
pytest --cov=surfanalysis
```
