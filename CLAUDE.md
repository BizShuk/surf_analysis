# CLAUDE.md — surfanalysis

## Project overview

Two-stage CLI for surfing video biomechanical analysis. `extract` runs MediaPipe Pose and emits `metrics.json`; `render` overlays skeleton + CoM + torso-lean angle line + foot-to-foot weight-distribution line + numbers onto the video.

## Structure

- `metrics/` — pure functions, mypy strict required, no I/O (incl. `wave_geometry.py`)
- `extraction/` — PoseEngine (ABC + MediaPipe impl), FrameAnalyzer, Pydantic schema
- `extraction/wave/` — WaveEngine (ABC + `ocean`/`static` impls), horizon/motion/region helpers, pre-scan (engine+view auto-select)
- `rendering/` — overlay + video writer; depends only on schema (`wave_overlay.py` decoupled from pose overlay)
- `cli.py` — argparse subcommands; no business logic

## Tech decisions

- Python 3.11+, MediaPipe Pose (default `model_complexity=1`), OpenCV for I/O and drawing, Pydantic v2 for schema
- Two stages decoupled by JSON contract (`schema_version="1.0"`)
- Strategy pattern on `PoseEngine` for future RTMPose / YOLO-pose swap
- MediaPipe Pose 必須跑在 `VisionTaskRunningMode.VIDEO` + `detect_for_video()`（非 `IMAGE`）。
    - `IMAGE` 模式逐幀重偵測、無時序追蹤，且忽略 `min_tracking_confidence`，導致小/快速移動主體標註閃爍（sample 實測：偵測率 49.6%、114 次 on/off）。
    - `VIDEO` 模式以前一幀姿態當追蹤先驗，跨越短暫漏偵（同樣 sample：94.5%、6 次 on/off）。
    - `detect_for_video` 要求嚴格遞增的整數毫秒 timestamp；引擎內建單調守衛（`ts <= last → last+1`）吸收重複/回退的時間戳。
    - `PoseEngine.detect(frame, timestamp_ms)` 契約已加 `timestamp_ms` 參數，由 `FrameAnalyzer` 傳入。
- 渲染分層：骨架（偵測證據）與衍生指標（CoM/重心分布）解耦。
    - `overlay.draw()` 只要有 keypoints 就畫骨架；CoM、重心線、傾角線、文字才 gate 在 `metrics is None`。
    - 避免某隻腳 visibility < 0.5 造成整幀標註消失。

兩個重點摘要關係：

- 抽取層 (mediapipe_engine.py) — IMAGE → VIDEO，解決偵測閃爍
- 渲染層 (overlay.py) — 骨架/指標解耦，解決偵測到卻不畫的空窗

## Run

- `surf extract <video> [--stance regular|goofy]` → `<file_name>.metrics.json` next to the video
- `surf render <video> <metrics.json>` → `<file_name>.annotated.<file_extension>` next to the video
- Default output naming is a contract covered by tests in `test_cli_extract.py` / `test_cli_render.py`; `-o` overrides
- Small/distant subjects: `--model-complexity 2 --min-confidence 0.3`
- Wave analysis (optional): `surf extract <video> --wave [--wave-engine auto|ocean|static] [--view auto|facing|side]` adds normalized `wave` per frame + `wave_summary`, and bumps `schema_version` to `1.1` (render still reads `1.0`). `surf render` draws it unless `--no-wave`.
    - Wave metrics 與 pose 完全解耦（歸一化、不碰人體關鍵點）。`facing` 量浪唇傾斜 (`crest_tilt`)、`side` 量浪面陡度 (`face_steepness`)，由 `angle_kind` 標明語意。
    - 實測 (sample.MOV 靜止造浪)：`auto` 會選 `static`(MOG2)，但 MOG2 會把「穩定的造浪水流」學進背景而漏偵（偵測率 ~15%）。`--wave-engine ocean`（顏色/泡沫法）在同片偵測率 100% 且 crest 追蹤良好。`靜止造浪池建議明確用 ocean 引擎`；MOG2 適合「固定機位 + 短暫前景」的真實海浪岸拍。
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

## References

- [video_format_decision.md](file:///Users/shuk/projects/surf_analysis/video_format_decision.md) — 影片格式決策 (Video Format Decision) 指南
- [golang-replacement.md](file:///Users/shuk/projects/surf_analysis/plans/golang-replacement.md) — Go 語言替代可行性評估與衝浪分析實作日誌 (Go Replacement Feasibility Assessment and Implementation Log)
- [docs/tutorials/pose-biomechanics-tutorial.md](file:///Users/shuk/projects/surf_analysis/docs/tutorials/pose-biomechanics-tutorial.md) — 衝浪姿態與生物力學分析教學 (Pose & Biomechanics Tutorial)：背景知識、系統運作、指標公式與調校
