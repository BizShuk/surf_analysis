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

## Wave height semantics

現況：`wave_summary.height_median` / `height_p90` 是畫面占比（正規化 0-1），不是物理浪高。命名雖遵循全專案「座標存 0-1」慣例，但與 WSL / 氣象慣例的 `wave height in meters` 語意落差大、誤讀風險高。改用 `wave_height_m`（單位公尺）取代 `height_*` 為目標。

不靠衝浪者當 reference scale 推算物理浪高的可行路徑（依推薦順序）：

1. 相機幾何 + 已知架設高度（one-view metrology,推薦 baseline）
    - 輸入：相機架設高度 `H`（已知）、焦距 `f`（EXIF / 設備規格）、`horizon_deg`（`horizon.py` 已能偵測）。
    - 推算：對畫面 `y` 處的點，深度 `Z(y) = H / tan(θ + arctan((y − cy)/f))`、世界高 `Y(y) = Z(y)·(y − cy)/f`；`wave_height_m = |Y(crest) − Y(trough)|`。
    - 適用：固定機位岸拍、堤防、空拍機。手持 / POV 不適用（`H` 與 `θ` 會隨相機抖動漂移）。
2. 波長–週期色散（純物理）
    - 深水重力波 `L = gT²/(2π)`；碎波 `H/L ≈ 1/7 ~ 1/10` (Miche 準則)。`L` 由 `ocean` 引擎 crest 軌跡間距推得；`T` 由 crest 通過定點的時序推得。
    - 優點：完全不需要相機參數。缺點：`L` 影像 → 世界的轉換仍需方法 1 的相機幾學，或承受 ±20% 的影像 proxy 誤差。
3. 場景已知物體
    - 防鯊網浮球（標準 40–60 cm）、消波塊（2–4 m）、救生員瞭望台 / 旗桿（固定尺寸）。
    - 適用：單一場域長期部署；不適用一般輸入。
4. 結構自運動（SfM）
    - 飄移空拍 / 船拍。COLMAP / OpenCV `stereoCalibrate` 路線。工程量最高。

方法 1 + 2 可交叉驗證：一致 → `confidence: high`；分歧 → `confidence: low`。

實作約束（與現有分層一致）：

- 新公式放 `metrics/wave_geometry.py`（pure functions, mypy strict）。
- I/O 與 `camera_height_m` / `focal_length_mm` 等輸入欄位放 `extraction/`；`prescan.py` 判斷能否計算,失敗時回 `wave_height_m: null` 而非丟誤導的 fraction。
- `rendering/wave_overlay.py` 解耦性已存在,只需新增「物理高度」overlay 分支（顯示 `m` 單位 + 信心標記）。

## Run

- `surf extract <video> [--stance regular|goofy]` → `<file_name>.metrics.json` next to the video
- `surf render <video> <metrics.json>` → `<file_name>.annotated.<file_extension>` next to the video
- Default output naming is a contract covered by tests in `test_cli_extract.py` / `test_cli_render.py`; `-o` overrides
- Small/distant subjects: `--model-complexity 2 --min-confidence 0.3`
- Wave analysis (optional): `surf extract <video> --wave [--wave-engine auto|ocean|static] [--view auto|facing|side]` adds normalized `wave` per frame + `wave_summary`, and bumps `schema_version` to `1.1` (render still reads `1.0`). `surf render` draws it unless `--no-wave`.
    - Wave metrics 與 pose 完全解耦（歸一化、不碰人體關鍵點）。`facing` 量浪唇傾斜 (`crest_tilt`)、`side` 量浪面陡度 (`face_steepness`)，由 `angle_kind` 標明語意。
    - 實測 (sample.MOV 靜止造浪)：`auto` 會選 `static`(MOG2)，但 MOG2 會把「穩定的造浪水流」學進背景而漏偵（偵測率 ~15%）。`--wave-engine ocean`（顏色/泡沫法）在同片偵測率 100% 且 crest 追蹤良好。`靜止造浪池建議明確用 ocean 引擎`；MOG2 適合「固定機位 + 短暫前景」的真實海浪岸拍。
    - ⚠ `wave_summary.height_median` / `height_p90` 目前是「畫面占比」（正規化 0-1），不是物理浪高（公尺）。命名遵循全專案「座標存 0-1」慣例，但易與 WSL / 氣象的 `wave height` 混淆；詳見下方「Wave height semantics」一節。
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
- [docs/pose-engines.md](file:///Users/shuk/projects/surf_analysis/docs/pose-engines.md) — 現代姿態引擎比較 (Modern Pose Engine Comparison)：各引擎選型分析、換引擎檢查清單
