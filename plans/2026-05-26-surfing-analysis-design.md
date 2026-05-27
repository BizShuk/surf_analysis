---
title: Surfing Movement Analysis — Design Spec
date: 2026-05-26
status: approved (pending user final review)
phase: Phase 1 (offline CLI)
---

# Surfing Movement Analysis — Design Spec

## 1. 範圍與目標 (Phase 1)

`目標`：給定一支衝浪影片，輸出一支標註姿態與生物力學指標的影片 + 一份 JSON 原始資料。

`Phase 1 範圍`

- 輸入：單一影片檔 (mp4/mov)
- 輸出：(a) 標註後 mp4，(b) `metrics.json` 每幀指標
- 兩階段 CLI 指令：`extract` 跑 pose、`render` 生成標註影片
- 採 MediaPipe Pose 作為預設 Engine，介面預留給後續實作 (RTMPose/YOLO-pose)
- `不分級`：只輸出原始指標，不做主觀等級判斷

`不在 Phase 1 範圍 (留給後續階段)`

- Phase 2：本機 Web UI (拆出 backend API + 前端 canvas overlay，沿用同一份 JSON 契約)
- Phase 3：即時 webcam/手機鏡頭
- 多人偵測 (假設一支影片一人；若多人取最大 bbox)
- 動作分類 (cutback / bottom turn / aerial)
- 衝浪板偵測

`範圍調整聲明`
原始 user request 為「see what level of surfing」。經 brainstorm 確認：本工具不做主觀等級判斷，僅輸出客觀生物力學指標，由使用者自行解讀。此調整由 user 明確選擇 (對齊運動分析業界標準做法，如 Kinovea/Dartfish)。

---

## 2. 架構總覽

採 `B+C 混合`：兩階段 pipeline (B) 內部使用 Strategy 模式抽象 PoseEngine (C)。

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI (argparse)                           │
│                                                                  │
│   surf extract <video> [--engine mediapipe]  →  metrics.json     │
│   surf render  <video> <metrics.json>        →  annotated.mp4    │
└────────────────────┬───────────────────┬─────────────────────────┘
                     │                   │
                     ▼                   ▼
       ┌─────────────────────┐   ┌────────────────────┐
       │ extraction package  │   │  rendering package │
       │                     │   │                    │
       │  PoseEngine (ABC)   │   │ OverlayRenderer    │
       │  ├─ MediaPipeEngine │   │  ├─ skeleton       │
       │  └─ (future) RTMPose│   │  ├─ CoM marker     │
       │                     │   │  └─ metric labels  │
       │  FrameAnalyzer      │   │                    │
       │  (Engine + Metrics) │   │                    │
       └──────────┬──────────┘   └────────────────────┘
                  │                       ▲
                  ▼                       │
           ┌──────────────┐               │
           │ metrics pkg  │ ──── shared ──┘
           │ (pure funcs) │
           │ CoM, angles, │
           │ weight dist  │
           └──────────────┘
```

`核心隔離原則`

- `metrics` 是純函數 module：吃 keypoints 陣列、吐數字。可獨立測試，零外部相依
- `extraction` 處理 video I/O、引擎呼叫、聚合輸出 JSON
- `rendering` 只吃 JSON + 原始影片，純視覺化，完全不碰 pose 模型
- 兩階段透過 JSON 而非 API 耦合 — JSON 是契約邊界，Phase 2 Web 後端可直接重用

---

## 3. 模組詳細

`專案結構`

```
surfing_analysis/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── docs/
├── plans/
├── src/
│   └── surfanalysis/
│       ├── __init__.py
│       ├── cli.py                ← argparse 主入口
│       ├── extraction/
│       │   ├── engine.py         ← PoseEngine ABC + Keypoints dataclass
│       │   ├── mediapipe_engine.py
│       │   ├── analyzer.py       ← FrameAnalyzer
│       │   ├── landmarks.py      ← MediaPipe 33 索引常數
│       │   └── schema.py         ← Pydantic models
│       ├── metrics/
│       │   ├── geometry.py       ← 向量、角度、投影
│       │   ├── com.py            ← Center of Mass
│       │   ├── weight_dist.py    ← 前後腳承重比例
│       │   ├── angles.py         ← 膝/肘/軀幹/肩髖差動
│       │   └── stability.py      ← 時序穩定度
│       └── rendering/
│           ├── overlay.py        ← OverlayRenderer
│           ├── skeleton.py       ← 33 keypoints 連接邏輯
│           └── writer.py         ← cv2.VideoWriter
└── tests/
    ├── fixtures/
    │   ├── keypoints_tpose.json
    │   ├── keypoints_surfer.json
    │   └── short_clip.mp4
    ├── test_geometry.py
    ├── test_com.py
    ├── test_weight_dist.py
    ├── test_angles.py
    ├── test_stability.py
    ├── test_analyzer.py
    ├── test_overlay.py
    └── test_e2e.py
```

`職責劃分`

- `cli.py`：argument parsing + dispatch，無業務邏輯
- `extraction/engine.py`：`class PoseEngine(ABC): def detect(self, frame: np.ndarray) -> Keypoints | None`
- `extraction/mediapipe_engine.py`：PoseEngine 具體實作，封裝 MediaPipe Pose 呼叫與 visibility 過濾
- `extraction/landmarks.py`：MediaPipe 33 keypoints 具名常數 (`L_HIP = 23` 等)；被 `metrics/*` 與 `rendering/skeleton.py` 共用，避免 magic numbers
- `extraction/analyzer.py`：迭代每幀、呼叫 metrics、組裝 SessionRecord
- `extraction/schema.py`：Pydantic models (Keypoints / FrameRecord / FrameMetrics / SessionRecord)
- `metrics/*`：純函數模組，吃 keypoints 陣列回傳數值，零外部相依
- `rendering/overlay.py`：吃 FrameRecord + 原始 frame → 標註後 frame
- `rendering/skeleton.py`：定義 33 keypoints 之間的連線拓樸 (例如 SHOULDER↔ELBOW↔WRIST)
- `rendering/writer.py`：cv2.VideoWriter 包裝，處理編碼器選擇與寫檔

`資料類型契約`

```python
class Keypoints(BaseModel):
    points: list[tuple[float, float, float, float]]  # x, y, z, visibility
    image_size: tuple[int, int]

class FrameRecord(BaseModel):
    frame_index: int
    timestamp_ms: float
    keypoints: Keypoints | None
    metrics: FrameMetrics | None

class FrameMetrics(BaseModel):
    com: tuple[float, float]
    weight_dist_front_pct: float
    knee_angle_left: float | None
    knee_angle_right: float | None
    elbow_angle_left: float | None
    elbow_angle_right: float | None
    torso_lean_deg: float | None
    shoulder_hip_rotation_deg: float | None
    com_stability_score: float | None
```

`visibility < 0.5` 的 keypoint 在 metrics 計算前過濾為缺失。

---

## 4. 資料流與 JSON Schema

`extract 階段資料流`

```text
mp4 ──cv2.VideoCapture──▶ frame loop
                              │
                              ▼
                       MediaPipeEngine.detect(frame)
                              │
                              ▼
                       Keypoints | None
                              │
                              ▼ (對非 None 幀)
                  metrics.compute_frame_metrics(kp, stance)
                              │
                              ▼
                         FrameRecord
                              │
                              ▼ (累積)
                       SessionRecord ──json.dump──▶ metrics.json
```

`render 階段資料流`

```text
metrics.json ──json.load──▶ SessionRecord
                                │
mp4 ──VideoCapture──▶ frame ────┤
                                ▼
                     OverlayRenderer.draw(frame, FrameRecord)
                                │
                                ▼
                       annotated frame ──VideoWriter──▶ annotated.mp4
```

`JSON 結構 (metrics.json schema)`

```json
{
    "schema_version": "1.0",
    "source": {
        "path": "input.mp4",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "total_frames": 900,
        "duration_ms": 30000
    },
    "engine": {
        "name": "mediapipe",
        "version": "0.10.x",
        "params": { "model_complexity": 1, "min_detection_confidence": 0.5 }
    },
    "stance": "regular",
    "frames": [
        {
            "frame_index": 0,
            "timestamp_ms": 0.0,
            "keypoints": {
                "image_size": [1920, 1080],
                "points": [[0.512, 0.421, -0.03, 0.98], "...(33 points)"]
            },
            "metrics": {
                "com": [0.508, 0.512],
                "weight_dist_front_pct": 58.2,
                "knee_angle_left": 112.3,
                "knee_angle_right": 108.7,
                "elbow_angle_left": 145.2,
                "elbow_angle_right": 138.4,
                "torso_lean_deg": -8.5,
                "shoulder_hip_rotation_deg": 14.3,
                "com_stability_score": 0.91
            }
        }
    ],
    "summary": {
        "frames_with_detection": 872,
        "frames_total": 900,
        "detection_rate": 0.969,
        "metrics_aggregate": {
            "com_x_mean": 0.504,
            "com_x_std": 0.018,
            "knee_angle_left_mean": 115.4
        }
    }
}
```

`schema_version` 用於未來相容性檢查；render 階段若見不識的版本則回傳 exit 4。

`Keypoint 索引慣例`：JSON 中以 0-32 原始順序儲存；模組內部用具名常數 (`L_HIP = 23`) 引用。

---

## 5. 指標數學定義

`MediaPipe 33 keypoints 索引常數` (`extraction/landmarks.py`)

```python
NOSE = 0
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW       = 13, 14
L_WRIST, R_WRIST       = 15, 16
L_HIP, R_HIP           = 23, 24
L_KNEE, R_KNEE         = 25, 26
L_ANKLE, R_ANKLE       = 27, 28
L_FOOT, R_FOOT         = 31, 32
```

`通用過濾`：所有指標計算前，`visibility < 0.5` 的點視為缺失；任一必要點缺失則該指標為 `None`。

`5.1 Center of Mass (com)` — Plagenhoef 分段質量模型

```
segments = [
    (head,         8.1%),   # NOSE
    (trunk,       49.7%),   # midpoint(shoulders, hips)
    (upper_arm_L,  2.8%),   # midpoint(L_SHOULDER, L_ELBOW)
    (upper_arm_R,  2.8%),
    (forearm_L,    1.6%),   # midpoint(L_ELBOW, L_WRIST)
    (forearm_R,    1.6%),
    (thigh_L,     10.0%),   # midpoint(L_HIP, L_KNEE)
    (thigh_R,     10.0%),
    (shank_L,      4.7%),   # midpoint(L_KNEE, L_ANKLE)
    (shank_R,      4.7%),
    (foot_L,       1.4%),   # midpoint(L_ANKLE, L_FOOT)
    (foot_R,       1.4%),
]
com = Σ(segment_pos × weight) / Σ(weight_of_present_segments)
```

若分母 < 0.8 → 回傳 `None`。輸出為 normalized image coords (0-1)。

`5.2 Weight distribution front_pct` — 重心投影到雙腳連線

```
front = L_FOOT if stance == "regular" else R_FOOT
back  = R_FOOT if stance == "regular" else L_FOOT
v = front - back
t = clamp(((com - back) · v) / (v · v), 0, 1)
front_pct = t × 100
```

`stance` 由 CLI flag 指定。Back 上 → 0%；Front 上 → 100%；中點 → 50%。

`5.3 Knee / Elbow angle` — 三點夾角

```
angle(A, B, C):  # 夾角在 B
    v1 = A - B
    v2 = C - B
    cos_θ = (v1 · v2) / (|v1| × |v2|)
    return degrees(acos(clamp(cos_θ, -1, 1)))

knee_angle_left  = angle(L_HIP, L_KNEE, L_ANKLE)
elbow_angle_left = angle(L_SHOULDER, L_ELBOW, L_WRIST)
```

完全伸直 ≈ 180°；直角 ≈ 90°。

`5.4 Torso lean (torso_lean_deg)` — 軀幹相對垂直

```
mid_shoulder = midpoint(L_SHOULDER, R_SHOULDER)
mid_hip      = midpoint(L_HIP, R_HIP)
trunk_vec    = mid_shoulder - mid_hip
torso_lean   = degrees(atan2(trunk_vec.x, -trunk_vec.y))
```

正值 = 前傾；負值 = 後傾。

`5.5 Shoulder-hip differential rotation` — 上下半身扭力差

```
shoulder_angle = degrees(atan2(R_SHOULDER.y - L_SHOULDER.y, R_SHOULDER.x - L_SHOULDER.x))
hip_angle      = degrees(atan2(R_HIP.y - L_HIP.y,        R_HIP.x - L_HIP.x))
diff           = wrap_to_180(shoulder_angle - hip_angle)
```

已知限制：2D projection 會低估真正的 3D 扭轉。未來可選用 MediaPipe z 軸做 3D 校正。

`5.6 CoM stability score` — 滑動窗變異度 (15 frame ≈ 0.5 sec @ 30fps)

```
window = last 15 com values
σ²     = var(window.x) + var(window.y)
score  = 1 / (1 + α × σ²)        # α = 100
```

窗內 < 8 frame 有效 com → `None`。Score ∈ (0, 1]。

---

## 6. CLI 介面與行為

`命令結構`

```
surf extract <video_path> [options]
surf render  <video_path> <metrics_json> [options]
```

`extract subcommand`

```
positional:
  video_path                       輸入影片 (mp4/mov)

options:
  -o, --output PATH                JSON 輸出 (default: <video>.metrics.json)
  --engine {mediapipe}             pose 引擎 (default: mediapipe)
  --stance {regular,goofy}         站姿 (default: regular)
  --model-complexity {0,1,2}       MediaPipe 複雜度 (default: 1)
  --min-confidence FLOAT           偵測門檻 (default: 0.5)
  --max-frames INT                 限制處理幀數 (debug 用)
  --quiet
```

`render subcommand`

```
positional:
  video_path                       原始影片
  metrics_json                     extract 階段輸出的 JSON

options:
  -o, --output PATH                標註後影片 (default: <video>_annotated.mp4)
  --show-secondary                 顯示次要指標 (default: 關)
  --codec {mp4v,avc1}              編碼器 (default: mp4v)
  --font-scale FLOAT               文字大小 (default: 0.6)
  --skeleton-color HEX             骨架線顏色 (default: #00FF00)
  --com-color HEX                  重心點顏色 (default: #FFFF00)
  --quiet
```

`Overlay 分層`

- 主要 (always)：骨架線、CoM 圓點、F/B 承重比文字、左右膝角
- 次要 (`--show-secondary`)：左右肘角、軀幹傾角、肩髖差動、CoM 穩定度

`典型使用流程`

```
$ surf extract surf_session.mp4 --stance regular
[INFO] Loaded video: 1920x1080, 30 fps, 900 frames (30.0s)
[INFO] Initializing MediaPipe Pose (complexity=1)...
[####################] 900/900 frames
[INFO] Detection rate: 87.2% (785/900 frames)
[INFO] Wrote surf_session.metrics.json (1.4 MB)

$ surf render surf_session.mp4 surf_session.metrics.json --show-secondary
[INFO] Rendering with secondary metrics overlay...
[####################] 900/900 frames
[INFO] Wrote surf_session_annotated.mp4 (28.4 MB)
```

`退出碼`

```
0  成功
1  IO 錯誤
2  Engine 載入失敗
3  影片解碼錯誤
4  JSON schema 版本不相容
```

---

## 7. 測試策略與相依套件

`測試金字塔`

```
              ┌──────────────────┐
              │  E2E (1-2 tests) │  小影片 fixture → 比對 JSON 摘要
              └──────────────────┘
            ┌────────────────────────┐
            │ Integration (3-4)      │  Analyzer + MockEngine, Overlay + JSON
            └────────────────────────┘
       ┌────────────────────────────────────┐
       │ Unit (~30+)                        │  metrics 純函數
       │ geometry / com / weight_dist /     │
       │ angles / stability                 │
       └────────────────────────────────────┘
```

`TDD 順序`

1. `test_geometry.py` — `angle_between`, `midpoint`, `project_onto_segment`
2. `test_com.py` — T-pose → 已知 CoM；缺失 segment → None
3. `test_weight_dist.py` — 中點 → 50%；front 上 → 100%；regular/goofy 對稱性
4. `test_angles.py` — 直腿 180°；直角 90°；缺 visibility → None
5. `test_stability.py` — 固定 CoM → score=1.0；震盪 → score < 0.3
6. `test_analyzer.py` — `MockEngine` + 驗 SessionRecord 結構
7. `test_overlay.py` — 已知 FrameRecord → 渲染後 frame shape/pixel 抽樣
8. `test_e2e.py` — short_clip.mp4 → extract → render → 斷言 detection_rate > 0.5

`Fixtures`

- `keypoints_tpose.json` — 標準 T-pose 33 點
- `keypoints_surfer.json` — 模擬衝浪站姿 (膝彎、軀幹前傾)
- `short_clip.mp4` — 3 秒測試影片

`相依套件 (pyproject.toml)`

```toml
[project]
name = "surfanalysis"
version = "0.1.0"
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
    "pytest-cov",
    "ruff",
    "mypy",
]

[project.scripts]
surf = "surfanalysis.cli:main"
```

`工具鏈`

- Python `3.11+`
- 套件管理：`uv` (建議) 或 `pip + venv`
- Linter：`ruff check`
- 型別：`mypy --strict` (重點：`metrics/` 強制過 strict)
- 測試：`pytest -v --cov=surfanalysis`

`效能基準` (informational, 非驗收標準)

- 1080p @ 30fps，MediaPipe complexity=1，CPU inference ~15-25 fps
- 30 秒影片 extract ~40-60 秒；render ~5-10 秒

---

## Open Questions / Future Work

- Phase 2：本機 Web UI 上傳介面 + canvas overlay (sharing JSON schema)
- Phase 3：即時模式 (webcam/手機) — 需考慮幀率管理與較輕量模型 (MediaPipe complexity=0)
- 衝浪板偵測 (獨立 detection 任務，不在 pose pipeline 內)
- 動作分類 (cutback / bottom turn / aerial) — 需標註訓練資料
- 多人場景 (目前單人；多人需 bbox tracking + 人物 ID)
