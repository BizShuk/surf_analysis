---
title: Wave Analysis — Design Spec (wave height / wave angle)
date: 2026-06-15
status: approved (pending user final review)
phase: extends Phase 1 (offline CLI)
schema_version: 1.1
---

# Wave Analysis — Design Spec

## 1. 範圍與目標

`目標`：在既有兩階段 CLI 上新增「浪面分析 (wave analysis)」，從場景影像偵測浪面，輸出每幀的 `浪高 (wave height)` 與 `浪面角度 (wave angle)`，並在標註影片上畫出對應的線與數值。

`本次範圍`

- 在 `extract` 階段新增可選的浪面偵測 pass，寫入 `metrics.json`（新欄位、向後相容）
- 在 `render` 階段新增浪面 overlay（線 + 數值），與既有 pose overlay 解耦
- 支援兩種攝影機視角：`facing`（正對浪面）與 `side`（側拍剖面），以 `--view auto` 自動判別
- 引擎採 Strategy 模式雙實作：`ocean`（海浪/會動攝影機）與 `static`（造浪池/固定機位），以 `--wave-engine auto` 自動判別
- 浪高/浪角採 `歸一化 (normalized)` 表示，與 pose 資料完全解耦
- 每幀輸出時間序列 + session 層級穩健聚合 (median / p90)

`不在本次範圍`

- 絕對尺度換算（公尺）：本次只輸出歸一化值，不做 scale calibration
- 剝離角 (peel angle)、浪週期 (wave period)、多浪追蹤
- 立體/多機位重建（單機位無法還原朝向深度的浪面陡度，已知限制）
- 即時模式

`需求決策來源`
原始 user request 為「wave analysis for wave height, wave angle」。經 brainstorm 逐項確認：資料來源=場景抽浪面 (segmentation)；偵測法=以地平線為錨的混合法；單位=歸一化；粒度=每幀+session 聚合；引擎=Strategy 雙引擎；視角=facing+side 且 `--view auto`；數值顯示=線上幾何 + 右上 HUD。各項皆由 user 明確選擇。

---

## 2. 架構總覽

浪面分析沿用 repo 既有的隔離原則，並與 pose pipeline `平行` 而非耦合：

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLI (argparse)                           │
│   surf extract <video> [--wave] [--wave-engine auto] [--view auto]│
│   surf render  <video> <metrics.json> [--show-wave]              │
└──────────────┬──────────────────────────────┬────────────────────┘
               │                              │
               ▼                              ▼
   ┌───────────────────────────┐   ┌──────────────────────────┐
   │ extraction package        │   │  rendering package       │
   │                           │   │                          │
   │  PoseEngine (ABC)  ──┐    │   │ OverlayRenderer (pose)   │
   │  WaveEngine (ABC) ───┤    │   │ WaveOverlay (新)         │
   │   ├ HorizonAnchored  │    │   │  ├ angle_line            │
   │   │   (ocean)        │    │   │  ├ height bracket        │
   │   └ Mog2WaveEngine   │    │   │  └ HUD (EMA + median)    │
   │       (static)       │    │   │                          │
   │  FrameAnalyzer ──────┘    │   │                          │
   └───────────┬───────────────┘   └──────────────────────────┘
               │                              ▲
               ▼                              │
       ┌───────────────────┐                 │
       │ metrics package   │ ──── shared ─────┘
       │ wave_geometry.py  │   (純函數：角度/歸一化/分類/聚合)
       └───────────────────┘
```

`核心隔離原則（沿用既有 + 延伸）`

- 浪面與 pose 解耦：浪面引擎 `不碰人體關鍵點`；衝浪者汙染前景靠「最大輪廓/面積過濾」排除，不靠 pose bbox
- 有時序狀態的 CV（背景模型、光流、前幀）歸在 `extraction/wave/`；純幾何（角度、歸一化、視角分類、聚合）歸在 `metrics/wave_geometry.py`，維持 `無 I/O、無狀態、mypy strict`
- 渲染分層延續 CLAUDE.md 既有原則：浪面標註 gate 在 `record.wave is not None`，`不受 keypoints 是否存在影響`（某幀沒偵測到人，浪面標註仍在）

---

## 3. 模組詳細

`新增/變更檔案`

```
src/surfanalysis/
├── extraction/
│   ├── wave/                      ← 新增子套件（每引擎一檔，保持小而專注）
│   │   ├── __init__.py
│   │   ├── base.py                WaveEngine (ABC) + MockWaveEngine + make_wave_engine() 工廠
│   │   ├── horizon.py             detect_horizon()：sea-sky / 主水平結構線偵測
│   │   ├── motion.py              估全域攝影機運動（供 auto 選引擎）
│   │   ├── ocean.py               HorizonAnchoredWaveEngine（每幀、motion-agnostic）
│   │   ├── static.py              Mog2WaveEngine（固定機位、背景相減）
│   │   └── prescan.py             pre-scan：一次回傳 (engine_name, view)
│   ├── analyzer.py                ← 改：FrameAnalyzer 多吃可選 wave_engine
│   └── schema.py                  ← 改：WaveMetrics / WaveSummary；schema_version 1.0→1.1
├── metrics/
│   └── wave_geometry.py           ← 新增純函數（mypy strict）
├── rendering/
│   └── wave_overlay.py            ← 新增：依 angle_line/height_* 畫線，與 pose overlay 解耦
└── cli.py                         ← 改：extract --wave/--wave-engine/--view；render --show-wave/--wave-color/--wave-height-pct
```

`職責劃分`

- `extraction/wave/base.py`：`WaveEngine(ABC).detect(frame, timestamp_ms) -> WaveMetrics | None` + `info() -> EngineInfo`；`MockWaveEngine`（測試替身，鏡像 `MockEngine`）；`make_wave_engine(name, view, ...)` 工廠
- `extraction/wave/horizon.py`：`detect_horizon(frame) -> HorizonLine | None`，海浪用 sea-sky 高對比直線；造浪池退回主水平結構線或假設影像水平
- `extraction/wave/motion.py`：以 `cv2.phaseCorrelate` / ORB 估全域位移，供 pre-scan 判斷固定 vs 會動機位
- `extraction/wave/ocean.py`：`HorizonAnchoredWaveEngine`，每幀獨立（不受攝影機平移影響），地平線 → 水域 ROI → 白沫帶(高 V 低 S)+浪面紋理/梯度分割 → crest/base/face line
- `extraction/wave/static.py`：`Mog2WaveEngine`，`cv2.BackgroundSubtractorMOG2` 學靜止場館為背景、流動水為前景 → 最大輪廓 → crest/base/face line；需暖身幀
- `extraction/wave/prescan.py`：`prescan(frames, n=15) -> (engine_name, view)`，一次預掃同時決定引擎（運動量）與視角（浪區/crest 線幾何），跑完鎖定不再每幀飄
- `metrics/wave_geometry.py`：`classify_view()`、`crest_tilt_deg()`、`face_steepness_deg()`、`normalized_height()`、`aggregate_wave()`（median/p90）。純函數、可單測
- `rendering/wave_overlay.py`：`WaveOverlay.draw(frame, record)`，只認 `angle_line` / `height_top` / `height_bottom`（視角無關），標籤文字依 `angle_kind` 切換

---

## 4. 資料契約 (schema v1.1)

`schema_version` 1.0 → 1.1。`所有新欄位皆 Optional 帶預設` → 舊的 1.0 JSON 仍能驗證通過（向後相容）。

```python
from typing import Literal

WaveView = Literal["facing", "side"]
AngleKind = Literal["crest_tilt", "face_steepness"]

class WaveMetrics(BaseModel):              # 每幀
    view: WaveView
    height: float                          # 歸一化 0-1：浪區垂直延伸 / 幀高
    angle_deg: float                       # 依 view：facing=crest 傾斜；side=面陣陡度（皆相對地平線、帶號）
    angle_kind: AngleKind                  # 標明 angle_deg 的語意，避免兩義誤導
    confidence: float                      # 0-1；< 門檻則該幀 wave=None
    angle_line: tuple[tuple[float, float], tuple[float, float]]   # 畫角度線的兩端點(歸一化)
    height_top: tuple[float, float]        # 高度括號頂(crest)
    height_bottom: tuple[float, float]     # 高度括號底(base/trough)
    horizon_deg: float | None = None       # 偵測到的地平線傾角(攝影機 roll)；None=假設水平

class WaveSummary(BaseModel):              # session
    frames_detected: int
    view: WaveView | Literal["mixed"]      # 主視角，或 mixed
    height_median: float
    height_p90: float
    angle_median: float
    engine: str                            # 實際跑的引擎 "ocean" | "static"

# 既有模型擴充（皆 Optional 預設，向後相容）：
# class FrameRecord:   wave: WaveMetrics | None = None
# class SessionRecord: wave_engine: EngineInfo | None = None
#                      wave_summary: WaveSummary | None = None
```

`幾何欄位設計`：用視角無關的 `angle_line`(兩端點) + `height_top/height_bottom`(括號) 描述要畫的線，`angle_kind` 只決定 HUD 標籤文字。overlay 與下游不需知道視角即可繪製；未來新增第三種視角不必動契約。

`版本相容`：`cmd_render` 的版本檢查由 `!= "1.0"` 改為「接受 major 版本為 1」（`1.0` 與 `1.1` 皆收，`1.0` 視為無 wave）。

`metrics.json 片段範例（新增部分）`

```json
{
  "schema_version": "1.1",
  "wave_engine": { "name": "wave-ocean", "version": "0.1.0",
                   "params": { "view": "facing", "min_confidence": 0.5 } },
  "frames": [
    {
      "frame_index": 0,
      "wave": {
        "view": "facing",
        "height": 0.42,
        "angle_deg": 8.3,
        "angle_kind": "crest_tilt",
        "confidence": 0.74,
        "angle_line": [[0.18, 0.31], [0.86, 0.27]],
        "height_top": [0.52, 0.29],
        "height_bottom": [0.52, 0.71],
        "horizon_deg": -1.2
      }
    }
  ],
  "wave_summary": {
    "frames_detected": 690,
    "view": "facing",
    "height_median": 0.40,
    "height_p90": 0.47,
    "angle_median": 7.6,
    "engine": "ocean"
  }
}
```

---

## 5. 視角 (view) 與量法定義

單機位下，浪面陡度沿攝影機光軸（深度）傾斜，無法從單張影像還原；因此 `facing` 與 `side` 量的是 `不同的物理量`，線的位置也不同。

| view | 浪角 angle 量什麼 | angle_kind | angle_line 是哪條 | 浪高 height 怎麼量 |
|---|---|---|---|---|
| facing（正對） | 浪唇線傾斜（浪肩坡向/peel 方向） | `crest_tilt` | 橫跨畫面的 crest/lip 線 | 垂直括號 crest→base 的垂直延伸 / 幀高 |
| side（側拍） | 浪面剖面陡度 (steepness) | `face_steepness` | 一條斜的浪面剖面線(trough→crest) | 同剖面線的垂直投影 / 幀高 |

```
facing（正對浪）— sample 即此種
   ~~~~~ 浪唇 crest line ~~~~~     ← angle_line：tilt vs horizon
   /      (foam 頂緣)      \
  /   surfer ●             \
  |          ┊ height       |     ← 垂直括號 height_top→height_bottom
  ~~~~~~ base/trough 底 ~~~~~~

side（側拍剖面）
            ●crest
          ╱ ┊
   face ╱   ┊ height           ← angle_line：斜剖面線；height=其垂直投影
   ∠  ╱     ┊
    ╱_______●trough ···· horizon
```

`視角偵測 (--view auto)`：在 `prescan` 一次決定，投票鎖定整支片段（避免逐幀飄）。判據為偵測到的浪區/crest 線「長寬比」與「相對地平線的延展方向」：crest 主要橫向延展且浪區縱向被透視壓縮 → `facing`；浪區呈斜向剖面、crest 偏向一端的高點 → `side`。可用 `--view facing|side` 強制。

`量法公式 (metrics/wave_geometry.py，純函數)`

```
# 角度一律相對地平線（吸收攝影機 roll）。h = horizon 傾角(度)
crest_tilt_deg(angle_line, h)     = wrap_to_180(line_angle(angle_line) - h)
face_steepness_deg(angle_line, h) = wrap_to_180(line_angle(angle_line) - h)
# 兩者公式同形，差別在「angle_line 由哪條線提供」與「angle_kind 標籤/語意」

normalized_height(top, bottom, frame_h) = |top.y - bottom.y|   # top/bottom 已歸一化(0-1)

aggregate_wave(frames):
    hs = [f.wave.height for f in frames if f.wave]
    as_ = [f.wave.angle_deg for f in frames if f.wave]
    return WaveSummary(
        frames_detected = len(hs),
        view            = 多數視角 or "mixed",
        height_median   = median(hs),
        height_p90      = percentile(hs, 90),
        angle_median    = median(as_),
        engine          = 實際引擎名,
    )
```

---

## 6. 引擎與 pipeline

`WaveEngine 介面`（對稱於既有 `PoseEngine`）

```python
class WaveEngine(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> WaveMetrics | None: ...
    @abstractmethod
    def info(self) -> EngineInfo: ...
```

`兩個引擎`

```
ocean: HorizonAnchoredWaveEngine（每幀、不受攝影機運動影響）
  detect_horizon → 水域 ROI(地平線下方)
  → 白沫帶(高 V 低 S) + 浪面紋理/梯度 分割
  → crest=浪唇/沫帶頂, base=陡面底, angle_line=依 view 取線
  → height/angle/confidence → WaveMetrics

static: Mog2WaveEngine（固定機位、造浪池）
  BackgroundSubtractorMOG2 學靜止場館為背景 → 流動水=前景
  → morphology → 最大輪廓(面積過濾濾掉衝浪者)
  → crest/base/face line 同上
  + 夜拍低光更穩、更快；需暖身幀(warmup 期 confidence 較低)
```

`auto 選擇 (--wave-engine auto，預設)`：`prescan` 預掃前 ~15 幀，全域位移中位數近零 → `static`，明顯位移 → `ocean`；同一次預掃也決定 `view`。CLI 預掃後 `cap.set(CAP_PROP_POS_FRAMES, 0)` 倒回從頭跑。可用 `--wave-engine ocean|static` 強制。

`pipeline 整合 (analyzer.py)`

```
FrameAnalyzer(engine: PoseEngine, stance, source, wave_engine: WaveEngine | None = None)

run(frames):
  for idx, frame in enumerate(frames):
      kp   = pose_engine.detect(frame, ts)          # 既有，不變
      wave = wave_engine.detect(frame, ts) if wave_engine else None   # 新增
      frames.append(FrameRecord(..., wave=wave))
  ...
  wave_summary = aggregate_wave(frames) if wave_engine else None
```

`None 語意`（對應既有 pose 的 visibility<0.5 gating）：`WaveMetrics=None` 當 confidence < 門檻 / 地平線測不到 / 遮罩過小；聚合跳過 None；零偵測則 `wave_summary=None`。

---

## 7. 渲染 / overlay

`WaveOverlay` 與 pose overlay 完全解耦，於 `cmd_render` 分開呼叫：

```
for record in frames:
    frame = renderer.draw(frame, record)        # pose（既有，keypoints None 時早退）
    frame = wave_overlay.draw(frame, record)    # wave（獨立，不受 keypoints 影響）
```

`WaveOverlay.draw` gate 在 `record.wave is not None`，畫：

- 淡色地平線（依 `horizon_deg`）
- `angle_line` 兩端點連線（含浪角標籤），視角無關
- `height_top → height_bottom` 垂直高度括號（含浪高數值）
- 右上角 HUD：即時值(EMA 平滑) + session 中位數

`數值顯示規則`（對齊既有 lean/weight 標籤慣例）

```
左上 pose(既有)        右上 wave HUD
F:60% B:40%           wave H 0.42       ← 即時值(EMA 平滑)
lean:+3 deg           tilt +8 deg       ← facing 顯示 tilt / side 顯示 face
                      med 0.40 / +7     ← session 中位數(穩定頭條)
```

- 角度一律 `deg` 字樣，`不用 °`（cv2 HERSHEY 字型無此 glyph，對齊 `overlay.py:109`）
- 浪高預設歸一化 `0.42`（對齊 JSON）；`--wave-height-pct` 可切 `42%`（幀高百分比）
- HUD 標籤隨 `angle_kind`：facing→`tilt`、side→`face`
- `顯示層 EMA 平滑、JSON 存原始每幀值`（資料/顯示分層，對齊 CoM/stability 做法）

---

## 8. CLI 介面與行為

`extract subcommand（新增）`

```
options:
  --wave                           啟用浪面偵測 (default: 關，保持現有行為與效能不變)
  --wave-engine {auto,ocean,static}  浪面引擎 (default: auto)
  --view {auto,facing,side}        攝影機視角 (default: auto)
```

`render subcommand（新增）`

```
options:
  --show-wave / --no-wave          畫浪面標註 (default: 有 wave 資料就畫)
  --wave-color HEX                 浪面線顏色 (default: 例如 #00E5FF 青)
  --wave-height-pct                浪高以 % 顯示 (default: 歸一化 0-1)
```

`預設輸出檔名契約不動`（`test_cli_extract.py` / `test_cli_render.py` 仍綠）。`--wave` 未開時 `extract` 行為與現況完全相同。

`退出碼`：沿用既有（0 成功 / 1 IO / 2 Engine / 3 解碼 / 4 schema）。`render` 接受 schema major 版本 1（1.0 與 1.1）。

---

## 9. 測試策略

沿用 repo 高覆蓋 TDD 風格，於既有測試上加「視角」維度。

```
test_wave_geometry.py   純函數：合成 crest/base+horizon → crest_tilt/face_steepness；
                        normalized_height；classify_view（facing vs side 合成輸入）；
                        aggregate_wave 的 median/p90/mixed；None 處理
test_wave_engine_ocean.py  合成幀：地平線+白沫帶+移動水 → height/angle 在容差內；confidence 行為
test_wave_engine_static.py 合成幀：靜止背景+移動亮塊 → 最大輪廓抽幾何；warmup 期 None
test_wave_prescan.py    靜止幀 → static；平移幀 → ocean；facing/side 合成 → 對應 view
test_wave_overlay.py    依 view 畫對的線(angle_line/bracket)；keypoints=None 時仍畫 wave；
                        angle_kind 對應 HUD 標籤
test_schema.py (擴充)    WaveMetrics/WaveSummary round-trip；1.1；1.0 向後相容(wave 預設 None)
test_cli_extract.py(擴充) --wave 產生 wave 欄位；未開時行為不變；預設檔名不變
test_cli_render.py(擴充)  render 接受 1.0/1.1；--wave-height-pct/--wave-color；無 wave 資料時不畫
test_e2e.py (擴充)        extract --wave 跑 sample → render → 斷言 wave_summary 存在且 frames_detected>0
```

`測試替身`：`MockWaveEngine`（回放固定 `WaveMetrics | None` 序列），鏡像既有 `MockEngine`。

---

## 10. 已知限制與未來工作

- `單機位無絕對尺度`：歸一化值在同一片段內可比，跨片段/縮放不可比；公尺換算需 scale calibration（範圍外）
- `facing 量不到面陣陡度`：只能給浪唇傾斜；要陡度需 side 視角或多機位（已於 view 設計中明確標示語意）
- `ocean 單機位精度有限`：會動的攝影機量海浪幾何本質上難，`地平線錨`是最大穩定度槓桿
- 未來：剝離角 (peel angle)、浪週期、多浪追蹤、絕對公尺、`WaveEngine` 換 ML 分割實作（介面已預留）
