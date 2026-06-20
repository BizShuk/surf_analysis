---
title: Physical Wave Height — Design Spec (meters)
date: 2026-06-21
status: draft
phase: extends Phase 1 (offline CLI); supersedes the "out of scope: absolute scale" line of 2026-06-15 wave-analysis-design.md
schema_version: 1.2 (target)
depends_on: 2026-06-15-wave-analysis-design.md (schema_version 1.1)
---

# Physical Wave Height (meters) — Design Spec

## 1. 範圍與目標

`目標`：在既有 wave analysis (schema 1.1) 上新增「物理浪高（公尺）」pass；沿用同一個 wave engine 的 crest / base 觀察，改用相機幾何（one-view metrology）+ 深水波色散交叉驗證，把目前 `wave_summary.height_median` / `height_p90` 從「畫面占比（0-1）」升級為「公尺（m）」。

`本次範圍`

- 新增 `WaveGeometry` 純函式（`metrics/wave_geometry.py`）:`normalized_to_world_height(...)` — 不 import Pydantic、不做 I/O。
- 新增 `PhysicalWaveComputer` (`extraction/wave/physical.py`):把 `WaveObservation` + `CameraModel` + horizon 一起丟進 `normalized_to_world_height`;輸出 `PhysicalWaveMetrics` 含 crest / base 的世界座標 (X, Y, Z)、wave height in meters、confidence。
- 擴充 `CameraModel` (`extraction/wave/camera.py`,新檔):內含 `camera_height_m`, `focal_length_mm`, `sensor_height_mm`, `image_height_px`, `pitch_deg`;`pitch_deg` 由 `horizon.py` 推得。
- `schema_version` 1.1 → 1.2:`WaveMetrics` 加 `physical: PhysicalWaveMetrics | None`;`WaveSummary` 加 `height_m_median`, `height_m_p90`, `confidence`。舊欄位 `height_median` / `height_p90` 標 `deprecated` 但保留（向下相容,直到 2.0）。
- `prescan.py` 擴充 `prescan_physical(...)`:判斷 EXIF/使用者輸入是否齊備 → 決定 `WaveSummary.physical_status` = `computed` / `insufficient_metadata` / `unsupported_view`。
- `cli.py` 新 flag `--camera-height-m`、`--focal-length-mm`、`--sensor-height-mm`（若 EXIF 有則預設,否則必填）;沿用 `--wave-engine` / `--view`。
- `wave_overlay.py` 顯示單位從 `(fraction)` 換成 `(m)` + 信心徽章;舊模式由 `--legacy-screen-fraction` 開啟。
- 文件同步：`CLAUDE.md` § "Wave height semantics" 標記「設計已落 → 見 `plans/2026-06-21-physical-wave-height-design.md`」;§ "Run" 把警告從「⚠ 是 fraction」改成「⚠ 預設 fraction;`--camera-height-m` 開啟後輸出公尺」。

`不在本次範圍`

- 衝浪者 / 衝浪板當 reference scale（明確排除;見 § 3 動機）。
- 多相機立體、SfM、IMU 整合;手持 / POV 影片只給 `confidence: low` 不報數。
- 絕對精度校正（驗證集 / 量測真值）— 本次只給演算法 + 路徑,精度評估留待樣本齊備後。
- 替換 `ocean` / `static` 引擎;既有 engine 介面不動,`PhysicalWaveComputer` 是純下游消費者。
- 移除 deprecated `height_*` 欄位（保留到 schema 2.0,給下游半年過渡）。

`需求決策來源`

本次對話中,user 明確表達：

1. `wave_summary.height_*` 目前是畫面占比、不是物理浪高 → 必須處理。
2. 不接受以衝浪者當 reference scale → 排除所有依賴 pose 的尺度推算。
3. 移除 `height` (fraction) 欄位、新欄位命名為 `wave_height` / `physical_height`（公尺）→ 對齊 WSL / 氣象語意。

設計選擇（須 user 在實作前再確認）：

- 保留 deprecated `height_*` 半年 vs 直接 break。預設保留（schema 1.2）。
- `confidence` 計算閾值（method 1+2 差多少算「分歧」）。預設 `|ΔH/H| > 0.20` = `low`。
- 不齊 metadata 時的 UX:自動 fallback 到 fraction + 警告,還是強制要 metadata？預設 fallback + 警告。

---

## 2. 動機（為何現在做）

`wave_summary.height_median = 0.41` 看起來像「0.41 公尺」,實際是「畫面高度的 41%」——這是語意落差,不是單位轉換能解決。三個污染源（相機距離、鏡頭焦段、相機俯仰角）讓 image-space metric 無法映射到 world-space metric。詳見 `CLAUDE.md` § "Wave height semantics"。

任何下游 consumer（衝浪報告、教練 UI、WSL 數據對比）都會把 0.41 當 0.41 m 解讀,靜默產生垃圾輸出。修這個語意落差是契約責任,不是 nice-to-have。

---

## 3. 演算法（兩路徑 + 交叉驗證）

### 3.1 方法 1 — 相機幾何 + 已知架設高度（推薦 baseline）

**前提**

- 相機架設高度 `H` 已知（岸拍、堤防、空拍機各自有典型值）。
- 鏡頭焦距 `f` 已知（EXIF 或設備規格;GoPro Hero 12 default = 2.71 mm fisheye 等效 16 mm,已知）。
- 水平線 `horizon_deg` 由 `horizon.py` 已能偵測 → 換算成相機俯仰角 `pitch_deg`。
- 影格中心 `cy = image_height_px / 2`(或從 EXIF / 標定得)。

**步驟**

```
θ = pitch_deg
對畫面 y 處的點:
    α = arctan((y − cy) / f_pixels)
    其中 f_pixels = image_height_px / (2 · tan(fov_half))   # 或 EXIF FocalLengthIn35mmFilm → 換算
    Z(y) = H / tan(θ + α)
    Y(y) = Z(y) · tan(α)
wave_height_m = |Y(crest) − Y(trough)|
```

**輸出**

```python
PhysicalWaveMetrics(
    crest_world: tuple[float, float, float],   # X, Y, Z in meters
    trough_world: tuple[float, float, float],
    height_m: float,
    method: Literal["camera_geometry"],
    confidence: Literal["high", "medium", "low"],
)
```

**限制**

- 鏡頭 roll 不為零時需把世界 Y 旋一個 `roll_deg`(從 `horizon.py` 取得,已在)。
- `H` 不準確 → 結果線性放大誤差;UI 要明確顯示「依使用者提供的 H 計算」。
- 手持 / POV:`H` 與 `θ` 每幀漂移,本方法結果視為 `low` confidence、不報單一數字,只報 session 中位數 + 變異。

### 3.2 方法 2 — 波長 / 週期色散（純物理交叉驗證）

**前提**

- 深水重力波假設（水深 > 半波長;岸拍 / 造浪池常用條件）。
- 同一引擎 (`ocean`) 給出連續 crest 軌跡 → 影像上的 crest 間距 = `L_pixels`。
- crest 通過定點的時序差 = `T`（秒）;數幀可估。

**公式**

```
L = g · T² / (2π)         # 深水波 dispersion, g = 9.81 m/s²
H_break ≈ L / 7            # Miche 準則 (H/L ∈ [1/10, 1/7] 為碎波常見值)
```

**與方法 1 的整合**

- 方法 2 給「H_break」上限（不能 > L/7,否則波不會碎）。
- 若方法 1 算的 H > L/7 → 標 `confidence: low`,提示「數值可能偏高」,而不是靜默報出。
- 若方法 1 算的 H < L/10 → 標 `confidence: low`,提示「數值可能偏低、未進入碎波條件」。

**輸出**

```python
WavelengthEstimate(
    wavelength_m: float,
    period_s: float,
    h_upper_m: float,       # = wavelength_m / 7
    h_lower_m: float,       # = wavelength_m / 10
    confidence: Literal["high", "medium", "low", "unavailable"],
)
```

**限制**

- `L_pixels → L_meters` 仍需方法 1 的相機幾何（除非接受影像 proxy ±20% 誤差）。
- 影像 proxy 路徑:若 `L_pixels` 來自同一影格的 crest 對 crest 距離,可用其比例推算,不需相機幾何——精度較差但獨立性高。
- 多個 crest 並存 / 平行波時 `L` 不單值 → 標 `unavailable`。

### 3.3 交叉驗證 → `confidence`

| 方法 1 vs 方法 2 | `confidence` |
|---|---|
| 兩者皆可用且 \|ΔH/H\| ≤ 0.20 | `high` |
| 兩者皆可用但 0.20 < \|ΔH/H\| ≤ 0.50 | `medium` |
| 兩者分歧 \|ΔH/H\| > 0.50 | `low` |
| 只有方法 1 可用,內部一致 | `medium` |
| 只有方法 1 可用且 n_frames < 30 | `low` |
| 兩者皆不可用 | `unavailable`（報 null） |

`WaveSummary.height_m_median` 只取 `confidence ∈ {high, medium}` 的幀;`low` 幀仍寫進 per-frame 紀錄但不進 session 聚合。

---

## 4. 架構總覽（與既有 wave pipeline 平行）

```text
┌──────────────────────────────────────────────────────────────────┐
│                         CLI (argparse)                           │
│   surf extract <video> --wave                                     │
│     [--camera-height-m 3.0]                                       │
│     [--focal-length-mm 16.0]                                      │
│     [--sensor-height-mm ...]                                      │
│     [--wave-engine auto|ocean|static] [--view auto|facing|side]  │
│   surf render  <video> <metrics.json> [--legacy-screen-fraction]  │
└──────────────┬──────────────────────────────┬────────────────────┘
               │                              │
               ▼                              ▼
   ┌───────────────────────────┐   ┌──────────────────────────┐
   │ extraction package        │   │  rendering package       │
   │                           │   │                          │
   │  WaveEngine ──────┐       │   │ WaveOverlay (existing)   │
   │   ├ ocean         │       │   │  ├ (m) + confidence 徽章 │
   │   └ static        │       │   │  ├ (fraction) [legacy]   │
   │  PhysicalComputer │ NEW   │   │                          │
   │   ├ CameraModel   │ NEW   │   │                          │
   │   ├ WaveGeometry  │ NEW   │   │                          │
   │   └ WavelengthEst │ NEW   │   │                          │
   │  prescan ─────────┘       │   │                          │
   │   └ prescan_physical NEW  │   │                          │
   └───────────┬───────────────┘   └──────────────────────────┘
               │                              ▲
               ▼                              │
        metrics.json (1.2) ───────────────────┘
```

`PhysicalComputer` 完全下游;既不碰 pose,也不修改 `WaveEngine` 介面。換 engine / 換 view 都不影響物理計算路徑。

---

## 5. Schema 變更（1.1 → 1.2）

```python
# src/surfanalysis/extraction/schema.py (新增)

class CameraModel(BaseModel):
    camera_height_m: float
    focal_length_mm: float | None = None        # 與 sensor_height_mm 二擇一
    sensor_height_mm: float | None = None
    image_height_px: int
    pitch_deg: float | None = None              # 由 horizon.py 推得
    roll_deg: float | None = None
    source: Literal["user", "exif", "default"]  # 預設值的來源,給 UI 提示


class PhysicalWaveFrame(BaseModel):
    crest_world: tuple[float, float, float] | None = None   # (X, Y, Z) in meters
    trough_world: tuple[float, float, float] | None = None
    height_m: float | None = None
    method: Literal["camera_geometry", "wavelength", "cross_validated", "skipped"]
    confidence: Literal["high", "medium", "low", "unavailable"]
    reason: str | None = None                               # 為何 skipped/low


class WaveMetrics(BaseModel):
    # ... 既有欄位不動 ...
    physical: PhysicalWaveFrame | None = None                # NEW


class WaveSummary(BaseModel):
    # ... 既有欄位 ...
    height_m_median: float | None = None                     # NEW
    height_m_p90: float | None = None                        # NEW
    confidence: Literal["high", "medium", "low", "unavailable"] = "unavailable"  # NEW
    camera: CameraModel | None = None                        # NEW（若有填 metadata）
    physical_status: Literal["computed", "insufficient_metadata", "unsupported_view"] = "insufficient_metadata"  # NEW
```

**向下相容**

- `metrics.json` 1.1 reader 讀 1.2:忽略新欄位,舊欄位仍可用。
- 1.2 reader 讀 1.1:`physical` / `height_m_*` 為 `None`,`physical_status` = `"insufficient_metadata"`;UI 顯示「需要 metadata 升級到 1.2」提示。
- `height_median` / `height_p90`(fraction)標 `deprecated`,`schema.py` 加 `model_config = ConfigDict(deprecated_fields_warn=True)` 或寫進 docstring,不強制移除。

---

## 6. CLI 介面

```text
surf extract <video>
    --wave
    [--camera-height-m FLOAT]            # 預設 None;若 EXIF 推得到則用 EXIF
    [--focal-length-mm FLOAT | --sensor-height-mm FLOAT]   # 二擇一
    [--image-height-px INT]              # 預設從 frame 讀
    [--wave-engine auto|ocean|static]
    [--view auto|facing|side]

surf render <video> <metrics.json>
    [--no-wave]
    [--legacy-screen-fraction]           # 顯示舊 fraction 而非 m（debug 用）
```

`--camera-height-m` 缺值時,EXIF 無法推得 → 整個 physical pass 跳過,`physical_status = "insufficient_metadata"`,`metrics.json` 仍輸出但 `physical: None`,`height_m_*: None`;既有的 fraction 結果照樣可用。這保證舊用法不退步。

---

## 7. Overlay 變更

`wave_overlay.py` 顯示規則：

- `WaveSummary.physical_status == "computed"` → 顯示 `H = 0.85 m (±0.10, confidence: high)`,數字 + 信賴區間 + 信心徽章。
- `physical_status == "insufficient_metadata"` → 顯示舊 fraction,並加小字「要 m 單位請加 `--camera-height-m`」。
- `physical_status == "unsupported_view"`（例如 handheld POV）→ 顯示「不支援物理量」,不報數。
- `--legacy-screen-fraction` 強制走舊路徑（debug / A/B 比對）。

---

## 8. 風險與限制

| 風險 | 影響 | 緩解 |
|---|---|---|
| `H` 填錯（使用者誤報相機高度） | 數值線性偏差 | CLI 加 hint「量測至水面的高度」;`source: user` 時顯示警告 |
| 手持 / POV | 每幀漂移,方法 1 不可信 | 自動偵測 global motion > threshold → 標 `low` 不聚合 |
| 深水波假設不成立（淺水 / 造浪池邊角） | 方法 2 公式失效 | 預設 `confidence: unavailable`,只在 L 來自連續 crest 時啟用 |
| 鏡頭 roll 不為零 | 世界 Y 軸歪斜 | `CameraModel.roll_deg` 旋轉補償,從 `horizon.py` 取 |
| EXIF 缺失或被剝離 | 焦距無法推得 | 強制要求 `--focal-length-mm` 或 `--sensor-height-mm` |
| 精度評估無 ground truth | 無法驗證演算法 | 留 task:蒐集樣本 + 量測實浪高 → calibration loop |

---

## 9. 參考

- `plans/2026-06-15-wave-analysis-design.md` — 既有 wave pipeline 設計,本次是其延伸。
- `plans/2026-06-16-wave-analysis-plan.md` — 既有 wave pipeline 實作計畫。
- `CLAUDE.md` § "Wave height semantics" — 語意落差描述與 4 條路徑的 overview。
- `extraction/wave/horizon.py` — `horizon_deg` 偵測,本次直接消費。
- `extraction/wave/prescan.py` — 擴充 `prescan_physical(...)`。
- `extraction/wave/ocean.py` — crest 軌跡,本次消費以推 `L_pixels` 與 `T`。
- `metrics/wave_geometry.py` — 新增 `normalized_to_world_height(...)` pure function。
- Surf engineering reference:深水波 dispersion `L = gT²/(2π)` 與 Miche 碎波準則 `H/L ≈ 1/7`。
