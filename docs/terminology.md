# surfanalysis — 術語表 (Terminology)

本檔是本專案術語的`單一定義來源 (single source of truth)`。`README.md`、`CLAUDE.md`、
程式碼與 commit message 一律沿用此處用詞,同一概念不得有第二種說法。

## 管線與契約 (Pipeline and Contract)

| 術語 (Term) | 英文 (English) | 定義 (Definition) | 出處 (Source) |
| ----------- | -------------- | ----------------- | ------------- |
| 抽取 | `extract` | 第一階段。跑姿態估計與浪況偵測,輸出 `metrics.json` | `src/surfanalysis/cli.py` |
| 渲染 | `render` | 第二階段。讀 `metrics.json` 把骨架與指標畫回影片 | `src/surfanalysis/cli.py`, `rendering/` |
| 指標檔 | `metrics.json` | 兩階段之間唯一的介面。`render` 不重跑推論,只消費此檔 | `extraction/schema.py` (`SessionRecord`) |
| 綱要版本 | `schema_version` | `metrics.json` 的相容性標記。可讀版本由 `SUPPORTED_SCHEMA_VERSIONS` 列舉 | `extraction/schema.py` |
| 不相容綱要例外 | `IncompatibleSchemaError` | 讀到不支援的 `schema_version` 時`大聲失敗`,要求使用者重跑 extract。刻意`不`靜默降級成部分解讀 | `extraction/exceptions.py` |
| 場次記錄 | `SessionRecord` | `metrics.json` 的根物件。含 source / engine / stance / frames / summary,浪況欄位為選填 | `extraction/schema.py` |
| 影格記錄 | `FrameRecord` | 單一影格的資料列。`keypoints` / `metrics` / `wave` 三者皆可為 `None` | `extraction/schema.py` |

## 姿態與生物力學 (Pose and Biomechanics)

| 術語 (Term) | 英文 (English) | 定義 (Definition) | 出處 (Source) |
| ----------- | -------------- | ----------------- | ------------- |
| 關鍵點 | Keypoint | MediaPipe Pose 的 33 個人體landmark。數量由 Pydantic validator 強制,不足即拒絕 | `extraction/landmarks.py`, `extraction/schema.py` (`Keypoints._exactly_33`) |
| 可見度門檻 | `VISIBILITY_THRESHOLD` | 判定關鍵點是否可信的下限。低於門檻時相關指標一律回 `None`,絕不回推測值 | `extraction/landmarks.py` |
| 影片模式 | `VIDEO` running mode | MediaPipe 的時序追蹤模式,以前一幀姿態當先驗。`必須`使用此模式,`IMAGE` 模式會造成標註閃爍 | `extraction/mediapipe_engine.py`, `CLAUDE.md` |
| 單調時戳守衛 | Monotonic Timestamp Guard | `detect_for_video` 要求嚴格遞增的整數毫秒。引擎內建 `ts <= last → last+1` 修正,吸收重複或回退的時間戳 | `extraction/mediapipe_engine.py` |
| 姿態引擎 | `PoseEngine` | 姿態估計的抽象介面 (Strategy pattern),目前只有 MediaPipe 實作。契約為 `detect(frame, timestamp_ms)` | `extraction/engine.py` |
| 質心 | CoM (Center of Mass) | 以 Plagenhoef 分段質量係數由關鍵點加權求得的全身質心,座標正規化 | `metrics/com.py` |
| 前腳配重 | `weight_dist_front_pct` | 質心投影在雙腳連線上的前後分配百分比。前後由 `stance` 決定 | `metrics/weight_dist.py` |
| 站姿 | `Stance` | `regular` 或 `goofy`,決定哪隻腳是前腳 | `extraction/schema.py` |
| 軀幹傾角 | `torso_lean_deg` | 肩髖連線相對垂直的傾斜角度 | `metrics/angles.py` |
| 穩定度分數 | `com_stability_score` | 以滾動視窗統計質心位移換算的 `(0, 1]` 分數;有效樣本不足時回 `None` | `metrics/stability.py` (`StabilityWindow`) |
| 分層渲染 | Layered Overlay | 骨架 (偵測證據) 與衍生指標 (CoM / 配重線 / 傾角線 / 文字) 解耦。只要有 keypoints 就畫骨架,避免單腳低可見度讓整幀標註消失 | `rendering/overlay.py`, `CLAUDE.md` |

## 浪況分析 (Wave Analysis)

| 術語 (Term) | 英文 (English) | 定義 (Definition) | 出處 (Source) |
| ----------- | -------------- | ----------------- | ------------- |
| 浪況引擎 | `WaveEngine` | 浪面偵測的抽象介面,兩個實作見下 | `extraction/wave/base.py` |
| 海洋引擎 | `ocean` | 顏色與泡沫特徵偵測。適合`靜止造浪池`等穩定浪面 | `extraction/wave/ocean.py` |
| 靜態引擎 | `static` | MOG2 背景相減。適合`固定機位 + 短暫前景`的真實海浪岸拍;會把穩定水流學進背景而漏偵 | `extraction/wave/static.py` |
| 預掃描 | `prescan` | 取樣前若干幀,依`相機運動量`自動選引擎與視角。運動量低於門檻選 `static`,否則 `ocean` | `extraction/wave/prescan.py` |
| 視角 | `WaveView` | `facing` (正對浪) 或 `side` (側面)。決定量測的角度語意 | `extraction/schema.py` |
| 角度語意 | `angle_kind` | 標明 `angle_deg` 量的是什麼:`facing` 視角量浪唇傾斜 (`crest_tilt`),`side` 視角量浪面陡度 (`face_steepness`) | `extraction/schema.py` |
| 地平線角 | `horizon_deg` | 偵測到的地平線傾角,同時是相機 pitch 的推斷來源 | `extraction/wave/horizon.py` |
| 浪峰 / 浪谷 | Crest / Trough | 浪面的最高點與最低點。物理浪高即兩者的世界座標高度差 | `extraction/wave/ocean.py`, `metrics/wave_geometry.py` |

## 物理浪高 (Physical Wave Height)

| 術語 (Term) | 英文 (English) | 定義 (Definition) | 出處 (Source) |
| ----------- | -------------- | ----------------- | ------------- |
| 相機模型 | `CameraModel` | 單視角測量所需的相機幾何。`focal_length_mm` 與 `sensor_height_mm` 是推導 focal pixels 的兩條替代路徑,擇一即可 | `extraction/schema.py`, `extraction/wave/camera.py` |
| 架設高度 | `camera_height_m` | 相機到水面的高度。單視角測量 (one-view metrology) 的唯一已知尺度,`不使用衝浪者當參考尺` | `extraction/schema.py`, `CLAUDE.md` |
| 俯仰角 | `pitch_deg` | 相機下視角。通常由 `horizon_deg` 推斷,可由使用者覆寫。`它是參數不是開關` — 不同 pitch 會給出不同浪高 | `extraction/schema.py`, `README.md` |
| 世界座標 | `crest_world` / `trough_world` | 浪峰與浪谷的 `(X, Y, Z)` 公尺座標。單視角無法解橫向深度,故 `X` 恆為 `0` | `extraction/schema.py` (`PhysicalWaveFrame`) |
| 物理浪高 | `height_m` | 浪峰與浪谷的世界高度差,單位公尺。無法計算時為 `None`,`下游禁止填任何數字` | `extraction/schema.py` |
| 來源標記 | `source` | 相機參數的出處 (`user` / `exif` / `default`),供 UI 對使用者自填值提出警告 | `extraction/schema.py` (`CameraModel`) |

## 狀態值 (Status Values)

字面值一律以 `src/surfanalysis/extraction/schema.py` 的 `Literal` 定義為準。

`PhysicalWaveFrame.method` — 浪高的推導方法:

| 字面值 (Literal) | 語意 |
| ---------------- | ---- |
| `camera_geometry` | 由相機幾何 + 已知架設高度推得 (baseline 路徑) |
| `wavelength` | 由波長–週期色散關係推得 |
| `cross_validated` | 上述兩法結果一致,交叉驗證通過 |
| `skipped` | 相機 metadata 缺失,未計算。`height_m` 為 `None` |

`confidence` — 信心等級 (同時用於 `PhysicalWaveFrame` 與 `WaveSummary`):
`high`、`medium`、`low`、`unavailable`。

`WaveSummary.physical_status` — 整場次的物理浪高可用性:

| 字面值 (Literal) | 語意 |
| ---------------- | ---- |
| `computed` | 條件齊備,已計算 |
| `insufficient_metadata` | 缺相機參數。預設值;CLI 會在 stderr 印警告 |
| `unsupported_view` | 視角不在 `facing` / `side` 之內,無法計算 |

其餘列舉:`Stance` = `regular` / `goofy`;`WaveView` = `facing` / `side`
(`WaveSummary.view` 另可為 `mixed`);`AngleKind` = `crest_tilt` / `face_steepness`;
`CameraModel.source` = `user` / `exif` / `default`。

## 縮寫 (Abbreviations)

| 縮寫 | 全稱 | 說明 |
| ---- | ---- | ---- |
| `CoM` | Center of Mass | 全身質心,以 Plagenhoef 分段質量係數求得 |
| `MOG2` | Mixture of Gaussians v2 | OpenCV 的背景相減演算法,`static` 引擎的核心 |
| `COCO-17` | COCO 17-keypoint format | RTMPose / YOLO-pose 的輸出格式。`缺 heel 與 foot_index`,是換引擎的主要阻礙 |
| `SfM` | Structure from Motion | 由相機運動重建三維結構,浪高推算路徑之一 (工程量最高) |
| `EXIF` | Exchangeable Image File Format | 影像 metadata,焦距等相機參數的來源之一 |

## 待補 (To be extended)

- `WavelengthEstimator` 的波長與週期欄位尚未進入 `metrics.json` 的公開 schema,
  待其欄位定案後補入本表。
