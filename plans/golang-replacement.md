# Go 語言替代可行性評估與實作記錄 (Go Replacement Feasibility Assessment and Implementation Log)

## 結論 (Conclusion)

技術上可行，但有一個關鍵缺口：Go 語言沒有官方 `MediaPipe SDK`。其餘依賴都有成熟的 Go 語言替代方案；`metrics` 純數學邏輯、`overlay` 繪圖、`Pydantic schema` 都可 `1:1` 直譯。

---

## Go 語言替代方案可行性評估 (Go Replacement Feasibility Assessment)

### 依賴對照表 (Dependency Mapping)

| Python 依賴         | Go 替代方案                                  | 狀態                            |
| ------------------- | -------------------------------------------- | ------------------------------- |
| `mediapipe`         | 無官方 `SDK`，三條補救路徑（見下）           | ⚠️ `關鍵缺口`                   |
| `opencv-python`     | `gocv.io/x/gocv` (`GoCV`)                    | ✅ `穩定，繪圖 API 幾乎同名`    |
| `numpy`             | `gonum` 或純 Go 切片                         | ✅ `可用`                       |
| `pydantic`          | `encoding/json` + `struct tag` + `validator` | ✅ `nullable 欄位使用 *float64` |
| `tqdm` / `argparse` | `progressbar` / `cobra`                      | ✅ `可用`                       |

### MediaPipe 缺口的三條路徑 (Three Paths for MediaPipe Gap)

- `路徑 A：GoCV DNN + OpenPose 模型 (Path A: GoCV DNN + OpenPose Model)`
    - `說明`：最快實現，但只有 `18` 個 `COCO` 關鍵點。
    - `缺點`：導致 `landmarks.py` 全部索引、`weight_dist`、`stability` 都要重寫。
- `路徑 B：ONNX Runtime Go (yalue/onnxruntime_go) + BlazePose ONNX (Path B: ONNX Runtime Go + BlazePose ONNX)`
    - `說明`：保留 `33` 個關鍵點，`schema` 與 `landmark` 索引零改動，`JSON` 契約 (`schema_version="1.0"`) 不變。這是本研究建議的方案。
    - `代價`：需要自行實作 `MediaPipe` 內部的前後處理（例如 `ROI 裁切`、`sigmoid`、`反正規化`），約需 `200` 至 `300` 行程式碼。
- `路徑 C：mattn/go-tflite + 直接使用 pose_landmarker.task (Path C: go-tflite + pose_landmarker.task)`
    - `說明`：行為最接近現行 Python 版本。
    - `缺點`：需要自行解包 `.task` 並串接 `detector` 與 `landmarker` 兩個子模型，且該綁定庫自 `2022` 年後維護趨緩。

### 工作量預估 (Effort Estimation)

- `推理引擎 + 前後處理`：約 `1` 至 `2` 週。
- `其餘業務邏輯 + CLI 直譯`：約 `1` 週。

### 兩階段 JSON 契約之優勢 (Benefits of Two-Stage JSON Contract)

由於專案採用兩階段架構，我們可以選擇只將 `extract` 階段換成 Go 語言實作，或者只將 `render` 階段換成 Go 語言實作。兩者透過 `metrics.json` 進行資料互通，不需一次性完整移植所有模組。

---

## Python 視覺疊加功能實作記錄 (Python Visual Overlays Implementation Log)

### 修改概述 (Changes Overview)

| 檔案路徑                                                                                      | 變更說明                                                                           |
| --------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| [overlay.py](file:///Users/shuk/projects/surf_analysis/src/surfanalysis/rendering/overlay.py) | 新增 `_draw_angle_line`、`_draw_weight_line`、虛線繪製、`stance` 參數支援          |
| [cli.py](file:///Users/shuk/projects/surf_analysis/src/surfanalysis/cli.py)                   | 新增 `--angle-color`、`--weight-color` 參數，並在 `render` 時傳入 `session.stance` |
| [test_overlay.py](file:///Users/shuk/projects/surf_analysis/tests/test_overlay.py)            | 新增 `3` 個單元測試（軀幹角度線、腳部隱藏跳過、`stance` 切換標籤交換）             |
| [CLAUDE.md](file:///Users/shuk/projects/surf_analysis/CLAUDE.md)                              | 同步 `render` 功能描述與執行命令說明                                               |
| [test_cli_render.py](file:///Users/shuk/projects/surf_analysis/tests/test_cli_render.py)      | 新增對預設輸出命名的測試契約                                                       |
| [test_cli_extract.py](file:///Users/shuk/projects/surf_analysis/tests/test_cli_extract.py)    | 新增對預設輸出命名的測試契約                                                       |

### 程式碼變更詳情 (Code Diff Details)

#### `src/surfanalysis/rendering/overlay.py` 的變更

```diff
@@ -2,8 +2,15 @@

 from __future__ import annotations

+import math
+
 import cv2
 import numpy as np

-from surfanalysis.extraction.schema import FrameRecord
+from surfanalysis.extraction.landmarks import (
+    L_FOOT,
+    L_HIP,
+    L_SHOULDER,
+    R_FOOT,
+    R_HIP,
+    R_SHOULDER,
+)
+from surfanalysis.extraction.schema import FrameRecord, Stance
 from surfanalysis.rendering.skeleton import SKELETON_EDGES

 VISIBILITY_DRAW = 0.5
+TRUNK_EXTEND = 1.25  # extend trunk line past shoulders for readability
+REF_LINE_COLOR = (200, 200, 200)  # dashed vertical reference (BGR)
+DASH_LEN = 8


 def hex_to_bgr(s: str) -> tuple[int, int, int]:
@@ -14,14 +21,114 @@
     return (b, g, r)


+def _dashed_line(
+    frame: np.ndarray,
+    p1: tuple[int, int],
+    p2: tuple[int, int],
+    color: tuple[int, int, int],
+    thickness: int = 2,
+) -> None:
+    length = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
+    if length < 1.0:
+        return
+    n = max(1, int(length // DASH_LEN))
+    steps = np.linspace(p1, p2, n + 1)
+    for i in range(0, n, 2):
+        a = tuple(np.round(steps[i]).astype(int))
+        b = tuple(np.round(steps[i + 1]).astype(int))
+        cv2.line(frame, a, b, color, thickness)
+
+
 class OverlayRenderer:
     def __init__(
         self,
         skeleton_color: str = "#00FF00",
         com_color: str = "#FFFF00",
         text_color: str = "#FFFFFF",
+        angle_color: str = "#FF40FF",
+        weight_color: str = "#FFA500",
         font_scale: float = 0.6,
         show_secondary: bool = False,
+        stance: Stance = "regular",
     ) -> None:
         self._sk = hex_to_bgr(skeleton_color)
         self._com = hex_to_bgr(com_color)
         self._tx = hex_to_bgr(text_color)
+        self._angle = hex_to_bgr(angle_color)
+        self._wt = hex_to_bgr(weight_color)
         self._font_scale = font_scale
         self._show_secondary = show_secondary
+        self._stance: Stance = stance

-    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
+    def _text(
+        self,
+        frame: np.ndarray,
+        text: str,
+        org: tuple[int, int],
+        color: tuple[int, int, int] | None = None,
+    ) -> None:
+        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
+                    self._font_scale, (0, 0, 0), 3, cv2.LINE_AA)
+        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
+                    self._font_scale, color or self._tx, 1, cv2.LINE_AA)
+
+    def _draw_angle_line(
+        self,
+        frame: np.ndarray,
+        pts: list[tuple[int, int, float]],
+        lean_deg: float,
+    ) -> None:
+        """Trunk line (mid-hip -> mid-shoulder) vs dashed vertical reference."""
+        needed = (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP)
+        if any(pts[i][2] < VISIBILITY_DRAW for i in needed):
+            return
+        mid_sh = ((pts[L_SHOULDER][0] + pts[R_SHOULDER][0]) // 2,
+                  (pts[L_SHOULDER][1] + pts[R_SHOULDER][1]) // 2)
+        mid_hp = ((pts[L_HIP][0] + pts[R_HIP][0]) // 2,
+                  (pts[L_HIP][1] + pts[R_HIP][1]) // 2)
+        trunk_len = math.hypot(mid_sh[0] - mid_hp[0], mid_sh[1] - mid_hp[1])
+        if trunk_len < 1.0:
+            return
+        ref_end = (mid_hp[0], int(mid_hp[1] - trunk_len * TRUNK_EXTEND))
+        _dashed_line(frame, mid_hp, ref_end, REF_LINE_COLOR)
+        ext = (int(mid_hp[0] + (mid_sh[0] - mid_hp[0]) * TRUNK_EXTEND),
+               int(mid_hp[1] + (mid_sh[1] - mid_hp[1]) * TRUNK_EXTEND))
+        cv2.line(frame, mid_hp, ext, self._angle, 3)
+        label_y = max(15, ext[1] - 8)
+        self._text(frame, f"{lean_deg:+.1f} deg", (ext[0] + 8, label_y), self._angle)
+
+    def _draw_weight_line(
+        self,
+        frame: np.ndarray,
+        pts: list[tuple[int, int, float]],
+        com_px: tuple[int, int],
+        front_pct: float,
+    ) -> None:
+        """Back-foot -> front-foot baseline with CoM projection marker."""
+        if self._stance == "regular":
+            front_idx, back_idx = L_FOOT, R_FOOT
+        else:
+            front_idx, back_idx = R_FOOT, L_FOOT
+        fx, fy, fv = pts[front_idx]
+        bx, by, bv = pts[back_idx]
+        if fv < VISIBILITY_DRAW or bv < VISIBILITY_DRAW:
+            return
+        cv2.line(frame, (bx, by), (fx, fy), self._wt, 3)
+        cv2.circle(frame, (bx, by), 5, self._wt, -1)
+        cv2.circle(frame, (fx, fy), 5, self._wt, -1)
+        t = front_pct / 100.0
+        mx = int(bx + (fx - bx) * t)
+        my = int(by + (fy - by) * t)
+        _dashed_line(frame, com_px, (mx, my), self._com)
+        cv2.circle(frame, (mx, my), 7, self._com, -1)
+        cv2.circle(frame, (mx, my), 8, (0, 0, 0), 1)
+        self._text(frame, f"F {front_pct:.0f}%", (fx - 30, fy + 24), self._wt)
+        self._text(frame, f"B {100 - front_pct:.0f}%", (bx - 30, by + 24), self._wt)
+
+    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
         if record.keypoints is None or record.metrics is None:
             return frame

@@ -35,9 +142,20 @@
                 continue
             cv2.line(frame, (xa, ya), (xb, yb), self._sk, 2)

-        # com marker
         cx = int(record.metrics.com[0] * w)
         cy = int(record.metrics.com[1] * h)
+
+        # weight distribution baseline (before com marker so the marker sits on top)
+        self._draw_weight_line(frame, pts, (cx, cy),
+                               record.metrics.weight_dist_front_pct)
+
+        # torso lean angle line
+        if record.metrics.torso_lean_deg is not None:
+            self._draw_angle_line(frame, pts, record.metrics.torso_lean_deg)
+
+        # com marker
         cv2.circle(frame, (cx, cy), 8, self._com, -1)
         cv2.circle(frame, (cx, cy), 9, (0, 0, 0), 1)

@@ -45,15 +162,15 @@
         # primary text block (top-left)
         weight_f = record.metrics.weight_dist_front_pct
         lines = [f"F:{weight_f:.0f}%  B:{100 - weight_f:.0f}%"]
+        if record.metrics.torso_lean_deg is not None:
+            lines.append(f"lean: {record.metrics.torso_lean_deg:+.1f} deg")
         if record.metrics.knee_angle_left is not None:
             lines.append(f"L knee: {record.metrics.knee_angle_left:.0f} deg")
         if record.metrics.knee_angle_right is not None:
             lines.append(f"R knee: {record.metrics.knee_angle_right:.0f} deg")

         if self._show_secondary:
-            if record.metrics.torso_lean_deg is not None:
-                lines.append(f"torso lean: {record.metrics.torso_lean_deg:+.1f} deg")
             if record.metrics.shoulder_hip_rotation_deg is not None:
                 lines.append(f"sh-hip rot: {record.metrics.shoulder_hip_rotation_deg:+.1f} deg")
             if record.metrics.com_stability_score is not None:
@@ -62,10 +179,6 @@

         for i, text in enumerate(lines):
-            y = 20 + i * int(20 * self._font_scale * 1.6)
-            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
-                        self._font_scale, (0, 0, 0), 3, cv2.LINE_AA)
-            cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
-                        self._font_scale, self._tx, 1, cv2.LINE_AA)
+            y = int(33 * self._font_scale) + i * int(20 * self._font_scale * 1.6)
+            self._text(frame, text, (12, y))
         return frame
```

#### `src/surfanalysis/cli.py` 的變更

```diff
@@ -103,11 +103,11 @@
     json_path = Path(args.metrics_json)
     out_path = (
         Path(args.output) if args.output
-        else video.with_name(f"{video.stem}_annotated.mp4")
+        else video.with_name(f"{video.stem}.annotated{video.suffix}")
     )

     if not video.exists():
@@ -135,5 +135,8 @@
     renderer = OverlayRenderer(
         skeleton_color=args.skeleton_color,
         com_color=args.com_color,
+        angle_color=args.angle_color,
+        weight_color=args.weight_color,
         font_scale=args.font_scale,
         show_secondary=args.show_secondary,
+        stance=session.stance,
     )

     progress = None if args.quiet else tqdm(total=len(session.frames), unit="frame")
@@ -187,5 +190,7 @@
     r.add_argument("--font-scale", type=float, default=0.6)
     r.add_argument("--skeleton-color", type=str, default="#00FF00")
     r.add_argument("--com-color", type=str, default="#FFFF00")
+    r.add_argument("--angle-color", type=str, default="#FF40FF")
+    r.add_argument("--weight-color", type=str, default="#FFA500")
     r.add_argument("--quiet", action="store_true")
     r.set_defaults(func=cmd_render)
```

#### `tests/test_overlay.py` 的變更

```diff
@@ -52,5 +52,33 @@
     assert out.sum() == 0


+def test_overlay_angle_line_adds_pixels():
+    blank = np.zeros((480, 640, 3), dtype=np.uint8)
+    record = _frame_record()
+    no_lean = record.model_copy(deep=True)
+    no_lean.metrics.torso_lean_deg = None
+    renderer = OverlayRenderer()
+    out_with = renderer.draw(blank.copy(), record)
+    out_without = renderer.draw(blank.copy(), no_lean)
+    assert out_with.sum() > out_without.sum()
+
+
+def test_overlay_weight_line_skipped_when_feet_hidden():
+    blank = np.zeros((480, 640, 3), dtype=np.uint8)
+    record = _frame_record()
+    hidden = record.model_copy(deep=True)
+    pts = list(hidden.keypoints.points)
+    pts[31] = (pts[31][0], pts[31][1], pts[31][2], 0.1)
+    pts[32] = (pts[32][0], pts[32][1], pts[32][2], 0.1)
+    hidden.keypoints.points = pts
+    renderer = OverlayRenderer()
+    out_with = renderer.draw(blank.copy(), record)
+    out_without = renderer.draw(blank.copy(), hidden)
+    assert out_with.sum() > out_without.sum()
+
+
+def test_overlay_stance_swaps_weight_labels():
+    blank = np.zeros((480, 640, 3), dtype=np.uint8)
+    record = _frame_record()
+    out_regular = OverlayRenderer(stance="regular").draw(blank.copy(), record)
+    out_goofy = OverlayRenderer(stance="goofy").draw(blank.copy(), record)
+    assert not np.array_equal(out_regular, out_goofy)
+
+
 def test_overlay_show_secondary_adds_more_text():
     blank = np.zeros((480, 640, 3), dtype=np.uint8)
     record = _frame_record()
```

#### `tests/test_cli_render.py` 與 `tests/test_cli_extract.py` 的變更

```diff
# tests/test_cli_render.py 新增測試
def test_render_default_output_name_next_to_video(tiny_video_and_json):
    video, jpath = tiny_video_and_json
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(video), str(jpath), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = video.with_name("tiny.annotated.mp4")
    assert out.exists()
    assert out.stat().st_size > 0

# tests/test_cli_extract.py 新增測試
def test_extract_default_output_name_next_to_video(tiny_video: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = tiny_video.with_name("tiny.metrics.json")
    assert out.exists()
```

### 驗證與測試 (Verification and Testing)

- 單元測試：已成功通過 `67` 個單元測試項目。
- `Linting` 工具：已通過 `ruff check` 與靜態分析檢測。
