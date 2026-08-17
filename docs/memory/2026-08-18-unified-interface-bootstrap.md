# 2026-08-18 — 統一介面補齊 (Unified Interface Bootstrap)

## 背景 (Context)

`~/projects/` 全域稽核發現本 repo 缺 `AGENTS.md` symlink、`docs/terminology.md`
與 `docs/memory/`。本次只補這三件。`README.md`、`CLAUDE.md`、`README.todo`、
`plans/`、`docs/tutorials/`、`docs/pose-engines.md` 與所有原始碼一律未動。

## 補齊項目 (What Was Added)

- `AGENTS.md` — 指向 `CLAUDE.md` 的相對 symlink。
- `docs/terminology.md` — 術語表。狀態值一節直接取自
  `extraction/schema.py` 的 `Literal` 定義,不寫美化版說法。
- `docs/memory/` — 本檔即目錄的第一份留底。

## 值得留底的現況 (Findings Worth Keeping)

- `CLAUDE.md 的「Wave height semantics」一節已經過期,而且是會誤導人的那種過期。`
  它以「現況」開頭寫道 `wave_summary.height_median` / `height_p90` 是畫面占比
  (正規化 0-1) 而非物理浪高,並把「改用公尺欄位」列為`目標`。
  但這件事`已經做完了`:schema 1.2 移除了那兩個 fraction 欄位,
  改成 `height_m_median` / `height_m_p90`(公尺)。
  驗證:`grep -rn 'height_median\|height_p90' src/` 只剩 schema docstring 裡
  標註 REMOVED 的那行,以及 `tests/test_schema.py` 用來斷言欄位已消失的負向測試;
  實際欄位在 `schema.py:122-123` 與 `analyzer.py:163-164`,名字都是 `height_m_*`。
  該節後半段 (四條推算路徑、實作分層約束) 仍然有效,`只有「現況」那段需要改寫`。
- 承上,`外部筆記若記著「height_median 是螢幕占比」也已作廢`。現在的語意風險反過來了:
  欄位名帶 `_m` 就是公尺,但`拿不到相機參數時值是 None 而不是 0` —
  下游必須檢查 `physical_status`,不能看到 `None` 就當成零或省略。
- `浪高的正確性依賴 pitch,而 pitch 是參數不是開關`。公式
  `h = H × (1 - tan(θ+α_crest)/tan(θ+α_base))` 裡 θ 直接參與運算,
  README 記錄的敏感度掃描顯示同一段影片在 15°–75° 之間會得到 0.30–0.60 m 的不同答案
  (U 形,30°–60° 是甜蜜點)。也就是`絕對值不可盡信,趨勢才可靠`。
- `引擎自動選擇目前是錯的,而且錯得很安靜`。`prescan` 依`相機運動量`挑引擎,
  固定機位的造浪池因此被判給 `static`(MOG2),而 MOG2 會把穩定水流學進背景,
  偵測率掉到個位數。正解是改用`前景穩定度`啟發式,不是相機運動;
  在那之前,造浪池素材一律顯式加 `--wave-engine ocean`。
- `schema 破壞是刻意大聲失敗的`。`IncompatibleSchemaError` 不做靜默降級,
  遇到舊版 `metrics.json` 直接要求重跑 extract。看起來不友善,但這是 2026-06-21 的
  明確決定 — 不要「順手」加相容層。
- 型別覆蓋是`部分的`:`mypy --strict` 只跑 `metrics/`,`extraction/` 與 `rendering/`
  未納入。改動後兩者時不要以為型別檢查會擋下錯誤。
- 本 repo 的 venv 有已知地雷 (目錄改名後 shebang 失效),用 `.venv/bin/python -m surfanalysis.cli`
  或 `python -m` 形式呼叫最穩,不要直接跑 `.venv/bin/surf`。
