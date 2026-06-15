"""Draw wave angle line + height bracket + HUD, decoupled from pose overlay."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import FrameRecord
from surfanalysis.rendering.overlay import hex_to_bgr

_EMA_ALPHA = 0.3


class WaveOverlay:
    def __init__(self, color: str = "#00E5FF", font_scale: float = 0.6,
                 height_pct: bool = False) -> None:
        self._c = hex_to_bgr(color)
        self._fs = font_scale
        self._pct = height_pct
        self._ema_h: float | None = None
        self._ema_a: float | None = None

    def _text(self, frame: np.ndarray, text: str, org: tuple[int, int]) -> None:
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._fs, self._c, 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
        w = record.wave
        if w is None:
            return frame
        h, ww = frame.shape[:2]

        def px(p: tuple[float, float]) -> tuple[int, int]:
            return (int(p[0] * ww), int(p[1] * h))

        # angle line (crest tilt for facing / face steepness for side)
        cv2.line(frame, px(w.angle_line[0]), px(w.angle_line[1]), self._c, 2)
        # vertical height bracket
        cv2.line(frame, px(w.height_top), px(w.height_bottom), self._c, 2)

        # EMA-smoothed display values (JSON keeps the raw per-frame numbers)
        self._ema_h = w.height if self._ema_h is None else \
            _EMA_ALPHA * w.height + (1 - _EMA_ALPHA) * self._ema_h
        self._ema_a = w.angle_deg if self._ema_a is None else \
            _EMA_ALPHA * w.angle_deg + (1 - _EMA_ALPHA) * self._ema_a

        label = "tilt" if w.angle_kind == "crest_tilt" else "face"
        htxt = f"{self._ema_h * 100:.0f}%" if self._pct else f"{self._ema_h:.2f}"
        x0 = max(0, ww - 230)
        self._text(frame, f"wave H {htxt}", (x0, 30))
        self._text(frame, f"{label} {self._ema_a:+.0f} deg", (x0, 54))
        return frame
