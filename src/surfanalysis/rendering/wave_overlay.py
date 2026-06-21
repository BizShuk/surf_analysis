"""Draw wave angle line + height bracket + HUD, decoupled from pose overlay.

Schema 1.2: height is reported in meters when camera metadata was provided;
otherwise the overlay displays a "needs --camera-height-m" hint instead of
silently rendering a fraction (which would conflict with the WSL "wave height"
contract — see CLAUDE.md § "Wave height semantics").
"""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.schema import FrameRecord
from surfanalysis.rendering.overlay import hex_to_bgr

_EMA_ALPHA = 0.3
_CONFIDENCE_COLOR: dict[str, tuple[int, int, int]] = {
    "high": (0, 255, 0),      # green
    "medium": (0, 200, 200),  # yellow-ish
    "low": (0, 100, 255),     # red-ish
    "unavailable": (160, 160, 160),  # gray
}


class WaveOverlay:
    def __init__(self, color: str = "#00E5FF", font_scale: float = 0.6) -> None:
        self._c = hex_to_bgr(color)
        self._fs = font_scale
        self._ema_a: float | None = None

    def _text(self, frame: np.ndarray, text: str, org: tuple[int, int],
              color: tuple[int, int, int] | None = None) -> None:
        col = color if color is not None else self._c
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._fs, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX,
                    self._fs, col, 1, cv2.LINE_AA)

    def draw(self, frame: np.ndarray, record: FrameRecord) -> np.ndarray:
        w = record.wave
        if w is None:
            return frame
        h, ww = frame.shape[:2]

        def px(p: tuple[float, float]) -> tuple[int, int]:
            return (int(p[0] * ww), int(p[1] * h))

        # angle line (crest tilt for facing / face steepness for side)
        cv2.line(frame, px(w.angle_line[0]), px(w.angle_line[1]), self._c, 2)
        # vertical height bracket — geometric endpoints still useful even
        # without a numeric readout
        cv2.line(frame, px(w.height_top), px(w.height_bottom), self._c, 2)

        # EMA-smoothed angle for the label (angle is still a clean fraction)
        self._ema_a = w.angle_deg if self._ema_a is None else \
            _EMA_ALPHA * w.angle_deg + (1 - _EMA_ALPHA) * self._ema_a

        label = "tilt" if w.angle_kind == "crest_tilt" else "face"
        x0 = max(0, ww - 280)
        self._text(frame, f"{label} {self._ema_a:+.0f} deg", (x0, 30))

        # Height readout — physical (meters) when available, hint otherwise.
        if w.physical is not None and w.physical.height_m is not None:
            conf_color = _CONFIDENCE_COLOR.get(w.physical.confidence, self._c)
            self._text(
                frame,
                f"H {w.physical.height_m:.2f} m ({w.physical.confidence})",
                (x0, 54),
                color=conf_color,
            )
        else:
            self._text(frame, "H: needs --camera-height-m", (x0, 54))
        return frame
