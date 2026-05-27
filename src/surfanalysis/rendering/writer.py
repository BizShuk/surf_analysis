"""Thin wrapper around cv2.VideoWriter with FourCC handling."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class VideoSink:
    def __init__(
        self,
        path: Path | str,
        width: int,
        height: int,
        fps: float,
        codec: str = "mp4v",
    ) -> None:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise OSError(f"failed to open writer for {path} ({codec})")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()
