"""Strategy-pattern pose engine abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from surfanalysis.extraction.schema import EngineInfo, Keypoints


class PoseEngine(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> Keypoints | None:
        """Return keypoints for a BGR frame, or None if no person detected.

        ``timestamp_ms`` is the frame's position in the source video. Engines
        with temporal tracking (e.g. MediaPipe VIDEO mode) use it to maintain
        pose continuity across frames; stateless engines may ignore it.
        """

    @abstractmethod
    def info(self) -> EngineInfo:
        """Return engine metadata for the JSON output."""


class MockEngine(PoseEngine):
    """Test double: replays a fixed sequence of detections."""

    def __init__(self, sequence: list[Keypoints | None]) -> None:
        self._sequence = sequence
        self._cursor = 0

    def detect(self, frame: np.ndarray, timestamp_ms: float = 0.0) -> Keypoints | None:  # noqa: ARG002
        if self._cursor >= len(self._sequence):
            return None
        out = self._sequence[self._cursor]
        self._cursor += 1
        return out

    def info(self) -> EngineInfo:
        return EngineInfo(name="mock", version="test", params={})
