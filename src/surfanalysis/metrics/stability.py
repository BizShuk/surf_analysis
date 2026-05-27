"""Rolling-window CoM stability scoring."""

from __future__ import annotations

from collections import deque

import numpy as np

_MIN_VALID_SAMPLES = 10


class StabilityWindow:
    """Track recent CoM positions and report a stability score in (0, 1]."""

    def __init__(self, size: int = 15, alpha: float = 100.0) -> None:
        self._buf: deque[tuple[float, float] | None] = deque(maxlen=size)
        self._alpha = alpha

    def push(self, com: tuple[float, float] | None) -> None:
        self._buf.append(com)

    def score(self) -> float | None:
        valid = [c for c in self._buf if c is not None]
        if len(valid) < _MIN_VALID_SAMPLES:
            return None
        arr = np.array(valid, dtype=np.float64)
        var = float(np.var(arr[:, 0]) + np.var(arr[:, 1]))
        return 1.0 / (1.0 + self._alpha * var)
