"""One pre-scan pass: pick the engine (motion) and the view (geometry)."""

from __future__ import annotations

import cv2
import numpy as np

from surfanalysis.extraction.wave.motion import global_motion
from surfanalysis.extraction.wave.ocean import wave_mask
from surfanalysis.extraction.wave.region import region_from_mask
from surfanalysis.metrics.wave_geometry import classify_view

_STATIC_MOTION_PX = 1.5  # median global motion below this => fixed camera


def prescan(frames: list[np.ndarray], n: int = 15) -> tuple[str, str]:
    """Return (engine_name, view) from the first n frames; locked for the clip."""
    sample = frames[:n]
    if len(sample) < 2:
        return ("ocean", "facing")

    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in sample]
    motions = [global_motion(grays[i - 1], grays[i]) for i in range(1, len(grays))]
    engine_name = "static" if float(np.median(motions)) < _STATIC_MOTION_PX else "ocean"

    votes: list[str] = []
    for f in sample:
        obs = region_from_mask(wave_mask(f, None))
        if obs is not None:
            votes.append(classify_view(obs.bbox[2], obs.bbox[3], obs.crest_line))
    view = max(set(votes), key=votes.count) if votes else "facing"
    return (engine_name, view)
