import numpy as np
import pytest

from surfanalysis.extraction.mediapipe_engine import MediaPipeEngine


def test_mediapipe_engine_returns_none_on_blank_frame():
    pytest.importorskip("mediapipe")
    engine = MediaPipeEngine(model_complexity=0, min_detection_confidence=0.5)
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    kp = engine.detect(blank)
    assert kp is None
    engine.close()


def test_mediapipe_engine_info_includes_params():
    engine = MediaPipeEngine(model_complexity=1, min_detection_confidence=0.5)
    info = engine.info()
    assert info.name == "mediapipe"
    assert info.params["model_complexity"] == 1
    engine.close()
