import numpy as np
import pytest

from surfanalysis.extraction.engine import PoseEngine, MockEngine
from surfanalysis.extraction.schema import Keypoints


def test_pose_engine_is_abstract():
    with pytest.raises(TypeError):
        PoseEngine()  # type: ignore[abstract]


def test_mock_engine_returns_supplied_keypoints():
    kp = Keypoints(points=[(0.5, 0.5, 0.0, 0.9)] * 33, image_size=(640, 480))
    engine = MockEngine(sequence=[kp, None, kp])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert engine.detect(frame) == kp
    assert engine.detect(frame) is None
    assert engine.detect(frame) == kp


def test_mock_engine_reports_metadata():
    engine = MockEngine(sequence=[])
    info = engine.info()
    assert info.name == "mock"