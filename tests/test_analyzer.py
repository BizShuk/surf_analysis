from pathlib import Path

import numpy as np
import pytest

from surfanalysis.extraction.analyzer import FrameAnalyzer
from surfanalysis.extraction.engine import MockEngine
from surfanalysis.extraction.schema import Keypoints, SourceInfo


def _placed_kp():
    pts = [(0.0, 0.0, 0.0, 0.0)] * 33
    placements = {
        0: (0.5, 0.10), 11: (0.45, 0.30), 12: (0.55, 0.30),
        13: (0.40, 0.40), 14: (0.60, 0.40),
        15: (0.35, 0.50), 16: (0.65, 0.50),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.45, 0.92), 28: (0.55, 0.92),
        31: (0.45, 0.95), 32: (0.55, 0.95),
    }
    for i, (x, y) in placements.items():
        pts[i] = (x, y, 0.0, 0.9)
    return Keypoints(points=pts, image_size=(640, 480))


def test_analyzer_assembles_session_record(tmp_path):
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=3, duration_ms=100.0)
    engine = MockEngine(sequence=[_placed_kp(), None, _placed_kp()])
    analyzer = FrameAnalyzer(engine=engine, stance="regular", source=src)
    fake_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(3)]
    session = analyzer.run(frames_iter=iter(fake_frames))

    assert session.source.total_frames == 3
    assert len(session.frames) == 3
    assert session.summary.frames_with_detection == 2
    assert session.summary.detection_rate == pytest.approx(2 / 3)
    assert session.frames[0].metrics is not None
    assert session.frames[1].metrics is None
    assert session.engine.name == "mock"


def test_analyzer_writes_json(tmp_path: Path):
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=1, duration_ms=33.0)
    engine = MockEngine(sequence=[_placed_kp()])
    analyzer = FrameAnalyzer(engine=engine, stance="regular", source=src)
    out = tmp_path / "metrics.json"
    session = analyzer.run(frames_iter=iter([np.zeros((480, 640, 3), dtype=np.uint8)]))
    out.write_text(session.model_dump_json(indent=2))
    assert out.read_text().startswith('{')