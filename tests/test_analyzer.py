from pathlib import Path

import numpy as np
import pytest

from surfanalysis.extraction.analyzer import FrameAnalyzer
from surfanalysis.extraction.engine import MockEngine
from surfanalysis.extraction.schema import Keypoints, SourceInfo
from surfanalysis.extraction.wave.base import (
    MockWaveEngine,
    WaveObservation,
    to_wave_metrics,
)


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


def _wave_metrics():
    obs = WaveObservation(
        crest=(0.5, 0.3), base=(0.5, 0.7),
        crest_line=((0.1, 0.30), (0.9, 0.28)),
        face_line=((0.5, 0.7), (0.5, 0.3)),
        bbox=(0.1, 0.28, 0.8, 0.42), confidence=0.8, horizon_deg=0.0,
    )
    return to_wave_metrics(obs, "facing")


def test_analyzer_populates_wave_and_summary():
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=2, duration_ms=66.0)
    engine = MockEngine(sequence=[_placed_kp(), _placed_kp()])
    wave = MockWaveEngine([_wave_metrics(), None])
    analyzer = FrameAnalyzer(engine=engine, stance="regular", source=src, wave_engine=wave)
    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
    session = analyzer.run(frames_iter=iter(frames))

    assert session.schema_version == "1.2"
    assert session.frames[0].wave is not None
    assert session.frames[1].wave is None
    assert session.wave_summary is not None
    assert session.wave_summary.frames_detected == 1
    assert session.wave_engine.name == "mock-wave"


def test_analyzer_without_wave_engine_stays_v1_0():
    src = SourceInfo(path="x.mp4", width=640, height=480, fps=30.0,
                     total_frames=1, duration_ms=33.0)
    analyzer = FrameAnalyzer(engine=MockEngine([_placed_kp()]), stance="regular", source=src)
    session = analyzer.run(frames_iter=iter([np.zeros((480, 640, 3), dtype=np.uint8)]))
    assert session.schema_version == "1.0"
    assert session.wave_summary is None


def test_analyzer_populates_physical_when_camera_model_provided():
    from surfanalysis.extraction.wave.camera import CameraModel

    src = SourceInfo(path="x.mp4", width=1080, height=1080, fps=30.0,
                     total_frames=2, duration_ms=66.0)
    engine = MockEngine(sequence=[_placed_kp(), _placed_kp()])
    wave = MockWaveEngine([_wave_metrics(), _wave_metrics()])
    cam = CameraModel.from_cli(
        camera_height_m=3.0, focal_length_mm=16.0,
        sensor_height_mm=7.0, image_height_px=1080,
    )
    analyzer = FrameAnalyzer(
        engine=engine, stance="regular", source=src,
        wave_engine=wave, camera_model=cam,
    )
    frames = [np.zeros((1080, 1080, 3), dtype=np.uint8) for _ in range(2)]
    session = analyzer.run(frames_iter=iter(frames))

    assert session.wave_summary is not None
    assert session.wave_summary.physical_status == "computed"
    assert session.wave_summary.camera is not None
    assert session.wave_summary.camera.camera_height_m == 3.0
    # Each frame should have a PhysicalWaveFrame attached
    assert all(f.wave.physical is not None for f in session.frames if f.wave)
    # Without wavelength cross-check, single-source gets 'medium' confidence
    assert session.wave_summary.confidence in ("medium", "high", "low")


def test_analyzer_marks_insufficient_metadata_when_no_camera():
    src = SourceInfo(path="x.mp4", width=1080, height=1080, fps=30.0,
                     total_frames=1, duration_ms=33.0)
    engine = MockEngine(sequence=[_placed_kp()])
    wave = MockWaveEngine([_wave_metrics()])
    analyzer = FrameAnalyzer(
        engine=engine, stance="regular", source=src, wave_engine=wave,
        # camera_model=None: PhysicalWaveComputer skipped
    )
    frames = [np.zeros((1080, 1080, 3), dtype=np.uint8)]
    session = analyzer.run(frames_iter=iter(frames))

    assert session.wave_summary is not None
    assert session.wave_summary.physical_status == "insufficient_metadata"
    assert session.wave_summary.camera is None
    assert session.wave_summary.height_m_median is None
    assert session.wave_summary.confidence == "unavailable"
    # per-frame physical should still be set (skipped frames carry the reason)
    frame_wave = session.frames[0].wave
    assert frame_wave is not None
    assert frame_wave.physical is not None
    assert frame_wave.physical.method == "skipped"
