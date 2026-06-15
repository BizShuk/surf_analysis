import numpy as np

from surfanalysis.extraction.wave.base import (
    MockWaveEngine,
    WaveObservation,
    to_wave_metrics,
)


def _obs():
    return WaveObservation(
        crest=(0.52, 0.29),
        base=(0.52, 0.71),
        crest_line=((0.1, 0.30), (0.9, 0.27)),
        face_line=((0.52, 0.71), (0.52, 0.29)),
        bbox=(0.1, 0.27, 0.8, 0.44),
        confidence=0.8,
        horizon_deg=0.0,
    )


def test_to_wave_metrics_facing_uses_crest_line():
    m = to_wave_metrics(_obs(), "facing")
    assert m.view == "facing"
    assert m.angle_kind == "crest_tilt"
    assert m.angle_line == ((0.1, 0.30), (0.9, 0.27))
    assert m.height == _obs().base[1] - _obs().crest[1]


def test_to_wave_metrics_side_uses_face_line():
    m = to_wave_metrics(_obs(), "side")
    assert m.angle_kind == "face_steepness"
    assert m.angle_line == ((0.52, 0.71), (0.52, 0.29))


def test_mock_wave_engine_replays_sequence():
    seq = [to_wave_metrics(_obs(), "facing"), None]
    eng = MockWaveEngine(seq)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    assert eng.detect(frame, 0.0) is not None
    assert eng.detect(frame, 1.0) is None
    assert eng.detect(frame, 2.0) is None  # past end
    assert eng.info().name == "mock-wave"
