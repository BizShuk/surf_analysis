import pytest

from surfanalysis.extraction.wave.base import WaveObservation, to_wave_metrics
from surfanalysis.extraction.wave.camera import CameraModel
from surfanalysis.extraction.wave.physical import (
    PhysicalWaveComputer,
    WavelengthEstimate,
    WavelengthEstimator,
    score_confidence,
)


def _cam():
    # 3 m above water, focal 16 mm / sensor 7 mm at 1080 px tall
    return CameraModel.from_cli(
        camera_height_m=3.0, focal_length_mm=16.0,
        sensor_height_mm=7.0, image_height_px=1080,
    )


def _wave_at_15m_1m():
    """Wave at ~15 m depth, 1 m tall: crest 1 m above water, base on water.

    Built via to_wave_metrics so we exercise the WaveMetrics path the
    FrameAnalyzer actually feeds to PhysicalWaveComputer.
    """
    obs = WaveObservation(
        crest=(0.5, 684.0 / 1080.0),
        base=(0.5, 756.0 / 1080.0),
        crest_line=((0.1, 0.30), (0.9, 0.28)),
        face_line=((0.5, 0.7), (0.5, 0.3)),
        bbox=(0.1, 0.28, 0.8, 0.42),
        confidence=0.8,
        horizon_deg=0.0,
    )
    return to_wave_metrics(obs, "facing")


class _StubEstimator(WavelengthEstimator):
    """Test double: returns a fixed break height to exercise cross-validation."""

    def __init__(self, h: float | None) -> None:
        super().__init__()
        self._h = h

    def estimate_break_height(self) -> float | None:
        return self._h


def test_compute_height_with_camera_geometry():
    pc = PhysicalWaveComputer(camera_model=_cam(), horizon_deg=0.0)
    frame = pc.compute(_wave_at_15m_1m())
    assert frame.method == "camera_geometry"
    assert frame.height_m is not None
    assert frame.height_m == pytest.approx(1.0, rel=0.05)
    assert frame.confidence in ("high", "medium", "low", "unavailable")
    assert frame.crest_world is not None
    assert frame.trough_world is not None


def test_compute_height_skipped_without_camera():
    pc = PhysicalWaveComputer(camera_model=None, horizon_deg=0.0)
    frame = pc.compute(_wave_at_15m_1m())
    assert frame.method == "skipped"
    assert frame.height_m is None
    assert frame.crest_world is None
    assert frame.confidence == "unavailable"
    assert frame.reason is not None
    assert "--camera-height-m" in frame.reason  # CLI flag form, user-facing


def test_compute_height_invalid_projection_is_low_confidence():
    pc = PhysicalWaveComputer(camera_model=_cam(), horizon_deg=0.0)
    bad_obs = WaveObservation(
        crest=(0.5, 0.1),  # above horizon (y < 0.5 for pitch=0)
        base=(0.5, 0.9),
        crest_line=((0.0, 0.0), (1.0, 0.0)),
        face_line=((0.0, 0.0), (0.0, 1.0)),
        bbox=(0.0, 0.0, 1.0, 1.0),
        confidence=0.0,
        horizon_deg=0.0,
    )
    frame = pc.compute(to_wave_metrics(bad_obs, "facing"))
    # Degenerate input -> either None height OR low confidence, never a
    # plausible high number.
    assert frame.height_m is None or frame.confidence in ("low", "unavailable")


def test_wavelength_from_period():
    est = WavelengthEstimator()
    result = est.from_period_s(8.0)
    assert isinstance(result, WavelengthEstimate)
    assert result.wavelength_m == pytest.approx(100.0, rel=1e-2)
    assert 14.0 < result.h_upper_m < 14.5  # L/7
    assert 9.9 < result.h_lower_m < 10.1   # L/10
    assert result.period_s == 8.0


def test_wavelength_unavailable_for_zero_period():
    est = WavelengthEstimator()
    assert est.from_period_s(0.0) is None
    assert est.from_period_s(-1.0) is None


def test_score_confidence_both_aligned():
    assert score_confidence(1.0, 1.0) == "high"
    assert score_confidence(1.15, 1.0) == "high"  # delta 0.15 within 0.20


def test_score_confidence_diverged_medium():
    assert score_confidence(1.30, 1.0) == "medium"  # delta 0.30 within 0.50


def test_score_confidence_diverged_too_much_low():
    assert score_confidence(1.80, 1.0) == "low"  # delta 0.80


def test_score_confidence_single_path():
    assert score_confidence(1.0, None) == "medium"


def test_score_confidence_no_camera():
    assert score_confidence(None, 1.0) == "unavailable"
    assert score_confidence(None, None) == "unavailable"


def test_physical_computer_uses_wavelength_when_available():
    pc = PhysicalWaveComputer(
        camera_model=_cam(), horizon_deg=0.0,
        wavelength_estimator=_StubEstimator(0.99),  # ~1.0 m break height
    )
    frame = pc.compute(_wave_at_15m_1m())
    assert frame.method == "cross_validated"
    assert frame.height_m is not None
    assert frame.confidence == "high"
