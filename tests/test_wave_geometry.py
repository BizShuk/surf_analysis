import pytest

from surfanalysis.metrics.wave_geometry import (
    angle_vs_horizon_deg,
    classify_view,
    line_angle_deg,
    median_p90,
    normalized_height,
)


def test_line_angle_horizontal_is_zero():
    assert line_angle_deg((0.0, 0.5), (1.0, 0.5)) == pytest.approx(0.0)


def test_line_angle_normalizes_to_pm90():
    # a near-vertical line (image y grows downward) -> close to +90
    assert line_angle_deg((0.5, 0.1), (0.5, 0.9)) == pytest.approx(90.0)


def test_angle_vs_horizon_subtracts_roll():
    line = ((0.0, 0.50), (1.0, 0.40))  # tilts up to the right
    bare = angle_vs_horizon_deg(line, 0.0)
    rolled = angle_vs_horizon_deg(line, -5.0)
    assert rolled == pytest.approx(bare + 5.0)


def test_normalized_height_is_vertical_extent():
    assert normalized_height((0.5, 0.20), (0.5, 0.75)) == pytest.approx(0.55)


def test_classify_view_facing_when_wide_and_flat():
    assert classify_view(0.8, 0.4, ((0.1, 0.3), (0.9, 0.28))) == "facing"


def test_classify_view_side_when_steep():
    assert classify_view(0.6, 0.6, ((0.2, 0.8), (0.6, 0.3))) == "side"


def test_median_p90():
    med, p90 = median_p90([0.1, 0.2, 0.3, 0.4, 0.5])
    assert med == pytest.approx(0.3)
    assert p90 == pytest.approx(0.5)


def test_median_p90_empty():
    assert median_p90([]) == (0.0, 0.0)
