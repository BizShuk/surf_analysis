import pytest

from surfanalysis.metrics.wave_geometry import (
    CameraIntrinsics,
    angle_vs_horizon_deg,
    classify_view,
    line_angle_deg,
    median_p90,
    normalized_height,
    normalized_to_world_height,
    wave_height_meters,
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


def test_normalized_to_world_height_basic():
    # Camera at (0, 3 m, 0) looking horizontally, 1080 px tall, focal 1080 px.
    # For a point on the water plane (Y=0), its ray intersects the water at:
    #   alpha = arctan((y_px - cy) / f)
    #   Z = H / tan(theta + alpha)
    # Point at y_norm=1.0 → y_px=1080, cy=540 → alpha=arctan(0.5)≈26.565°
    # Z = 3 / tan(26.565°) = 6.0 m.  Y = 0 (on water plane).
    intr = CameraIntrinsics(
        camera_height_m=3.0,
        focal_pixels=1080.0,
        image_height_px=1080,
        pitch_deg=0.0,
    )
    X, Y, Z = normalized_to_world_height((0.5, 1.0), intr)
    assert Z == pytest.approx(6.0, rel=1e-3)
    assert Y == 0.0  # water-plane intersection, not point's world Y
    assert X == 0.0  # lateral not modeled (single-view)


def test_normalized_to_world_height_with_pitch():
    # 3 m above water, pitch 30° down, 1080 px tall, focal 1080 px.
    # Center pixel y=0.5 has alpha=0, so total below-horizon angle = pitch.
    # Z = 3 / tan(30°) = 3 / 0.5774 ≈ 5.196 m.
    intr = CameraIntrinsics(3.0, 1080.0, 1080, pitch_deg=30.0)
    _, _, Z = normalized_to_world_height((0.5, 0.5), intr)
    assert Z == pytest.approx(5.196, rel=1e-2)


def test_wave_height_from_observations():
    # Wave at ~15 m depth, 1 m tall: base on water, crest 1 m above.
    # For H=3, f=1080, image_height=1080 (so cy=540):
    #   base: y_px = cy + f*H/D = 540 + 1080*3/15 = 756 → y_norm ≈ 0.7
    #   crest: y_px = cy + f*(H-h)/D = 540 + 1080*2/15 = 684 → y_norm ≈ 0.633
    # wave_height_meters must round-trip close to 1.0 m.
    intr = CameraIntrinsics(3.0, 1080.0, 1080, pitch_deg=0.0)
    crest = (0.5, 684.0 / 1080.0)
    base = (0.5, 756.0 / 1080.0)
    h = wave_height_meters(crest, base, intr)
    assert h == pytest.approx(1.0, rel=0.05)


def test_normalized_to_world_height_invalid_intrinsics():
    intr = CameraIntrinsics(camera_height_m=0.0, focal_pixels=1080.0,
                            image_height_px=1080, pitch_deg=0.0)
    with pytest.raises(ValueError, match="camera_height_m"):
        normalized_to_world_height((0.5, 0.5), intr)
