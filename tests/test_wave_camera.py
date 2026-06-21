import pytest

from surfanalysis.extraction.wave.camera import CameraModel
from surfanalysis.metrics.wave_geometry import CameraIntrinsics


def test_camera_model_from_user_input():
    cam = CameraModel.from_cli(
        camera_height_m=3.0,
        focal_length_mm=16.0,
        sensor_height_mm=7.0,
        image_height_px=1080,
    )
    assert cam is not None
    assert cam.schema.camera_height_m == 3.0
    assert cam.schema.source == "user"


def test_camera_model_returns_none_without_height():
    assert CameraModel.from_cli(
        camera_height_m=None, focal_length_mm=16.0, sensor_height_mm=7.0,
        image_height_px=1080,
    ) is None


def test_camera_model_focal_pixels_computed():
    cam = CameraModel.from_cli(
        camera_height_m=3.0,
        focal_length_mm=16.0,
        sensor_height_mm=7.0,
        image_height_px=1080,
    )
    # focal_pixels = image_height_px * focal_length_mm / sensor_height_mm
    expected = 1080 * 16.0 / 7.0
    intr = cam.to_intrinsics(pitch_deg=0.0)
    assert isinstance(intr, CameraIntrinsics)
    assert intr.focal_pixels == pytest.approx(expected, rel=1e-4)
    assert intr.camera_height_m == 3.0


def test_camera_model_requires_focal_or_sensor():
    with pytest.raises(ValueError, match="focal_length_mm"):
        CameraModel.from_cli(
            camera_height_m=3.0, focal_length_mm=None, sensor_height_mm=None,
            image_height_px=1080,
        )


def test_camera_model_pitch_override():
    cam = CameraModel.from_cli(
        camera_height_m=3.0, focal_length_mm=16.0, sensor_height_mm=7.0,
        image_height_px=1080, pitch_deg=15.0,
    )
    intr_default = cam.to_intrinsics()  # uses stored pitch
    intr_override = cam.to_intrinsics(pitch_deg=30.0)
    assert intr_default.pitch_deg == pytest.approx(15.0)
    assert intr_override.pitch_deg == pytest.approx(30.0)


def test_prescan_physical_with_camera():
    from surfanalysis.extraction.wave.prescan import prescan_physical

    cam = CameraModel.from_cli(3.0, 16.0, 7.0, 1080)
    assert prescan_physical(cam, view="facing") == "computed"


def test_prescan_physical_without_camera():
    from surfanalysis.extraction.wave.prescan import prescan_physical

    assert prescan_physical(None, view="facing") == "insufficient_metadata"


def test_prescan_physical_unsupported_view():
    from surfanalysis.extraction.wave.prescan import prescan_physical

    cam = CameraModel.from_cli(3.0, 16.0, 7.0, 1080)
    assert prescan_physical(cam, view="overhead") == "unsupported_view"
