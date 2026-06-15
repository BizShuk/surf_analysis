import numpy as np

from surfanalysis.extraction.wave.horizon import detect_horizon


def _sky_sea(tilt_rows: int = 0) -> np.ndarray:
    """320x240 BGR: bright sky on top, dark sea below a sharp boundary."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in range(320):
        cut = 120 + int(tilt_rows * (x / 320.0))
        img[:cut, x] = (235, 235, 235)
        img[cut:, x] = (40, 30, 20)
    return img


def test_detect_horizon_flat_is_near_zero():
    ang = detect_horizon(_sky_sea(tilt_rows=0))
    assert ang is not None
    assert abs(ang) < 3.0


def test_detect_horizon_none_on_uniform_frame():
    assert detect_horizon(np.full((240, 320, 3), 80, dtype=np.uint8)) is None
