import numpy as np

from surfanalysis.extraction.wave.motion import global_motion


def _texture(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(120, 160), dtype=np.uint8)


def test_global_motion_identical_is_near_zero():
    g = _texture(1)
    assert global_motion(g, g) < 0.5


def test_global_motion_detects_shift():
    g = _texture(2)
    shifted = np.roll(g, 8, axis=1)  # shift 8 px horizontally
    assert global_motion(g, shifted) > 3.0
