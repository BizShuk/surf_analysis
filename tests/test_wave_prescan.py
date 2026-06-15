import numpy as np

from surfanalysis.extraction.wave import make_wave_engine
from surfanalysis.extraction.wave.ocean import HorizonAnchoredWaveEngine
from surfanalysis.extraction.wave.prescan import prescan
from surfanalysis.extraction.wave.static import Mog2WaveEngine


def _ocean_frame(shift: int) -> np.ndarray:
    # Real footage always carries texture; phaseCorrelate needs broadband
    # content to localize a shift, so add deterministic noise to the bands.
    rng = np.random.default_rng(42)
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:90, :] = (235, 235, 235)
    img[90:200, :] = (160, 140, 40)
    img[95:115, :] = (250, 250, 250)
    noise = rng.integers(0, 15, size=img.shape)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
    return np.roll(img, shift, axis=1)


def test_prescan_static_when_camera_fixed():
    frames = [_ocean_frame(0) for _ in range(8)]
    engine_name, _view = prescan(frames)
    assert engine_name == "static"


def test_prescan_ocean_when_camera_pans():
    frames = [_ocean_frame(i * 10) for i in range(8)]  # panning
    engine_name, _view = prescan(frames)
    assert engine_name == "ocean"


def test_make_wave_engine_returns_right_type():
    assert isinstance(make_wave_engine("ocean", "facing"), HorizonAnchoredWaveEngine)
    assert isinstance(make_wave_engine("static", "side"), Mog2WaveEngine)
