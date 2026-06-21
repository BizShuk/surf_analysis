import numpy as np

from surfanalysis.extraction.wave.ocean import HorizonAnchoredWaveEngine, wave_mask
from surfanalysis.extraction.wave.static import Mog2WaveEngine


def _ocean_frame() -> np.ndarray:
    """Bright sky on top, blue-green water band, white foam crest."""
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:90, :] = (235, 235, 235)            # sky
    img[90:200, :] = (160, 140, 40)          # blue-green water (BGR)
    img[95:115, :] = (250, 250, 250)         # foam crest band
    return img


def test_wave_mask_marks_foam_and_water():
    mask = wave_mask(_ocean_frame(), horizon_deg=0.0)
    assert mask.dtype == np.uint8
    assert mask.max() == 255
    assert mask[100, 160] == 255             # foam pixel is in the mask


def test_ocean_engine_detects_on_synthetic_wave():
    eng = HorizonAnchoredWaveEngine(view="facing", min_confidence=0.0)
    m = eng.detect(_ocean_frame(), 0.0)
    assert m is not None
    assert m.view == "facing"
    # schema 1.2: physical replaces fraction; engines don't compute it
    assert m.physical is None
    assert eng.info().name == "wave-ocean"


def test_ocean_engine_none_on_uniform_frame():
    eng = HorizonAnchoredWaveEngine(view="facing", min_confidence=0.5)
    assert eng.detect(np.full((240, 320, 3), 80, dtype=np.uint8), 0.0) is None


def test_static_engine_detects_moving_blob_after_warmup():
    eng = Mog2WaveEngine(view="facing", min_confidence=0.0, warmup=5)
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, size=(240, 320, 3), dtype=np.uint8)  # dark static venue
    result = None
    for i in range(12):
        frame = bg.copy()
        # a bright moving water band sweeping downward each frame
        top = 40 + i * 6
        frame[top:top + 80, 30:290] = 240
        result = eng.detect(frame, float(i))
    assert result is not None          # detects after warmup frames
    assert eng.info().name == "wave-static"
