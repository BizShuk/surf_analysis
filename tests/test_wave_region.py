import numpy as np

from surfanalysis.extraction.wave.region import region_from_mask


def test_region_from_mask_extracts_crest_and_base():
    mask = np.zeros((240, 320), dtype=np.uint8)
    # a wide band from row 60 (top/crest) to row 180 (bottom/base)
    mask[60:180, 40:280] = 255
    obs = region_from_mask(mask, horizon_deg=0.0)
    assert obs is not None
    assert obs.crest[1] < obs.base[1]                  # crest above base
    assert obs.base[1] - obs.crest[1] > 0.3            # spans a real height
    assert 0.0 <= obs.confidence <= 1.0


def test_region_from_mask_none_when_too_small():
    mask = np.zeros((240, 320), dtype=np.uint8)
    mask[10:14, 10:14] = 255  # tiny speck below min area
    assert region_from_mask(mask) is None


def test_region_from_mask_none_when_empty():
    assert region_from_mask(np.zeros((240, 320), dtype=np.uint8)) is None
