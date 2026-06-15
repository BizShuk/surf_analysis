import numpy as np

from surfanalysis.extraction.schema import FrameRecord, WaveMetrics
from surfanalysis.rendering.wave_overlay import WaveOverlay


def _record_with_wave():
    wave = WaveMetrics(
        view="facing", height=0.42, angle_deg=8.0, angle_kind="crest_tilt",
        confidence=0.8, angle_line=((0.1, 0.30), (0.9, 0.27)),
        height_top=(0.5, 0.29), height_bottom=(0.5, 0.71), horizon_deg=0.0,
    )
    return FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None,
                       metrics=None, wave=wave)


def test_wave_overlay_draws_when_wave_present():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = WaveOverlay().draw(blank, _record_with_wave())
    assert out.sum() > 0                       # drew lines + text even with keypoints=None


def test_wave_overlay_noop_without_wave():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    record = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None,
                         metrics=None, wave=None)
    out = WaveOverlay().draw(blank, record)
    assert out.sum() == 0


def test_wave_overlay_pct_changes_label_pixels():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out_norm = WaveOverlay(height_pct=False).draw(blank.copy(), _record_with_wave())
    out_pct = WaveOverlay(height_pct=True).draw(blank.copy(), _record_with_wave())
    assert not np.array_equal(out_norm, out_pct)
