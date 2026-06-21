import numpy as np

from surfanalysis.extraction.schema import FrameRecord, PhysicalWaveFrame, WaveMetrics
from surfanalysis.rendering.wave_overlay import WaveOverlay


def _record_with_wave(physical: PhysicalWaveFrame | None = None):
    wave = WaveMetrics(
        view="facing", angle_deg=8.0, angle_kind="crest_tilt",
        confidence=0.8, angle_line=((0.1, 0.30), (0.9, 0.27)),
        height_top=(0.5, 0.29), height_bottom=(0.5, 0.71), horizon_deg=0.0,
        physical=physical,
    )
    return FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None,
                       metrics=None, wave=wave)


def _phys(h: float = 0.85, conf: str = "high"):
    return PhysicalWaveFrame(
        crest_world=(0.0, 0.5, 3.0),
        trough_world=(0.0, -0.35, 2.5),
        height_m=h,
        method="camera_geometry",
        confidence=conf,
    )


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


def test_wave_overlay_shows_meters_when_physical_present():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    out = WaveOverlay().draw(blank, _record_with_wave(physical=_phys(0.85, "high")))
    # pixels — confirm drawing happened and differs from the no-physical case
    rendered = out
    blank2 = np.zeros((480, 640, 3), dtype=np.uint8)
    out_no_phys = WaveOverlay().draw(blank2, _record_with_wave(physical=None))
    assert not np.array_equal(rendered, out_no_phys)
