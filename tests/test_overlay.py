import numpy as np

from surfanalysis.extraction.schema import FrameMetrics, FrameRecord, Keypoints
from surfanalysis.rendering.overlay import OverlayRenderer, hex_to_bgr


def _frame_record():
    pts = [(0.0, 0.0, 0.0, 0.0)] * 33
    placements = {
        11: (0.45, 0.30), 12: (0.55, 0.30),
        13: (0.40, 0.40), 14: (0.60, 0.40),
        15: (0.35, 0.50), 16: (0.65, 0.50),
        23: (0.46, 0.55), 24: (0.54, 0.55),
        25: (0.46, 0.72), 26: (0.54, 0.72),
        27: (0.45, 0.92), 28: (0.55, 0.92),
        31: (0.45, 0.95), 32: (0.55, 0.95),
    }
    for i, (x, y) in placements.items():
        pts[i] = (x, y, 0.0, 0.9)
    kp = Keypoints(points=pts, image_size=(640, 480))
    metrics = FrameMetrics(
        com=(0.5, 0.6),
        weight_dist_front_pct=58.2,
        knee_angle_left=112.0,
        knee_angle_right=110.0,
        elbow_angle_left=140.0, elbow_angle_right=138.0,
        torso_lean_deg=-8.5,
        shoulder_hip_rotation_deg=12.3,
        com_stability_score=0.91,
    )
    return FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=kp, metrics=metrics)


def test_hex_to_bgr_basic():
    assert hex_to_bgr("#00FF00") == (0, 255, 0)
    assert hex_to_bgr("FF0000") == (0, 0, 255)


def test_overlay_returns_modified_frame():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    renderer = OverlayRenderer()
    out = renderer.draw(blank, _frame_record())
    assert out.shape == blank.shape
    assert out.sum() > 0


def test_overlay_no_metrics_returns_unchanged():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    record = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None, metrics=None)
    renderer = OverlayRenderer()
    out = renderer.draw(blank, record)
    assert out.sum() == 0


def test_overlay_show_secondary_adds_more_text():
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    record = _frame_record()
    r_main = OverlayRenderer(show_secondary=False)
    r_full = OverlayRenderer(show_secondary=True)
    out_main = r_main.draw(blank.copy(), record)
    out_full = r_full.draw(blank.copy(), record)
    assert out_full.sum() > out_main.sum()