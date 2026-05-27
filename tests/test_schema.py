import pytest
from pydantic import ValidationError

from surfanalysis.extraction.schema import (
    EngineInfo,
    FrameMetrics,
    FrameRecord,
    Keypoints,
    SessionRecord,
    SessionSummary,
    SourceInfo,
)


def _kp_33():
    return [(0.5, 0.5, 0.0, 0.9)] * 33


def test_keypoints_requires_33_points():
    with pytest.raises(ValidationError):
        Keypoints(points=[(0.5, 0.5, 0.0, 0.9)] * 32, image_size=(1920, 1080))


def test_keypoints_accepts_exactly_33():
    kp = Keypoints(points=_kp_33(), image_size=(1920, 1080))
    assert len(kp.points) == 33


def test_frame_record_allows_none_keypoints():
    fr = FrameRecord(frame_index=0, timestamp_ms=0.0, keypoints=None, metrics=None)
    assert fr.keypoints is None


def test_session_record_round_trip_json():
    src = SourceInfo(path="x.mp4", width=1920, height=1080, fps=30.0,
                     total_frames=900, duration_ms=30000.0)
    eng = EngineInfo(name="mediapipe", version="0.10.x",
                     params={"model_complexity": 1, "min_detection_confidence": 0.5})
    summary = SessionSummary(frames_with_detection=0, frames_total=0,
                             detection_rate=0.0, metrics_aggregate={})
    session = SessionRecord(schema_version="1.0", source=src, engine=eng,
                            stance="regular", frames=[], summary=summary)
    json_str = session.model_dump_json()
    restored = SessionRecord.model_validate_json(json_str)
    assert restored.stance == "regular"
    assert restored.source.fps == 30.0


def test_frame_metrics_all_optional_except_com_and_weight():
    fm = FrameMetrics(com=(0.5, 0.5), weight_dist_front_pct=50.0)
    assert fm.knee_angle_left is None


def test_stance_must_be_regular_or_goofy():
    src = SourceInfo(path="x.mp4", width=1, height=1, fps=1.0,
                     total_frames=0, duration_ms=0.0)
    eng = EngineInfo(name="mediapipe", version="x", params={})
    summary = SessionSummary(frames_with_detection=0, frames_total=0,
                             detection_rate=0.0, metrics_aggregate={})
    with pytest.raises(ValidationError):
        SessionRecord(schema_version="1.0", source=src, engine=eng,
                      stance="sideways", frames=[], summary=summary)
