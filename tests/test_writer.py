from pathlib import Path

import cv2
import numpy as np

from surfanalysis.rendering.writer import VideoSink


def test_video_sink_writes_file(tmp_path: Path):
    out = tmp_path / "out.mp4"
    sink = VideoSink(out, width=320, height=240, fps=15.0, codec="mp4v")
    for _ in range(10):
        sink.write(np.zeros((240, 320, 3), dtype=np.uint8))
    sink.close()
    assert out.exists()
    assert out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    cap.release()
