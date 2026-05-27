import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tiny_video_and_json(tmp_path: Path) -> tuple[Path, Path]:
    video = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video), fourcc, 15.0, (320, 240))
    for _ in range(15):
        writer.write(np.full((240, 320, 3), 80, dtype=np.uint8))
    writer.release()

    json_path = tmp_path / "tiny.metrics.json"
    subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(video), "-o", str(json_path), "--quiet"],
        check=True,
    )
    return video, json_path


def test_render_produces_output_video(tiny_video_and_json, tmp_path: Path):
    video, jpath = tiny_video_and_json
    out = tmp_path / "out.mp4"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(video), str(jpath), "-o", str(out), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
    assert out.stat().st_size > 0


def test_render_rejects_unknown_schema_version(tiny_video_and_json, tmp_path: Path):
    video, jpath = tiny_video_and_json
    data = json.loads(jpath.read_text())
    data["schema_version"] = "99.9"
    jpath.write_text(json.dumps(data))
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(video), str(jpath), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 4