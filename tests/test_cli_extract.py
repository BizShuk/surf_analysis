import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def tiny_video(tmp_path: Path) -> Path:
    """Generate a 1-second synthetic 320x240 video at 15 fps."""
    out = tmp_path / "tiny.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (320, 240))
    for _ in range(15):
        writer.write(np.full((240, 320, 3), 80, dtype=np.uint8))
    writer.release()
    return out


def test_extract_writes_valid_json(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.0"
    assert data["source"]["fps"] == 15.0
    assert len(data["frames"]) == 15
    assert data["summary"]["frames_total"] == 15


def test_extract_default_output_name_next_to_video(tiny_video: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    out = tiny_video.with_name("tiny.metrics.json")
    assert out.exists()


def test_extract_missing_file_returns_exit_1(tmp_path: Path):
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tmp_path / "nope.mp4"), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 1
