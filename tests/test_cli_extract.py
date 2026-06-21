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


@pytest.fixture
def wave_video(tmp_path: Path) -> Path:
    """Generate a video with sky / water / foam bands so the wave engine fires."""
    out = tmp_path / "wave.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out), fourcc, 15.0, (320, 240))
    for i in range(15):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:90, :] = (235, 235, 235)              # sky
        frame[90:200, :] = (160, 140, 40)            # blue-green water
        frame[95 + (i % 4):115 + (i % 4), :] = (250, 250, 250)  # foam band wobbles
        writer.write(frame)
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


def test_extract_wave_adds_wave_fields(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--wave",
         "--wave-engine", "static", "--view", "facing", "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.2"
    assert "wave" in data["frames"][0]          # key present (value may be null)
    assert data["wave_engine"]["name"] == "wave-static"


def test_extract_without_wave_unchanged(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.0"
    assert data["wave_summary"] is None


def test_extract_with_camera_height_creates_physical_metrics(wave_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(wave_video), "-o", str(out_json), "--wave",
         "--wave-engine", "ocean", "--view", "facing",
         "--camera-height-m", "3.0",
         "--focal-length-mm", "16.0", "--sensor-height-mm", "7.0",
         "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["schema_version"] == "1.2"
    ws = data["wave_summary"]
    assert ws["physical_status"] == "computed"
    assert ws["camera"] is not None
    assert ws["camera"]["camera_height_m"] == 3.0
    assert ws["camera"]["focal_length_mm"] == 16.0


def test_extract_without_camera_height_warns(wave_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(wave_video), "-o", str(out_json), "--wave",
         "--wave-engine", "ocean", "--view", "facing", "--quiet"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out_json.read_text())
    assert data["wave_summary"]["physical_status"] == "insufficient_metadata"
    # Warning is on stderr regardless of --quiet so users notice the gap.
    assert "--camera-height-m" in proc.stderr


def test_extract_camera_height_without_focal_errors(tiny_video: Path, tmp_path: Path):
    out_json = tmp_path / "out.metrics.json"
    proc = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(tiny_video), "-o", str(out_json), "--wave",
         "--camera-height-m", "3.0", "--quiet"],
        capture_output=True, text=True,
    )
    # EXIT_ENGINE = 2
    assert proc.returncode == 2
    assert "focal_length_mm" in proc.stderr or "sensor_height_mm" in proc.stderr
