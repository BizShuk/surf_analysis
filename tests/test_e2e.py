import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


@pytest.fixture
def synthetic_clip(tmp_path: Path) -> Path:
    """Create a 2-second synthetic clip with a moving rectangle (no real pose)."""
    out = tmp_path / "synth.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w = cv2.VideoWriter(str(out), fourcc, 15.0, (320, 240))
    for i in range(30):
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        x = 50 + i * 4
        cv2.rectangle(frame, (x, 80), (x + 40, 160), (200, 200, 200), -1)
        w.write(frame)
    w.release()
    return out


def test_e2e_pipeline(synthetic_clip: Path, tmp_path: Path):
    metrics_json = tmp_path / "synth.metrics.json"
    annotated = tmp_path / "synth_annotated.mp4"

    r1 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract",
         str(synthetic_clip), "-o", str(metrics_json), "--quiet"],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    data = json.loads(metrics_json.read_text())
    assert data["summary"]["frames_total"] == 30

    r2 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render",
         str(synthetic_clip), str(metrics_json), "-o", str(annotated),
         "--show-secondary", "--quiet"],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert annotated.exists()
    assert annotated.stat().st_size > 0


def test_e2e_extract_wave_then_render(tmp_path):
    video = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (320, 240))
    rng = np.random.default_rng(0)
    bg = rng.integers(0, 60, size=(240, 320, 3), dtype=np.uint8)
    for i in range(15):
        frame = bg.copy()
        top = 40 + i * 4
        frame[top:top + 80, 30:290] = 240        # moving bright water band
        writer.write(frame)
    writer.release()

    jpath = tmp_path / "clip.metrics.json"
    r1 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "extract", str(video),
         "-o", str(jpath), "--wave", "--wave-engine", "static", "--quiet"],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    data = json.loads(jpath.read_text())
    assert data["schema_version"] == "1.1"

    out = tmp_path / "annotated.mp4"
    r2 = subprocess.run(
        [sys.executable, "-m", "surfanalysis.cli", "render", str(video), str(jpath),
         "-o", str(out), "--quiet"],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    assert out.exists() and out.stat().st_size > 0
