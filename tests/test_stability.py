import pytest

from surfanalysis.metrics.stability import StabilityWindow


def test_constant_com_yields_score_one():
    win = StabilityWindow(size=15, alpha=100.0)
    for _ in range(15):
        win.push((0.5, 0.5))
    score = win.score()
    assert score == pytest.approx(1.0)


def test_oscillating_com_yields_lower_score():
    win = StabilityWindow(size=15, alpha=105.0)
    for i in range(15):
        x = 0.5 + (0.1 if i % 2 == 0 else -0.1)
        win.push((x, 0.5))
    score = win.score()
    assert 0.0 < score < 0.5


def test_score_none_when_too_few_samples():
    win = StabilityWindow(size=15, alpha=100.0)
    for _ in range(5):
        win.push((0.5, 0.5))
    assert win.score() is None


def test_none_com_does_not_increment_count():
    win = StabilityWindow(size=15, alpha=100.0)
    for _ in range(5):
        win.push((0.5, 0.5))
    for _ in range(20):
        win.push(None)  # missed detections
    # only 5 valid samples remain in the rolling window; below threshold
    assert win.score() is None


def test_window_evicts_oldest():
    win = StabilityWindow(size=10, alpha=100.0)
    for _ in range(7):
        win.push((0.0, 0.0))
    win.push((10.0, 10.0))  # evicts older samples
    win.push((9.0, 9.0))
    win.push((8.0, 8.0))
    score = win.score()
    assert score is not None
    assert score < 0.99