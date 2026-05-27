import math

import numpy as np
import pytest

from surfanalysis.metrics.geometry import (
    angle_at_vertex,
    distance,
    midpoint,
    project_onto_segment,
    wrap_to_180,
)


def test_midpoint_basic():
    a = np.array([0.0, 0.0])
    b = np.array([2.0, 4.0])
    assert np.allclose(midpoint(a, b), [1.0, 2.0])


def test_distance_pythagorean():
    a = np.array([0.0, 0.0])
    b = np.array([3.0, 4.0])
    assert distance(a, b) == pytest.approx(5.0)


def test_angle_at_vertex_straight_line():
    a = np.array([0.0, 0.0])
    b = np.array([1.0, 0.0])
    c = np.array([2.0, 0.0])
    assert angle_at_vertex(a, b, c) == pytest.approx(180.0)


def test_angle_at_vertex_right_angle():
    a = np.array([0.0, 1.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    assert angle_at_vertex(a, b, c) == pytest.approx(90.0)


def test_angle_at_vertex_degenerate_returns_nan_or_zero():
    a = np.array([0.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([1.0, 0.0])
    result = angle_at_vertex(a, b, c)
    assert math.isnan(result)


def test_project_onto_segment_midpoint():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    p = np.array([5.0, 3.0])
    assert project_onto_segment(p, a, b) == pytest.approx(0.5)


def test_project_onto_segment_clamps_below():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    p = np.array([-5.0, 0.0])
    assert project_onto_segment(p, a, b) == pytest.approx(0.0)


def test_project_onto_segment_clamps_above():
    a = np.array([0.0, 0.0])
    b = np.array([10.0, 0.0])
    p = np.array([20.0, 0.0])
    assert project_onto_segment(p, a, b) == pytest.approx(1.0)


def test_wrap_to_180_positive_overflow():
    assert wrap_to_180(190.0) == pytest.approx(-170.0)


def test_wrap_to_180_negative_overflow():
    assert wrap_to_180(-190.0) == pytest.approx(170.0)


def test_wrap_to_180_in_range_unchanged():
    assert wrap_to_180(45.0) == pytest.approx(45.0)
