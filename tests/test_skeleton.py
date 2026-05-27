from surfanalysis.rendering.skeleton import SKELETON_EDGES, valid_indices


def test_edges_are_index_pairs():
    for a, b in SKELETON_EDGES:
        assert 0 <= a < 33
        assert 0 <= b < 33
        assert a != b


def test_includes_arm_chain():
    assert (11, 13) in SKELETON_EDGES or (13, 11) in SKELETON_EDGES
    assert (13, 15) in SKELETON_EDGES or (15, 13) in SKELETON_EDGES


def test_includes_leg_chain():
    assert (23, 25) in SKELETON_EDGES or (25, 23) in SKELETON_EDGES
    assert (25, 27) in SKELETON_EDGES or (27, 25) in SKELETON_EDGES


def test_valid_indices_complete():
    assert set(valid_indices()) == set(range(33))
