from gtda.externals import SparseRipsComplex

"""Test comes from
https://github.com/GUDHI/gudhi-devel/blob/master/src/python/test/test_rips_complex.py
"""


def test_empty_rips():
    rips_complex = SparseRipsComplex()
    del(rips_complex)


def test_sparse_filtered_rips_from_points():
    point_list = [[0, 0], [1, 0], [0, 1], [1, 1]]
    filtered_rips = SparseRipsComplex(points=point_list, max_edge_length=1.0,
                                      sparse=0.001)

    simplex_tree = filtered_rips.create_simplex_tree(max_dimension=1)

    assert simplex_tree._SimplexTree__is_defined() is True
    assert simplex_tree._SimplexTree__is_persistence_defined() is False

    assert simplex_tree.num_simplices() == 8
    assert simplex_tree.num_vertices() == 4


def test_rips_from_points():
    point_list = [[0, 0], [1, 0], [0, 1], [1, 1]]
    rips_complex = SparseRipsComplex(points=point_list, max_edge_length=42)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

    assert simplex_tree._SimplexTree__is_defined() is True
    assert simplex_tree._SimplexTree__is_persistence_defined() is False

    assert simplex_tree.num_simplices() == 10
    assert simplex_tree.num_vertices() == 4

    assert simplex_tree.get_filtration() == [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([3], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([1, 3], 1.0),
        ([2, 3], 1.0),
        ([1, 2], 1.4142135623730951),
        ([0, 3], 1.4142135623730951),
    ]
    assert simplex_tree.get_star([0]) == [
        ([0], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]
    assert simplex_tree.get_cofaces([0], 1) == [
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]


def test_rips_from_distance_matrix():
    from math import sqrt
    distance_matrix = [[0], [1, 0], [1, sqrt(2), 0], [sqrt(2), 1, 1, 0]]
    rips_complex = SparseRipsComplex(distance_matrix=distance_matrix,
                               max_edge_length=42)

    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

    assert simplex_tree._SimplexTree__is_defined() is True
    assert simplex_tree._SimplexTree__is_persistence_defined() is False

    assert simplex_tree.num_simplices() == 10
    assert simplex_tree.num_vertices() == 4

    assert simplex_tree.get_filtration() == [
        ([0], 0.0),
        ([1], 0.0),
        ([2], 0.0),
        ([3], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([1, 3], 1.0),
        ([2, 3], 1.0),
        ([1, 2], 1.4142135623730951),
        ([0, 3], 1.4142135623730951),
    ]
    assert simplex_tree.get_star([0]) == [
        ([0], 0.0),
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]
    assert simplex_tree.get_cofaces([0], 1) == [
        ([0, 1], 1.0),
        ([0, 2], 1.0),
        ([0, 3], 1.4142135623730951),
    ]
