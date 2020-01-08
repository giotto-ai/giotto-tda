from .. import WitnessComplex, StrongWitnessComplex

""" Test comes from
https://github.com/GUDHI/gudhi-devel/blob/master/src/python/test/test_witness_complex.py

"""


def test_empty_witness_complex():
    witness = WitnessComplex()
    assert witness._WitnessComplex__is_defined() is False


def test_witness_complex():
    nearest_landmark_table = [
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        [[1, 0], [2, 1], [3, 2], [4, 3], [0, 4]],
        [[2, 0], [3, 1], [4, 2], [0, 3], [1, 4]],
        [[3, 0], [4, 1], [0, 2], [1, 3], [2, 4]],
        [[4, 0], [0, 1], [1, 2], [2, 3], [3, 4]],
    ]

    witness_complex = \
        WitnessComplex(nearest_landmark_table=nearest_landmark_table)
    simplex_tree = witness_complex.create_simplex_tree(max_alpha_square=4.1)
    assert simplex_tree.num_vertices() == 5
    assert simplex_tree.num_simplices() == 31
    simplex_tree = witness_complex.create_simplex_tree(
        max_alpha_square=4.1, limit_dimension=2
    )
    assert simplex_tree.num_vertices() == 5
    assert simplex_tree.num_simplices() == 25


def test_strong_witness_complex():
    nearest_landmark_table = [
        [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]],
        [[1, 0], [2, 1], [3, 2], [4, 3], [0, 4]],
        [[2, 0], [3, 1], [4, 2], [0, 3], [1, 4]],
        [[3, 0], [4, 1], [0, 2], [1, 3], [2, 4]],
        [[4, 0], [0, 1], [1, 2], [2, 3], [3, 4]],
    ]

    strong_witness_complex = StrongWitnessComplex(
        nearest_landmark_table=nearest_landmark_table
    )
    simplex_tree = \
        strong_witness_complex.create_simplex_tree(max_alpha_square=4.1)
    assert simplex_tree.num_vertices() == 5
    assert simplex_tree.num_simplices() == 31
    simplex_tree = strong_witness_complex.create_simplex_tree(
        max_alpha_square=4.1, limit_dimension=2
    )
    assert simplex_tree.num_vertices() == 5
    assert simplex_tree.num_simplices() == 25
