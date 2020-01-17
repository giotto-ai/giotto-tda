from .. import CechComplex

""" Test comes from

"""


def test_minimal_cech():
    points = [[1, 2]]
    cech_complex = CechComplex(points, 0)
    del(cech_complex)


def test_example_from_points():
    from math import sqrt
    verification_filtration = [([0], 0.0), ([1], 0.0), ([2], 0.0), ([3], 0.0),
                               ([4], 0.0), ([5], 0.0), ([6], 0.0), ([7], 0.0),
                               ([8], 0.0), ([9], 0.0), ([10], 0.0),
                               ([4, 9], 0.5), ([6, 9], 0.5),
                               ([1, 10], 0.55901700258255),
                               ([4, 10], 0.55901700258255),
                               ([0, 1], 0.7071067690849304),
                               ([0, 2], 0.7071067690849304),
                               ([2, 3], 0.7071067690849304),
                               ([4, 6], 0.7071067690849304),
                               ([4, 6, 9], 0.7071067811865476), ([1, 2], 1.0),
                               ([0, 1, 2], 1.0), ([1, 4], 1.0), ([3, 5], 1.0),
                               ([3, 7], 1.0), ([5, 7], 1.0), ([6, 7], 1.0),
                               ([6, 8], 1.0), ([7, 8], 1.0), ([1, 4, 10], 1.0)]

    points = []
    points.append([1., 0.])                  # 0
    points.append([0., 1.])                  # 1
    points.append([2., 1.])                  # 2
    points.append([3., 2.])                  # 3
    points.append([0., 3.])                  # 4
    points.append([3. + sqrt(3.), 3.])       # 5
    points.append([1., 4.])                  # 6
    points.append([3., 4.])                  # 7
    points.append([2., 4. + sqrt(3.)])       # 8
    points.append([0., 4.])                  # 9
    points.append([-0.5, 2.])                # 10

    cech_complex = CechComplex(points, 1.0)
    stree = cech_complex.create_simplex_tree(max_dimension=2)

    assert stree.dimension() == 2
    assert stree.num_simplices() == 30
    assert stree.num_vertices() == 11
    assert stree.get_filtration() == verification_filtration
