# from .. import choose_n_farthest_points, pick_n_random_points, \
#     sparsify_point_set

import pytest

@pytest.mark.skip(reason="requires CGAL")
def test_simple_choose_n_farthest_points_with_a_starting_point():
    point_set = [[0, 1], [0, 0], [1, 0], [1, 1]]
    i = 0
    for point in point_set:
        # The iteration starts with the given starting point
        sub_set = choose_n_farthest_points(
            points=point_set, nb_points=1, starting_point=i
        )
        assert sub_set[0] == point_set[i]
        i = i + 1

    # The iteration finds then the farthest
    sub_set = choose_n_farthest_points(
        points=point_set, nb_points=2, starting_point=1
    )
    assert sub_set[1] == point_set[3]
    sub_set = choose_n_farthest_points(
        points=point_set, nb_points=2, starting_point=3
    )
    assert sub_set[1] == point_set[1]
    sub_set = choose_n_farthest_points(
        points=point_set, nb_points=2, starting_point=0
    )
    assert sub_set[1] == point_set[2]
    sub_set = choose_n_farthest_points(
        points=point_set, nb_points=2, starting_point=2
    )
    assert sub_set[1] == point_set[0]

    # Test the limits
    assert (
        choose_n_farthest_points(points=[], nb_points=0,
                                 starting_point=0) == []
    )
    assert (
        choose_n_farthest_points(points=[], nb_points=1,
                                 starting_point=0) == []
    )
    assert (
        choose_n_farthest_points(points=[], nb_points=0,
                                 starting_point=1) == []
    )
    assert (
        choose_n_farthest_points(points=[], nb_points=1,
                                 starting_point=1) == []
    )


@pytest.mark.skip(reason="requires CGAL")
def test_simple_choose_n_farthest_points_randomed():
    point_set = [[0, 1], [0, 0], [1, 0], [1, 1]]
    # Test the limits
    assert choose_n_farthest_points(points=[], nb_points=0) == []
    assert choose_n_farthest_points(points=[], nb_points=1) == []
    assert choose_n_farthest_points(points=point_set, nb_points=0) == []

    # Go furter than point set on purpose
    for iter in range(1, 10):
        sub_set = choose_n_farthest_points(points=point_set, nb_points=iter)
        for sub in sub_set:
            found = False
            for point in point_set:
                if point == sub:
                    found = True
            # Check each sub set point is existing in the point set
            assert found is True


@pytest.mark.skip(reason="requires CGAL")
def test_simple_pick_n_random_points():
    point_set = [[0, 1], [0, 0], [1, 0], [1, 1]]
    # Test the limits
    assert pick_n_random_points(points=[], nb_points=0) == []
    assert pick_n_random_points(points=[], nb_points=1) == []
    assert pick_n_random_points(points=point_set, nb_points=0) == []

    # Go furter than point set on purpose
    for iter in range(1, 10):
        sub_set = pick_n_random_points(points=point_set, nb_points=iter)
        print(5)
        for sub in sub_set:
            found = False
            for point in point_set:
                if point == sub:
                    found = True
            # Check each sub set point is existing in the point set
            assert found is True


@pytest.mark.skip(reason="requires CGAL")
def test_simple_sparsify_points():
    point_set = [[0, 1], [0, 0], [1, 0], [1, 1]]
    # Test the limits
    assert sparsify_point_set(points=point_set, min_squared_dist=0.0) ==\
        point_set
    assert sparsify_point_set(points=point_set, min_squared_dist=1.0) ==\
        point_set
    assert sparsify_point_set(points=point_set, min_squared_dist=2.0) == [
        [0, 1],
        [1, 0],
    ]
    assert sparsify_point_set(points=point_set, min_squared_dist=2.01) ==\
        [[0, 1]]
