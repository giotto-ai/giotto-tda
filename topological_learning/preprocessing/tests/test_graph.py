"""Testing for TransitionGraph, KNeighborsGraph and GraphGeodesicDistance"""

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from topological_learning.preprocessing import GraphGeodesicDistance
import pytest

X = np.array([
    np.array(
        [[0, 1, 3, 0, 0],
         [1, 0, 5, 0, 0],
         [3, 5, 0, 4, 0],
         [0, 0, 4, 0, 0],
         [0, 0, 0, 0, 0]]),
    np.array(
        [[0, 1, 3, 0, 0],
         [1, 0, 1, 0, 0],
         [3, 1, 0, 4, 0],
         [0, 0, 4, 0, 0],
         [0, 0, 0, 0, 0]])
])

X_target = np.array([
    [[0., 1., 3., 7., np.inf],
     [1., 0., 4., 8., np.inf],
     [3., 4., 0., 4., np.inf],
     [7., 8., 4., 0., np.inf],
     [np.inf, np.inf, np.inf, np.inf, 0.]],

    [[0., 1., 2., 6., np.inf],
     [1., 0., 1., 5., np.inf],
     [2., 1., 0., 4., np.inf],
     [6., 5., 4., 0., np.inf],
     [np.inf, np.inf, np.inf, np.inf, 0.]]
])


@pytest.fixture
def graph_geodesic_distance():
    return GraphGeodesicDistance(n_jobs=1)


def test_graph_geodesic_distance_init():
    graph_geodesic_distance = GraphGeodesicDistance(n_jobs=1)
    assert graph_geodesic_distance.n_jobs == 1
    assert graph_geodesic_distance.get_params()['n_jobs'] == 1


def test_not_fitted(graph_geodesic_distance):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'GraphGeodesicDistance',
                         graph_geodesic_distance.transform, X)


def test_graph_geodesic_distance_transform(graph_geodesic_distance):
    assert graph_geodesic_distance.fit_transform(X).all() == X_target.all()


def test_parallel_graph_geodesic_distance_transform():
    # FIXME: BrokenProcessPool when fit_transform with n_jobs > 1
    # graph_geodesic_distance = GraphGeodesicDistance(n_jobs=2)

    pass
