"""Testing for TransitionGraph, KNeighborsGraph and GraphGeodesicDistance"""

import numpy as np
import pytest

from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message
from giotto.graphs import GraphGeodesicDistance, TransitionGraph

X_tg = np.array([
    [['a'], ['b'], ['c']],
    [['c'], ['a'], ['b']]
])

X_tg_res = np.array([
    csr_matrix((np.array([True] * 4),
                (np.array([0, 1, 1, 2]),
                 np.array([1, 0, 2, 1]))), shape=(3, 3)),
    csr_matrix((np.array([True] * 4),
                (np.array([0, 0, 1, 2]),
                 np.array([1, 2, 0, 0]))), shape=(3, 3))])


@pytest.fixture
def transition_graph():
    return TransitionGraph(n_jobs=1)


def test_transition_graph_init():
    transition_graph = TransitionGraph(n_jobs=1)
    assert transition_graph.n_jobs == 1
    assert transition_graph.get_params()['n_jobs'] == 1


def test_transition_graph_not_fitted(transition_graph):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'TransitionGraph',
                         transition_graph.transform, X_tg)


def test_transition_graph_transform(transition_graph):
    assert (transition_graph.fit_transform(X_tg)[0] != X_tg_res[0]).nnz == 0
    assert (transition_graph.fit_transform(X_tg)[1] != X_tg_res[1]).nnz == 0


def test_parallel_transition_graph_transform():
    # FIXME: BrokenProcessPool when fit_transform with n_jobs > 1
    # transition_graph_parallel = TransitionGraph(n_jobs=2)

    pass


X_ggd = np.array([
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

X_ggd_res = np.array([
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


def test_graph_geodesic_distance_not_fitted(graph_geodesic_distance):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'GraphGeodesicDistance',
                         graph_geodesic_distance.transform, X_ggd)


def test_graph_geodesic_distance_transform(graph_geodesic_distance):
    assert graph_geodesic_distance.fit_transform(
        X_ggd).all() == X_ggd_res.all()


def test_parallel_graph_geodesic_distance_transform():
    # FIXME: BrokenProcessPool when fit_transform with n_jobs > 1
    # graph_geodesic_distance = GraphGeodesicDistance(n_jobs=2)

    pass
