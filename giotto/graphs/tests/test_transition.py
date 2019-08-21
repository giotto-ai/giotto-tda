"""Testing for TransitionGraph"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.graphs import TransitionGraph

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
