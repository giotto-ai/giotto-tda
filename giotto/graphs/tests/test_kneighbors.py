"""Testing for KNeighborsGraph"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.graphs import KNeighborsGraph

X_kng = np.array([
    [[1], [2], [3]],                    # 1 dim, 1 cluster
    [[1], [3], [4], [9], [10]],         # 1 dim, 2 clusters
    [[0, 0], [1, 2], [4, 3], [6, 2]]    # 2 dim, 2 clusters
])

X_kng_res = np.array([
    csr_matrix((np.array([1] * 4),
                (np.array([0, 1, 1, 2]),
                 np.array([1, 0, 2, 1]))), shape=(3, 3)),
    csr_matrix((np.array([1] * 6),
                (np.array([0, 1, 1, 2, 4, 3]),
                 np.array([1, 2, 0, 1, 3, 4]))), shape=(5, 5)),
    csr_matrix((np.array([1] * 4),
                (np.array([0, 1, 2, 3]),
                 np.array([1, 0, 3, 2]))), shape=(4, 4))
])


@pytest.fixture
def kneighbors_graph():
    return KNeighborsGraph(n_neighbors=2, n_jobs=1)


def test_kneighbors_graph_init():
    kneighbors_graph = KNeighborsGraph(n_neighbors=2, n_jobs=1)
    assert kneighbors_graph.n_jobs == 1
    assert kneighbors_graph.get_params()['n_jobs'] == 1


def test_kneighbors_graph_not_fitted(kneighbors_graph):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'KNeighborsGraph',
                         kneighbors_graph.transform, X_kng)


def test_kneighbors_graph_transform(kneighbors_graph):
    # FIXME: https://github.com/scipy/scipy/issues/10695
    # temporary fix: use scipy==1.2.1
    assert (kneighbors_graph.fit_transform(X_kng)[0] != X_kng_res[0]).nnz == 0
    assert (kneighbors_graph.fit_transform(X_kng)[1] != X_kng_res[1]).nnz == 0


def test_parallel_kneighbors_graph_transform():
    # FIXME: BrokenProcessPool when fit_transform with n_jobs > 1
    # kneighbors_graph_parallel = KNeighborsGraph(n_jobs=2)

    pass
