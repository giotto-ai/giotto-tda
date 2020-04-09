"""Testing for KNeighborsGraph."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from gtda.graphs import KNeighborsGraph

X_kng = np.array([[[0, 0], [1, 2], [4, 3], [6, 2]]])

X_kng_res = np.array([
    csr_matrix((np.array([1] * 4),
                (np.array([0, 1, 2, 3]),
                 np.array([1, 0, 3, 2]))), shape=(4, 4))])

X_kng_res_k2 = np.array([csr_matrix(np.array([[0., 1., 1., 0.],
                                              [1., 0., 1., 1.],
                                              [1., 1., 0., 1.],
                                              [0., 1., 1., 0.]]))])


def test_kng_not_fitted():
    kn_graph = KNeighborsGraph()

    with pytest.raises(NotFittedError):
        kn_graph.transform(X_kng)


@pytest.mark.parametrize(('n_neighbors', 'expected'),
                         [(1, X_kng_res), (2, X_kng_res_k2)])
def test_kng_transform(n_neighbors, expected):
    kn_graph = KNeighborsGraph(n_neighbors=n_neighbors)

    assert (kn_graph.fit_transform(X_kng)[0] != expected[0]).nnz == 0


def test_parallel_kng_transform():
    kn_graph = KNeighborsGraph(n_jobs=1, n_neighbors=2)
    kn_graph_parallel = KNeighborsGraph(n_jobs=2, n_neighbors=2)

    assert (kn_graph.fit_transform(X_kng)[0] !=
            kn_graph_parallel.fit_transform(X_kng)[0]).nnz == 0


def test_symmetric():
    kn_graph = KNeighborsGraph(n_neighbors=2)
    X_kng_transformed = kn_graph.fit_transform(X_kng)

    assert (X_kng_transformed[0] !=
            X_kng_transformed[0].transpose()).nnz == 0
