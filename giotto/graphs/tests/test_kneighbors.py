"""Testing for KNeighborsGraph"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, SparseEfficiencyWarning
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message
import warnings

from giotto.graphs import KNeighborsGraph

X_kng = np.array([
    [[1], [2], [3]],  # 1 dim, 1 cluster
    [[1], [3], [4], [9], [10]],  # 1 dim, 2 clusters
    [[0, 0], [1, 2], [4, 3], [6, 2]]  # 2 dim, 2 clusters
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

X_kng_res_k2 = np.array(
    [csr_matrix(np.array([[0., 1., 1.],
                          [1., 0., 1.],
                          [1., 1., 0.]])),
     csr_matrix(np.array([[0., 1., 1., 0., 0.],
                          [1., 0., 1., 0., 0.],
                          [1., 1., 0., 1., 1.],
                          [0., 0., 1., 0., 1.],
                          [0., 0., 1., 1., 0.]])),
     csr_matrix(np.array([[0., 1., 1., 0.],
                          [1., 0., 1., 1.],
                          [1., 1., 0., 1.],
                          [0., 1., 1., 0.]]))]
)


@pytest.fixture
def kn_graph():
    return KNeighborsGraph(n_neighbors=1, n_jobs=1)


@pytest.fixture
def kn_graph_k2():
    return KNeighborsGraph(n_neighbors=2, n_jobs=1)


@pytest.fixture
def kn_graph_parallel():
    return KNeighborsGraph(n_neighbors=2, n_jobs=1)


def test_kneighbors_graph_init():
    kneighbors_graph = KNeighborsGraph(n_neighbors=1, n_jobs=1)
    assert kneighbors_graph.n_jobs == 1
    assert kneighbors_graph.get_params()['n_jobs'] == 1


def test_kneighbors_graph_not_fitted(kn_graph):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'KNeighborsGraph',
                         kn_graph.transform, X_kng)


def test_kneighbors_graph_transform(kn_graph, kneighbors_graph_k2):
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    for i in range(len(X_kng)):
        assert (kn_graph.fit_transform(X_kng)[i] != X_kng_res[i]).nnz == 0
        assert (kneighbors_graph_k2.fit_transform(X_kng)[i] !=
                X_kng_res_k2[i]).nnz == 0


def test_parallel_kneighbors_graph_transform(kneighbors_graph_k2,
                                             kn_graph_parallel):
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    for i in range(len(X_kng)):
        assert (kneighbors_graph_k2.fit_transform(X_kng)[i] !=
                kn_graph_parallel.fit_transform(X_kng)[i]).nnz == 0


def test_symmetric(kn_graph, kn_graph_k2,
                   kn_graph_parallel):
    warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)
    for i in range(len(X_kng)):
        assert (kn_graph.fit_transform(X_kng)[i] !=
                kn_graph.fit_transform(X_kng)[i].transpose()).nnz == 0
        assert (kn_graph_k2.fit_transform(X_kng)[i] !=
                kn_graph_k2.fit_transform(X_kng)[i].transpose()).nnz == 0
        assert (kn_graph_parallel.fit_transform(X_kng)[i] !=
                kn_graph_parallel.fit_transform(X_kng)[i].transpose()).nnz == 0
