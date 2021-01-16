"""Testing for KNeighborsGraph."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.spatial.distance import pdist, squareform
from sklearn.exceptions import NotFittedError

from gtda.graphs import KNeighborsGraph

X_kng = np.array([[[0, 0],
                   [1, 2],
                   [4, 3],
                   [6, 2]]])
X_kng_list = list(X_kng)
dmat_0 = squareform(pdist(X_kng[0]))
X_kng_precomputed = dmat_0[None, :, :]
X_kng_precomputed_list = [dmat_0]

X_kng_res = [csr_matrix((np.array([1] * 4),
                         (np.array([0, 1, 2, 3]), np.array([1, 0, 3, 2]))))]

X_kng_res_k2 = [csr_matrix(np.array([[0, 1, 1, 0],
                                     [1, 0, 1, 0],
                                     [0, 1, 0, 1],
                                     [0, 1, 1, 0]]))]


def test_kng_not_fitted():
    kn_graph = KNeighborsGraph()

    with pytest.raises(NotFittedError):
        kn_graph.transform(X_kng)


@pytest.mark.parametrize(('X', 'metric'),
                         [(X_kng, 'euclidean'), (X_kng_list, 'euclidean'),
                          (X_kng_precomputed, 'precomputed'),
                          (X_kng_precomputed_list, 'precomputed')])
@pytest.mark.parametrize(('n_neighbors', 'expected'),
                         [(1, X_kng_res), (2, X_kng_res_k2)])
def test_kng_transform(X, metric, n_neighbors, expected):
    kn_graph = KNeighborsGraph(n_neighbors=n_neighbors, metric=metric)
    assert (kn_graph.fit_transform(X)[0] != expected[0]).nnz == 0


def test_parallel_kng_transform():
    kn_graph = KNeighborsGraph(n_jobs=1, n_neighbors=2)
    kn_graph_parallel = KNeighborsGraph(n_jobs=2, n_neighbors=2)

    assert (kn_graph.fit_transform(X_kng)[0] !=
            kn_graph_parallel.fit_transform(X_kng)[0]).nnz == 0
