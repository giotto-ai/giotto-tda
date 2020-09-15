"""Testing for TransitionGraph."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from gtda.graphs import TransitionGraph

X_tg = np.array([[[1, 0],
                  [2, 3],
                  [5, 4]],
                 [[0, 1],
                  [3, 2],
                  [4, 5]],
                 [[5, 4],
                  [5, 4],
                  [5, 4]]])

X_tg_res = [
    csr_matrix((np.array([1] * 2), (np.array([0, 1]), np.array([1, 0])))),
    csr_matrix((np.array([1] * 2), (np.array([0, 1]), np.array([1, 0])))),
    csr_matrix(np.zeros((1, 1)))
    ]


def test_transition_graph_not_fitted():
    tg = TransitionGraph()

    with pytest.raises(NotFittedError):
        tg.transform(X_tg)


def test_transition_graph_transform():
    tg = TransitionGraph()
    Xt = tg.fit_transform(X_tg)

    for xt, x_tg_res in zip(Xt, X_tg_res):
        assert np.array_equal(xt.toarray(), x_tg_res.toarray())


def test_parallel_transition_graph_transform():
    tg = TransitionGraph(n_jobs=1)
    tg_parallel = TransitionGraph(n_jobs=2)

    Xt = tg.fit_transform(X_tg)
    Xt_parallel = tg_parallel.fit_transform(X_tg)

    for xt, xt_parallel in zip(Xt, Xt_parallel):
        assert np.array_equal(xt.toarray(), xt_parallel.toarray())
