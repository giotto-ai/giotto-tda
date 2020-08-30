"""Testing for TransitionGraph."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from gtda.graphs import TransitionGraph

X_tg = np.array([[[1, 0], [2, 3], [5, 4]],
                 [[0, 1], [3, 2], [4, 5]]])

X_tg_res = np.array([
    csr_matrix((np.array([True] * 2),
                (np.array([0, 1]),
                 np.array([1, 0]))), shape=(2, 2)),
    csr_matrix((np.array([True] * 2),
                (np.array([0, 1]),
                 np.array([1, 0]))), shape=(2, 2)),
])


def test_transition_graph_not_fitted():
    tg = TransitionGraph()

    with pytest.raises(NotFittedError):
        tg.transform(X_tg)


def test_transition_graph_transform():
    tg = TransitionGraph()
    Xt = tg.fit_transform(X_tg)

    assert np.array_equal(Xt[0].toarray(), X_tg_res[0].toarray())
    assert np.array_equal(Xt[1].toarray(), X_tg_res[1].toarray())


def test_parallel_transition_graph_transform():
    tg = TransitionGraph(n_jobs=1)
    tg_parallel = TransitionGraph(n_jobs=2)

    Xt = tg.fit_transform(X_tg)
    Xt_parallel = tg_parallel.fit_transform(X_tg)

    assert np.array_equal(Xt[0].toarray(), Xt_parallel[0].toarray())
    assert np.array_equal(Xt[1].toarray(), Xt_parallel[1].toarray())
