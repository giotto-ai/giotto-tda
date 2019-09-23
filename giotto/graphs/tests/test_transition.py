"""Testing for TransitionGraph"""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from sklearn.exceptions import NotFittedError

from giotto.graphs import TransitionGraph

X_tg = np.array([[['a'], ['b'], ['c']]])

X_tg_res = np.array([
    csr_matrix((np.array([True] * 4),
                (np.array([0, 1, 1, 2]),
                 np.array([1, 0, 2, 1]))), shape=(3, 3))])


def test_transition_graph_not_fitted():
    tg = TransitionGraph()

    with pytest.raises(NotFittedError):
        tg.transform(X_tg)


def test_transition_graph_transform():
    tg = TransitionGraph()

    assert (tg.fit_transform(X_tg)[0] != X_tg_res[0]).nnz == 0


def test_parallel_transition_graph_transform():
    tg = TransitionGraph(n_jobs=1)
    tg_parallel = TransitionGraph(n_jobs=2)

    assert (tg.fit_transform(X_tg)[0] !=
            tg_parallel.fit_transform(X_tg)[0]).nnz == 0
