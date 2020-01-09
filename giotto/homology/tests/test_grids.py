"""Testing for persistent homology on grids."""
# License : Apache 2.0

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from giotto.homology import CubicalPersistence

X = np.array([[[2., 2.47942554],
               [2.47942554, 2.84147098],
               [2.98935825, 2.79848711],
               [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])


def test_cp_not_fitted():
    cp = CubicalPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X)


def test_cp_transform():
    cp = CubicalPersistence()
    X_cp_res = np.array([[[1.9248489, 2.9893582, 0.],
                         [2., 2.79848711, 0], [0., 0., 1]]])

    assert_almost_equal(cp.fit_transform(X), X_cp_res)
