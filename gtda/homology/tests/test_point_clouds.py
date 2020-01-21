"""Testing for persistent homology on grid."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.homology import VietorisRipsPersistence

X = np.array([[[2., 2.47942554],
               [2.47942554, 2.84147098],
               [2.98935825, 2.79848711],
               [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])


def test_vrp_params():
    metric = 'not_defined'
    vrp = VietorisRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X)


def test_vrp_not_fitted():
    vrp = VietorisRipsPersistence()

    with pytest.raises(NotFittedError):
        vrp.transform(X)


def test_vrp_transform():
    vrp = VietorisRipsPersistence()
    X_vrp_res = np.array([[[0., 0.43094373, 0],
                           [0., 0.5117411, 0],
                           [0., 0.60077095, 0],
                           [0., 0.62186205, 0],
                           [0.69093919, 0.80131882, 1]]])

    assert_almost_equal(vrp.fit_transform(X), X_vrp_res)
