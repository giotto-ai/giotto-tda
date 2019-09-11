"""Testing for VietorisRipsPersistence, CubicalPersistence and
PersistentEntropy"""

import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from giotto.homology import VietorisRipsPersistence, CubicalPersistence, \
    PersistentEntropy

X = np.array([[[2., 2.47942554],
               [2.47942554, 2.84147098],
               [2.98935825, 2.79848711],
               [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])

X_pe = {0: np.array([[[0, 1], [2, 3]]]), 1: np.array([[[4, 6], [2, 6]]])}


def test_vrp_params():
    metric = 'not_defined'
    vrp = VietorisRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit(X)


def test_vrp_not_fitted():
    vrp = VietorisRipsPersistence()

    with pytest.raises(NotFittedError):
        vrp.transform(X)


def test_vrp_transform():
    vrp = VietorisRipsPersistence()
    X_vrp_res = {0: np.array([[[0., 0.43094373],
                               [0., 0.5117411],
                               [0., 0.60077095],
                               [0., 0.62186205]]]),
                 1: np.array([[[0.69093919, 0.80131882]]])}

    assert_almost_equal(vrp.fit_transform(X)[0], X_vrp_res[0])
    assert_almost_equal(vrp.fit_transform(X)[1], X_vrp_res[1])


def test_cp_params():
    homology_dimensions = [0, 1, 2]
    cp = CubicalPersistence(homology_dimensions=homology_dimensions)
    msg = 'The homology_dimensions specified contains element\\(s\\) that ' \
          'are not within the range 0 to the dimension of the images.'

    with pytest.raises(ValueError, match=msg):
        cp.fit(X)


def test_cp_not_fitted():
    cp = CubicalPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X)


def test_cp_transform():
    cp = CubicalPersistence()
    X_cp_res = {0: np.array([[[2., 2.79848711]]]),
                1: np.array([[[0., 0.]]])}

    assert_almost_equal(cp.fit_transform(X)[0], X_cp_res[0])
    assert_almost_equal(cp.fit_transform(X)[1], X_cp_res[1])


def test_pe_not_fitted():
    pe = PersistentEntropy()

    with pytest.raises(NotFittedError):
        pe.transform(X_pe)


def test_pe_transform():
    pe = PersistentEntropy()
    X_pe_res = np.array([[0.69314718, 0.63651417]])

    assert_almost_equal(pe.fit_transform(X_pe), X_pe_res)
