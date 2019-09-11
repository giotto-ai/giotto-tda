"""Testing for VietorisRipsPersistence, CubicalPersistence and
PersistentEntropy"""

import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.homology import VietorisRipsPersistence, CubicalPersistence, \
    PersistentEntropy

X = np.array([[[2., 2.47942554],
               [2.47942554, 2.84147098],
               [2.98935825, 2.79848711],
               [2.79848711, 2.41211849],
               [2.41211849, 1.92484888]]])

X_vrp_transformed = {0: np.array([[[0., 0.43094373],
                                   [0., 0.5117411],
                                   [0., 0.60077095],
                                   [0., 0.62186205]]]),
                     1: np.array([[[0.69093919, 0.80131882]]])}

X_cp_transformed = {0: np.array([[[2., 2.79848711]]]),
                    1: np.array([[[0., 0.]]])}

X_pe = {0: np.array([[[0, 1], [2, 3]]]), 1: np.array([[[4, 6], [2, 6]]])}

X_pe_transformed = np.array([[0.69314718, 0.63651417]])


@pytest.fixture()
def vrp():
    return VietorisRipsPersistence()


def test_vrp_init():
    metric = 'precomputed'
    max_edge_length = 1
    homology_dimensions = [0]
    n_jobs = 1
    vrp = VietorisRipsPersistence(
        metric=metric,
        max_edge_length=max_edge_length,
        homology_dimensions=homology_dimensions,
        n_jobs=n_jobs)
    assert vrp.get_params()['metric'] == metric
    assert vrp.get_params()['max_edge_length'] == max_edge_length
    assert vrp.get_params()['homology_dimensions'] == homology_dimensions
    assert vrp.get_params()['n_jobs'] == n_jobs


def test_vrp_params():
    metric = 'not_defined'
    vrp = VietorisRipsPersistence(metric=metric)
    msg = 'The metric %s is not supported'
    assert_raise_message(ValueError, msg % metric, vrp.fit, X)


def test_vrp_not_fitted(vrp):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'VietorisRipsPersistence',
                         vrp.transform, X)


def test_vrp_transform(vrp):
    assert_almost_equal(vrp.fit_transform(X)[0], X_vrp_transformed[0])
    assert_almost_equal(vrp.fit_transform(X)[1], X_vrp_transformed[1])


@pytest.fixture()
def cp():
    return CubicalPersistence()


def test_cp_init():
    max_edge_length = 1
    homology_dimensions = [0]
    n_jobs = 2
    cp = CubicalPersistence(
        max_edge_length=max_edge_length,
        homology_dimensions=homology_dimensions,
        n_jobs=n_jobs)
    assert cp.get_params()['max_edge_length'] == max_edge_length
    assert cp.get_params()['homology_dimensions'] == homology_dimensions
    assert cp.get_params()['n_jobs'] == n_jobs


def test_cp_params():
    homology_dimensions = [0, 1, 2]
    cp = CubicalPersistence(homology_dimensions=homology_dimensions)
    msg = 'The homology_dimensions specified contains element(s) that are ' \
          'not within the range 0 to the dimension of the images.'
    assert_raise_message(ValueError, msg, cp.fit, X)


def test_cp_not_fitted(cp):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'CubicalPersistence',
                         cp.transform, X)


def test_cp_transform(cp):
    assert_almost_equal(cp.fit_transform(X)[0], X_cp_transformed[0])
    assert_almost_equal(cp.fit_transform(X)[1], X_cp_transformed[1])


@pytest.fixture()
def pe():
    return PersistentEntropy()


def test_pe_init():
    len_vector = 4
    n_jobs = 2
    pe = PersistentEntropy(
        len_vector=len_vector,
        n_jobs=n_jobs)
    assert pe.get_params()['len_vector'] == len_vector
    assert pe.get_params()['n_jobs'] == n_jobs


def test_pe_not_fitted(pe):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'PersistentEntropy',
                         pe.transform, X)


def test_pe_transform(pe):
    assert_almost_equal(pe.fit_transform(X_pe), X_pe_transformed)
