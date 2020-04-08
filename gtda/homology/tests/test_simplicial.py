"""Testing for simplicial persistent homology."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.homology import VietorisRipsPersistence, SparseRipsPersistence, \
    EuclideanCechPersistence

pio.renderers.default = 'plotly_mimetype'

X = np.array([[[2., 2.47942554], [2.47942554, 2.84147098],
               [2.98935825, 2.79848711], [2.79848711, 2.41211849],
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


X_vrp_res = np.array([[[0., 0.43094373, 0], [0., 0.5117411, 0],
                       [0., 0.60077095, 0], [0., 0.62186205, 0],
                       [0.69093919, 0.80131882, 1]]])


def test_vrp_transform():
    vrp = VietorisRipsPersistence()

    assert_almost_equal(vrp.fit_transform(X), X_vrp_res)


def test_vrp_list_of_arrays():
    X_2 = np.array([[0., 1.], [1., 2.]])
    X_list = [X[0].copy(), X_2]
    vrp = VietorisRipsPersistence()
    vrp.fit(X_list)


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_vrp_fit_transform_plot(hom_dims):
    VietorisRipsPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_srp_params():
    metric = 'not_defined'
    vrp = SparseRipsPersistence(metric=metric)

    with pytest.raises(ValueError):
        vrp.fit_transform(X)


def test_srp_not_fitted():
    srp = SparseRipsPersistence()

    with pytest.raises(NotFittedError):
        srp.transform(X)


X_srp_res_2 = np.array([[[0., 0.43094373, 0], [0., 0.5117411, 0],
                         [0., 0.60077095, 0], [0., 0.62186205, 0],
                         [0.69093919, 0.80131882, 1]]])


@pytest.mark.parametrize("epsilon, point_clouds, expected",
                         [(0.0, X, X_vrp_res),
                          (1.0, X, X_srp_res_2)])
def test_srp_transform(epsilon, point_clouds, expected):
    srp = SparseRipsPersistence(epsilon=epsilon)

    assert_almost_equal(np.sort(srp.fit_transform(point_clouds), axis=1),
                        np.sort(expected, axis=1))


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_srp_fit_transform_plot(hom_dims):
    SparseRipsPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)


def test_cp_params():
    coeff = 'not_defined'
    cp = EuclideanCechPersistence(coeff=coeff)

    with pytest.raises(TypeError):
        cp.fit_transform(X)


def test_cp_not_fitted():
    cp = EuclideanCechPersistence()

    with pytest.raises(NotFittedError):
        cp.transform(X)


X_cp_res = np.array(
    [[[0., 0.31093103, 0.], [0., 0.30038548, 0.],
      [0., 0.25587055, 0.], [0., 0.21547186, 0.],
      [0.34546959, 0.41473758, 1.], [0.51976681, 0.55287585, 1.],
      [0.26746207, 0.28740871, 1.], [0.52355742, 0.52358794, 1.],
      [0.40065941, 0.40067135, 1.], [0.45954496, 0.45954497, 1.]]])


def test_cp_transform():
    cp = EuclideanCechPersistence()

    assert_almost_equal(cp.fit_transform(X), X_cp_res)


@pytest.mark.parametrize('hom_dims', [None, (0,), (1,), (0, 1)])
def test_cp_fit_transform_plot(hom_dims):
    EuclideanCechPersistence().fit_transform_plot(
        X, sample=0, homology_dimensions=hom_dims)
