"""Testing for PersistenceEntropy"""

import numpy as np

import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import integers, floats, composite

from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from giotto.diagrams import PersistenceEntropy, HeatKernel

X_pe = np.array([[[0, 1, 0], [2, 3, 0], [4, 6, 1], [2, 6, 1]]])


def test_pe_not_fitted():
    pe = PersistenceEntropy()

    with pytest.raises(NotFittedError):
        pe.transform(X_pe)


def test_pe_transform():
    pe = PersistenceEntropy()
    X_pe_res = np.array([[0.69314718, 0.63651417]])

    assert_almost_equal(pe.fit_transform(X_pe), X_pe_res)


pts_gen = arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=0.,
                    max_value=10),
    shape=(1, 20, 2)
)
dims_gen = arrays(
    dtype=np.int,
    elements=integers(min_value=0.,
                      max_value=3),
    shape=(1, 20, 1)
)


@given(pts_gen, dims_gen)
def test_hk_shape(pts, dims):
    n_values = 10
    hk = HeatKernel(sigma=1, n_values=n_values)

    X = np.concatenate([pts, dims], axis=2)
    num_dimensions = np.unique(dims)
    X_T = hk.fit(X).transform(X)

    assert X_T.shape == (X.shape[0], num_dimensions, n_values, n_values)


@given(pts_gen, dims_gen)
def test_hk_positive(pts, dims):
    n_values = 10
    hk = HeatKernel(sigma=1, n_values=n_values)

    X = np.concatenate([pts, dims], axis=2)
    num_dimensions = np.unique(dims)
    X_T = hk.fit(X).transform(X)

    assert np.all(X_T >= 0)


@given(pts_gen, dims_gen)
def test_hk_with_diag_points(pts, dims):
    n_values = 10
    hk = HeatKernel(sigma=1, n_values=n_values)

    X = np.concatenate([pts, np.zeros((pts.shape[0], pts.shape[1], 1))], axis=2)
    num_dimensions = np.unique(dims)
    diag_points = np.array([[[2, 2, 0], [3, 3, 0], [7, 7, 0]]])
    X_with_diag_points = np.concatenate([X, diag_points], axis=1)

    X_T, X_with_diag_points_T = [hk.fit(x).transform(x) for x in [X, X_with_diag_points]]

    assert_almost_equal(X_with_diag_points_T, X_T)
