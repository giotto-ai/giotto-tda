"""Testing for PersistenceEntropy"""

import numpy as np

import pytest
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import integers, floats, composite

from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from giotto.diagrams import PersistenceEntropy, HeatKernel
from giotto.utils.validation import _validate_distinct

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
                    min_value=1.,
                    max_value=10),
    shape=(1, 20, 2)
)
dims_gen = arrays(
    dtype=np.int,
    elements=integers(min_value=0,
                      max_value=3),
    shape=(1, 20, 1)
)


def get_input(pts, dims):
    for p in pts:
        try:
            _validate_distinct([pts])
        except ValueError:
            p[0, 0:2] += 0.3 # add a distinct value
    X = np.concatenate([np.sort(pts, axis=2), dims], axis=2)
    return X


def test_all_pts_the_same():
    X = np.zeros((1, 4, 3))
    hk = HeatKernel(sigma=1)
    with pytest.raises(ValueError):
        X_ = hk.fit(X).transform(X)


@given(pts_gen, dims_gen)
def test_hk_shape(pts, dims):
    n_values = 10
    x = get_input(pts, dims)

    hk = HeatKernel(sigma=1, n_values=n_values)
    num_dimensions = len(np.unique(dims))
    x_t = hk.fit(x).transform(x)

    assert x_t.shape == (x.shape[0], num_dimensions, n_values, n_values)


@given(pts_gen, dims_gen)
def test_hk_positive(pts, dims):
    """ We expect the points above the PD-diagonal to be non-negative. (up to a numerical error)"""
    n_values = 10
    hk = HeatKernel(sigma=1, n_values=n_values)

    x = get_input(pts, dims)
    x_t = hk.fit(x).transform(x)

    assert np.all((np.tril(x_t[:, :, ::-1, :]) + 1e-13) >= 0.)


@given(pts_gen)
def test_hk_with_diag_points(pts):
    """Add points on the diagonal, and verify that we have the same results (on the same fitted values)."""
    n_values = 10
    hk = HeatKernel(sigma=1, n_values=n_values)

    x = get_input(pts, np.zeros((pts.shape[0], pts.shape[1], 1)))
    diag_points = np.array([[[2, 2, 0], [3, 3, 0], [7, 7, 0]]])
    x_with_diag_points = np.concatenate([x, diag_points], axis=1)

    # X_total = np.concatenate([X,X_with_diag_points], axis =0)
    hk = hk.fit(x_with_diag_points)

    x_t, x_with_diag_points_t = [hk.transform(x_) for x_ in [x, x_with_diag_points]]

    assert_almost_equal(x_with_diag_points_t, x_t, decimal=13)
