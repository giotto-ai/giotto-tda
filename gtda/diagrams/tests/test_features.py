"""Testing for features"""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers

from gtda.diagrams import PersistenceEntropy, HeatKernel, PersistenceImage

diagram = np.array([[[0, 1, 0], [2, 3, 0], [4, 6, 1], [2, 6, 1]]])


def test_pe_not_fitted():
    pe = PersistenceEntropy()

    with pytest.raises(NotFittedError):
        pe.transform(diagram)


def test_pe_transform():
    pe = PersistenceEntropy()
    diagram_res = np.array([[0.69314718, 0.63651417]])

    assert_almost_equal(pe.fit_transform(diagram), diagram_res)


def test_pi_not_fitted():
    pi = PersistenceImage(sigma=1)
    with pytest.raises(NotFittedError):
        pi.transform(diagram)


@given(X=arrays(dtype=np.float, unique=True,
                elements=integers(min_value=-1e10,
                                  max_value=1e6),
                shape=array_shapes(min_dims=1, max_dims=1, min_side=11)))
def test_pi_null(X):
    """Test that, if one trivial diagram (all pts on the diagonal) is provided,
    (along with a non-trivial one), then its pi is null"""
    pi = PersistenceImage(sigma=1, n_bins=10)
    X = np.append(X, 1 + X[-1])
    diagrams = np.expand_dims(np.stack([X, X,
                                        np.zeros((X.shape[0],),
                                                 dtype=int)]).transpose(),
                              axis=0)
    diagrams = np.repeat(diagrams, 2, axis=0)
    diagrams[1, :, 1] += 1

    assert_almost_equal(pi.fit_transform(diagrams)[0], 0)


@given(pts=arrays(dtype=np.float, unique=True,
                  elements=floats(allow_nan=False,
                                  allow_infinity=False,
                                  min_value=-1e10,
                                  max_value=1e6),
                  shape=(20, 2)))
def test_pi_positive(pts):
    pi = PersistenceImage(sigma=1)
    diagrams = np.expand_dims(np.concatenate([
        np.sort(pts, axis=1), np.zeros((pts.shape[0], 1))],
        axis=1), axis=0)
    assert np.all(pi.fit_transform(diagrams) >= 0.)


pts_gen = arrays(
    dtype=np.float,
    elements=floats(allow_nan=False,
                    allow_infinity=False,
                    min_value=1.,
                    max_value=10),
    shape=(1, 20, 2), unique=True
)
dims_gen = arrays(
    dtype=np.int,
    elements=integers(min_value=0,
                      max_value=3),
    shape=(1, 20, 1)
)


def _validate_distinct(X):
    unique_values = [np.unique(x[0:2, :]) for x in X]
    if np.any([len(u) < 2 for u in unique_values]):
        raise ValueError("There should be at least two distinct points"
                         "in the persistent diagrams:" +
                         "now, only {} is present".format(*unique_values))
    return 0


def get_input(pts, dims):
    for p in pts:
        try:
            _validate_distinct([pts])
        except ValueError:
            p[0, 0:2] += 0.3
            # add a distinct value, if not provided by hypothesis
    X = np.concatenate([np.sort(pts, axis=2), dims], axis=2)
    return X


def test_all_pts_the_same():
    X = np.zeros((1, 4, 3))
    hk = HeatKernel(sigma=1)
    with pytest.raises(IndexError):
        _ = hk.fit(X).transform(X)


@given(pts_gen, dims_gen)
def test_hk_shape(pts, dims):
    n_bins = 10
    x = get_input(pts, dims)

    hk = HeatKernel(sigma=1, n_bins=n_bins)
    num_dimensions = len(np.unique(dims))
    x_t = hk.fit(x).transform(x)

    assert x_t.shape == (x.shape[0], num_dimensions, n_bins, n_bins)


@given(pts_gen, dims_gen)
def test_hk_positive(pts, dims):
    """ We expect the points above the PD-diagonal to be non-negative,
    (up to a numerical error)"""
    n_bins = 10
    hk = HeatKernel(sigma=1, n_bins=n_bins)

    x = get_input(pts, dims)
    x_t = hk.fit(x).transform(x)

    assert np.all((np.tril(x_t[:, :, ::-1, :]) + 1e-13) >= 0.)


@given(pts_gen)
def test_hk_with_diag_points(pts):
    """Add points on the diagonal, and verify that we have the same results
    (on the same fitted values)."""
    n_bins = 10
    hk = HeatKernel(sigma=1, n_bins=n_bins)

    x = get_input(pts, np.zeros((pts.shape[0], pts.shape[1], 1)))
    diag_points = np.array([[[2, 2, 0], [3, 3, 0], [7, 7, 0]]])
    x_with_diag_points = np.concatenate([x, diag_points], axis=1)

    # X_total = np.concatenate([X,X_with_diag_points], axis =0)
    hk = hk.fit(x_with_diag_points)

    x_t, x_with_diag_points_t = [hk.transform(x_)
                                 for x_ in [x, x_with_diag_points]]

    assert_almost_equal(x_with_diag_points_t, x_t, decimal=13)
