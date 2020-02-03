"""Testing for PersistenceEntropy"""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from  hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats, integers

from gtda.diagrams import PersistenceEntropy, PersistentImage

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
    pi = PersistentImage(sigma=1)
    with pytest.raises(NotFittedError):
        pi.transform(diagram)


@given(X=arrays(dtype=np.float, unique=True,
                elements=integers(min_value=-1e10,
                                  max_value=1e6),
                shape=array_shapes(min_dims=1, max_dims=1, min_side=11)))
def test_pi_null(X):
    """Test that, if one trivial diagram (all pts on the diagonal) is provided,
    (along with a non-trivial one), then its pi is null"""
    pi = PersistentImage(sigma=1, n_values=10)
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
    pi = PersistentImage(sigma=1)
    diagrams = np.expand_dims(np.concatenate([
        np.sort(pts, axis=1), np.zeros((pts.shape[0], 1))],
        axis=1), axis=0)
    assert np.all(pi.fit_transform(diagrams) >= 0.)


