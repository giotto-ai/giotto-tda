"""Testing for feature creation from time series."""
# License: GNU AGPLv3

import numpy as np
from numpy.testing import assert_almost_equal
from gtda.time_series import PermutationEntropy
from itertools import product

X = np.ones((10, 200, 3))  # 10 samples, of 200 points embedded in a 3d space
X_unif = np.tile(np.random.randn(200, 3), (10, 1, 1))
X_3 = np.array([[[1, 2, 3],
                 [1, 2, 3],
                 [7, 6, 5]]])
pe_3 = 0.91829583


def test_entropy_shape():
    pe = PermutationEntropy()
    x_transformed = pe.fit_transform(X)
    assert x_transformed.shape == (X.shape[0], 1)


def test_entropy_unif():
    """Check that the process gives the same results on the same samples"""
    pe = PermutationEntropy()
    x_transformed = pe.fit_transform(X_unif)
    are_equal = [a == b for a, b in product(x_transformed, x_transformed)]
    assert np.all(are_equal)


def test_entropy_calc():
    pe = PermutationEntropy()
    x_transformed = pe.fit_transform(X_3)
    assert_almost_equal(x_transformed[0], pe_3, decimal=6)
