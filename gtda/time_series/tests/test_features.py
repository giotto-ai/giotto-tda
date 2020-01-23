"""Testing for feature creation from time series."""
# License: GNU AGPLv3

import numpy as np
from numpy.testing import assert_almost_equal
from gtda.time_series.features import PermutationEntropy
from itertools import product

X = np.ones((10, 200, 3)) # 10 samples, of 200 points embedded in a 3d space
X_unif = np.tile(np.random.randn(200,3), (10,1,1))

def test_entropy_shape():
    pe = PermutationEntropy()
    x_transformed = pe.fit_transform(X)
    assert x_transformed.shape == (X.shape[0], 1)


def test_entropy_unif():
    """Checkl that the process gives the same on the same samples"""
    pe = PermutationEntropy()
    x_transformed = pe.fit_transform(X_unif)
    are_equal = [a==b for a,b in product(x_transformed, x_transformed)]
    assert np.all(are_equal)

def test_uniform():
    pe = PermutationEntropy()
    x_transformed = pe.fit_transform(X)
    assert False
