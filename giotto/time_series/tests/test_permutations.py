"""Testing for time series ordinal representation."""
# License : Apache 2.0

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

from giotto.time_series import OrdinalRepresentation, Entropy


def test_ordinal_rep_transform():
    orep = OrdinalRepresentation()
    X = np.array([[[1, 2, 3]]])
    X_res = np.array([[[0, 1, 2]]])

    assert_equal(orep.fit_transform(X), X_res)


def test_entropy_transform():
    entropy = Entropy()
    X = np.array([[[1, 2, 3], [1, 2, 3], [2, 4, 6]]])
    X_res = np.array([[0.63651417]])

    assert_almost_equal(entropy.fit_transform(X), X_res)
