"""Testing for filter utils."""
# License : Apache 2.0

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from giotto.mapper import Eccentricity

X = np.array([[0, 0], [1, 2], [5, 6]])


def test_eccentricity_transform():
    ecc = Eccentricity()
    X_ecc = ecc.fit_transform(X)

    assert X_ecc.shape == (X.shape[0],)
