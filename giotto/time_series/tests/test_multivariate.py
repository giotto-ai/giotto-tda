"""Testing for multivariate time series embedding."""
# License : Apache 2.0

import numpy as np
from numpy.testing import assert_almost_equal

from giotto.time_series import PearsonCorrelation


def test_multivariate_transform():
    corr = PearsonCorrelation()
    X = np.array([[1, 1, -3], [1.1, 2, 1]])
    X_res = np.ones((3, 3)) - np.abs(np.corrcoef(X.T))

    assert_almost_equal(corr.fit_transform(X), X_res)
