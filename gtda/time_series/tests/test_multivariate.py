"""Testing for multivariate time series embedding."""
# License: GNU AGPLv3

import numpy as np
from numpy.testing import assert_almost_equal

from gtda.time_series import PearsonDissimilarity


def test_multivariate_transform():
    corr = PearsonDissimilarity(absolute_value=True)
    X0 = np.array([[1, 1, -3], [1.1, 2, 1]])
    X = X0.reshape(1, X0.shape[0], X0.shape[1])
    X0_res = np.ones((3, 3)) - np.abs(np.corrcoef(X0.T))

    assert_almost_equal(corr.fit_transform(X)[0], X0_res)
