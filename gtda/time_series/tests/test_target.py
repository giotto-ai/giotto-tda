"""Testing for time series labelling."""
# License: GNU AGPLv3

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from gtda.time_series.target import Labeller

signal = np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)])
X = np.tile(np.arange(10), reps=2).reshape(-1, 1)


def test_labeller_shape():
    width = 3
    labeller = Labeller(width=width, func=np.std, func_params={},
                        percentiles=None, n_steps_future=1)
    signal_transformed = labeller.fit_transform(signal)
    assert signal_transformed.shape == (20-(width+1)+1, 1)


def test_labeller_transformed():
    width = 5
    n_steps_future = 1
    labeller = Labeller(width=width, func=np.max, func_params={},
                        percentiles=None, n_steps_future=n_steps_future)
    x, y = labeller.fit_transform_resample(X, X)
    assert_almost_equal(x, X[(width-1):-n_steps_future])


def test_labeller_resampled():
    width = 5
    n_steps_future = 1
    labeller = Labeller(width=width, func=np.max, func_params={},
                        percentiles=None, n_steps_future=n_steps_future)
    x, y = labeller.fit_transform_resample(X, X)
    assert_almost_equal(y, np.array([5, 6, 7, 8, 9, 9, 9,
                                     9, 9, 9, 5, 6, 7, 8, 9]))


def test_labeller_with_percentage():
    width = 5
    n_steps_future = 1
    labeller = Labeller(width=width, func=np.max, func_params={},
                        percentiles=[100], n_steps_future=n_steps_future)
    labeller.fit(X)
    assert np.max(X) == labeller.thresholds_[0]


def test_labeller_invalid_percentage():
    labeller = Labeller(width=5, func=np.max, func_params={},
                        percentiles=[101], n_steps_future=2)
    with pytest.raises(ValueError):
        labeller.fit_transform_resample(X, signal)
