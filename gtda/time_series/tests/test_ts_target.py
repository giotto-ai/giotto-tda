"""Testing for time series labelling."""
# License: GNU AGPLv3

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from gtda.time_series import Labeller

signal = np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)])
X = np.tile(np.arange(10), reps=2)


@pytest.mark.parametrize("size", [0, -1])
def test_labeller_params(size):
    labeller = Labeller(size=size)
    with pytest.raises(ValueError):
        labeller.fit(signal)


def test_labeller_shape():
    size = 4
    labeller = Labeller(size=size, func=np.std, func_params={},
                        percentiles=None, n_steps_future=1)
    signal_transformed = labeller.fit_transform(signal)
    assert signal_transformed.shape == (20 - size + 1,)


def test_labeller_transformed():
    size = 6
    n_steps_future = 1
    labeller = Labeller(size=size, func=np.max, func_params={},
                        percentiles=None, n_steps_future=n_steps_future)
    x, y = labeller.fit_transform_resample(X, X)
    assert_almost_equal(x, X[(size - 2):-n_steps_future])
    assert len(x) == len(y)


def test_labeller_resampled():
    size = 6
    labeller = Labeller(size=size, func=np.max, func_params={},
                        percentiles=None, n_steps_future=1)
    x, y = labeller.fit_transform_resample(X, X)
    assert_almost_equal(y, np.array([5, 6, 7, 8, 9, 9, 9,
                                     9, 9, 9, 5, 6, 7, 8, 9]))
    assert len(x) == len(y)

    # Test behaviour when n_steps_future = size - 1
    labeller.set_params(n_steps_future=size - 1)
    x, y = labeller.fit_transform_resample(X, X)
    assert_almost_equal(y, np.array([5, 6, 7, 8, 9, 9, 9,
                                     9, 9, 9, 5, 6, 7, 8, 9]))
    assert len(x) == len(y)

    # Test behaviour when n_steps_future > size - 1
    labeller.set_params(n_steps_future=size)
    x, y = labeller.fit_transform_resample(X, X)
    assert_almost_equal(y, np.array([6, 7, 8, 9, 9, 9, 9,
                                     9, 9, 5, 6, 7, 8, 9]))
    assert len(x) == len(y)


def test_labeller_with_percentage():
    size = 6
    n_steps_future = 1
    labeller = Labeller(size=size, func=np.max, func_params={},
                        percentiles=[100], n_steps_future=n_steps_future)
    labeller.fit(X)
    assert np.max(X) == labeller.thresholds_[0]


def test_labeller_invalid_percentage():
    labeller = Labeller(size=6, func=np.max, func_params={},
                        percentiles=[101], n_steps_future=2)
    with pytest.raises(ValueError):
        labeller.fit_transform_resample(X, signal)
