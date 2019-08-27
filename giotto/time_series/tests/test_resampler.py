"""Testing for Resampler"""

import pytest

import pandas as pd
import numpy as np
import datetime as dt

from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.time_series import Resampler

signal = pd.DataFrame(np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)]))
signal.index = pd.to_datetime(signal.index, utc=True, unit='d')

signal_resampled = np.array(
    [[2.],
     [2.84147098],
     [2.90929743],
     [2.14112001],
     [1.2431975],
     [1.04107573],
     [1.7205845],
     [2.6569866],
     [2.98935825],
     [2.41211849]])

signal_resampled_no_weekends = np.array(
    [[2.],
     [2.90929743],
     [2.14112001],
     [1.2431975],
     [1.7205845],
     [2.6569866],
     [2.41211849]])

sampling_times = pd.date_range(start='1/1/1970', periods=20, freq='d').tolist()


@pytest.fixture
def resampler():
    return Resampler(sampling_type='periodic', sampling_period='2d',
                     sampling_times=None, remove_weekends=False)


@pytest.fixture
def fixed_resampler():
    return Resampler(sampling_type='fixed', sampling_times=sampling_times,
                     remove_weekends=False)


def test_resampler_init():
    sampling_type = 'periodic'
    sampling_period = '2d'
    remove_weekends = False
    resampler = Resampler(sampling_type=sampling_type,
                          sampling_period=sampling_period,
                          sampling_times=None,
                          remove_weekends=remove_weekends)
    assert resampler.get_params()['sampling_type'] == sampling_type
    assert resampler.get_params()['sampling_period'] == sampling_period
    assert resampler.get_params()['sampling_times'] == [dt.time(0, 0, 0)]
    assert resampler.get_params()['remove_weekends'] == remove_weekends

    resampler = Resampler(sampling_type=sampling_type,
                          sampling_period=sampling_period,
                          sampling_times=[dt.time(0, 0, 0)],
                          remove_weekends=remove_weekends)

    assert resampler.get_params()['sampling_times'] == [dt.time(0, 0, 0)]


def test_resampler_not_fitted(resampler):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'Resampler',
                         resampler.transform, signal)


def test_resampler_not_valid():
    sampling_type = 'not_defined'
    resampler = Resampler(sampling_type=sampling_type,
                          sampling_period='2d',
                          sampling_times=None,
                          remove_weekends=False)
    msg = "The sampling type %s is not supported"
    assert_raise_message(ValueError, msg % sampling_type,
                         resampler.fit, signal)


@pytest.mark.parametrize("remove_weekends, expected",
                         [(False, signal_resampled),
                          (True, signal_resampled_no_weekends)])
def test_resampler_transform(fixed_resampler, remove_weekends, expected):
    resampler = Resampler(sampling_type='periodic', sampling_period='2d',
                          sampling_times=None, remove_weekends=remove_weekends)
    assert resampler.fit_transform(signal).all() == expected.all()
    assert fixed_resampler.fit_transform(
        signal).all() == signal_resampled.all()
