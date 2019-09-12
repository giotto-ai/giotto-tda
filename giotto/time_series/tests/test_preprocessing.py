"""Testing for Resampler and Stationarizer"""

import pytest

import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.time_series import Resampler, Stationarizer

signal_array = np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)])
signal_df = pd.DataFrame(signal_array)
signal_df.index = pd.to_datetime(signal_df.index, utc=True, unit='d')

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


@pytest.fixture
def resampler():
    return Resampler(period=3)


def test_resampler_init():
    period = 2
    resampler = Resampler(period=period)
    assert resampler.get_params()['period'] == period


def test_resampler_not_fitted(resampler):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'Resampler',
                         resampler.transform, signal_df)


def test_resampler_not_valid():
    sampling_type = 'not_defined'
    resampler = Resampler(sampling_type=sampling_type,
                          sampling_period='2d',
                          sampling_times=None,
                          remove_weekends=False)
    msg = "The sampling type %s is not supported"
    assert_raise_message(ValueError, msg % sampling_type,
                         resampler.fit, signal_df)


@pytest.mark.parametrize("remove_weekends, expected",
                         [(False, signal_resampled),
                          (True, signal_resampled_no_weekends)])
def test_resampler_transform(fixed_resampler, remove_weekends, expected):
    resampler = Resampler(sampling_type='periodic', sampling_period='2d',
                          sampling_times=None, remove_weekends=remove_weekends)
    assert_almost_equal(resampler.fit_transform(signal_df), expected)
    assert_almost_equal(fixed_resampler.fit_transform(signal_df),
                        signal_resampled)


signal = signal_array.reshape(-1, 1)

signal_stationarized_return = np.array(
    [[0.19336154],
     [0.1274148],
     [0.05205146],
     [-0.03031576],
     [-0.11961848],
     [-0.21360416],
     [-0.29826475],
     [-0.32659273],
     [-0.21587689],
     [0.01787175],
     [0.19574495],
     [0.24766283],
     [0.22325449],
     [0.16630367],
     [0.09564785],
     [0.01718037],
     [-0.06820511],
     [-0.16017813],
     [-0.25314694]])

signal_stationarized_log_return = np.array(
    [[0.21487972],
     [0.13629497],
     [0.05345506],
     [-0.02986532],
     [-0.11298798],
     [-0.19359458],
     [-0.26102857],
     [-0.2826138],
     [-0.19546554],
     [0.01803338],
     [0.21783884],
     [0.28457069],
     [0.25264251],
     [0.18188605],
     [0.10053645],
     [0.01732966],
     [-0.06597978],
     [-0.14857355],
     [-0.22565794]])


@pytest.fixture
def stationarizer():
    return Stationarizer(stationarization_type='return')


def test_stationarizer_init():
    stationarization_type = 'return'
    stationarizer = Stationarizer(stationarization_type=stationarization_type)
    assert stationarizer.get_params()[
               'stationarization_type'] == stationarization_type


def test_stationarizer_not_fitted(stationarizer):
    msg = ("This %s instance is not fitted yet. Call \'fit\'"
           " with appropriate arguments before using this method.")
    assert_raise_message(NotFittedError, msg % 'Stationarizer',
                         stationarizer.transform, signal)


def test_stationarizer_not_valid():
    stationarization_type = 'not_defined'
    stationarizer = Stationarizer(stationarization_type=stationarization_type)
    msg = "The transformation type %s is not supported"
    assert_raise_message(ValueError, msg % stationarization_type,
                         stationarizer.fit, signal)


@pytest.mark.parametrize("stationarization_type, expected",
                         [('return', signal_stationarized_return),
                          ('log-return', signal_stationarized_log_return)])
def test_stationarizer_transform(stationarization_type, expected):
    stationarizer = Stationarizer(stationarization_type=stationarization_type)
    assert_almost_equal(stationarizer.fit_transform(signal), expected)
