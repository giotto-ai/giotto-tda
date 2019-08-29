"""Testing for Stationarizer"""

import pytest

import numpy as np

from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import assert_raise_message

from giotto.time_series import Stationarizer

signal = np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)]).reshape(-1, 1)

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
    assert stationarizer.fit_transform(signal).all() == expected.all()
