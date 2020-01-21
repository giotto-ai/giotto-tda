"""Testing for time series embedding."""
# License: GNU AGPLv3

import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.time_series import TakensEmbedding
from gtda.time_series import SlidingWindow

signal = np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)])

signal_embedded_search = np.array([[2., 2.47942554],
                                   [2.47942554, 2.84147098],
                                   [2.84147098, 2.99749499],
                                   [2.99749499, 2.90929743],
                                   [2.90929743, 2.59847214],
                                   [2.59847214, 2.14112001],
                                   [2.14112001, 1.64921677],
                                   [1.64921677, 1.2431975],
                                   [1.2431975, 1.02246988],
                                   [1.02246988, 1.04107573],
                                   [1.04107573, 1.29445967],
                                   [1.29445967, 1.7205845],
                                   [1.7205845, 2.21511999],
                                   [2.21511999, 2.6569866],
                                   [2.6569866, 2.93799998],
                                   [2.93799998, 2.98935825],
                                   [2.98935825, 2.79848711],
                                   [2.79848711, 2.41211849],
                                   [2.41211849, 1.92484888]])

signal_embedded_fixed = \
    np.array([[2., 2.47942554, 2.84147098, 2.99749499, 2.90929743],
              [2.47942554, 2.84147098, 2.99749499, 2.90929743, 2.59847214],
              [2.84147098, 2.99749499, 2.90929743, 2.59847214, 2.14112001],
              [2.99749499, 2.90929743, 2.59847214, 2.14112001, 1.64921677],
              [2.90929743, 2.59847214, 2.14112001, 1.64921677, 1.2431975],
              [2.59847214, 2.14112001, 1.64921677, 1.2431975, 1.02246988],
              [2.14112001, 1.64921677, 1.2431975, 1.02246988, 1.04107573],
              [1.64921677, 1.2431975, 1.02246988, 1.04107573, 1.29445967],
              [1.2431975, 1.02246988, 1.04107573, 1.29445967, 1.7205845],
              [1.02246988, 1.04107573, 1.29445967, 1.7205845, 2.21511999],
              [1.04107573, 1.29445967, 1.7205845, 2.21511999, 2.6569866],
              [1.29445967, 1.7205845, 2.21511999, 2.6569866, 2.93799998],
              [1.7205845, 2.21511999, 2.6569866, 2.93799998, 2.98935825],
              [2.21511999, 2.6569866, 2.93799998, 2.98935825, 2.79848711],
              [2.6569866, 2.93799998, 2.98935825, 2.79848711, 2.41211849],
              [2.93799998, 2.98935825, 2.79848711, 2.41211849, 1.92484888]])


def test_embedder_params():
    parameters_type = 'not_defined'
    embedder = TakensEmbedding(parameters_type=parameters_type)
    with pytest.raises(ValueError):
        embedder.fit(signal)


def test_embedder_not_fitted():
    embedder = TakensEmbedding()
    with pytest.raises(NotFittedError):
        embedder.transform(signal)


@pytest.mark.parametrize("parameters_type, expected",
                         [('search', signal_embedded_search),
                          ('fixed', signal_embedded_fixed)])
def test_embedder_transform(parameters_type, expected):
    embedder = TakensEmbedding(parameters_type=parameters_type)

    assert_almost_equal(embedder.fit_transform(signal), expected)


def test_window_params():
    window = SlidingWindow(width=0)
    with pytest.raises(ValueError):
        window.fit(signal)
