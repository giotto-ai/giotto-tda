"""Testing for time series embedding."""
# License: GNU AGPLv3

from functools import partial

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.time_series import SlidingWindow, TakensEmbedding, \
    time_delay_embedding

pio.renderers.default = 'plotly_mimetype'

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

y = np.arange(signal.shape[0])

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


def test_embedder_resample():
    embedder = TakensEmbedding(parameters_type='fixed', time_delay=3,
                               dimension=2, stride=3)
    embedder.fit(signal)
    y_resampled = embedder.resample(y)
    assert_almost_equal(y_resampled, y[np.arange(4, 20, 3)])


def test_window_params():
    windows = SlidingWindow(width=0)
    with pytest.raises(ValueError):
        windows.fit(signal)


def test_window_transform():
    windows = SlidingWindow(width=3, stride=2)
    X_windows = windows.fit_transform(signal_embedded_search)
    assert (X_windows.shape == (8, 4, 2))


def test_window_resample():
    windows = SlidingWindow(width=3, stride=2)
    windows.fit(y)
    y_resampled = windows.resample(y)
    assert_almost_equal(y_resampled, y[np.arange(3, 20, 2)])


def test_window_plot():
    windows = SlidingWindow(width=3, stride=2)
    X_windows = windows.fit_transform(signal_embedded_search)
    windows.plot(X_windows, sample=0)


@pytest.mark.parametrize('X', [signal, signal[:, None]])
@pytest.mark.parametrize('time_delay', list(range(1, 5)))
@pytest.mark.parametrize('dimension', list(range(1, 5)))
@pytest.mark.parametrize('stride', list(range(1, 5)))
def test_time_delay_embedding_consistent_with_takens_embedding(
        X, time_delay, dimension, stride
        ):
    """Test that time_delay_embedding and TakensEmbedding give identical
    outputs on 1D arrays."""
    n_points = (len(signal) - time_delay * (dimension - 1) - 1) // stride + 1
    embedder = TakensEmbedding(parameters_type='fixed', time_delay=time_delay,
                               dimension=dimension, stride=stride)
    func = partial(time_delay_embedding, time_delay=time_delay,
                   dimension=dimension, stride=stride)
    if n_points <= 0:
        with pytest.raises(ValueError):
            embedder.fit_transform(X)
        with pytest.raises(ValueError):
            func(X.ravel())
    else:
        assert np.array_equal(embedder.fit_transform(X), func(X.ravel()))


@pytest.mark.parametrize('params',
                         [{'time_delay': 0}, {'time_delay': -1},
                          {'dimension': 0}, {'dimension': -1},
                          {'stride': 0}, {'stride': -1}])
def test_time_delay_embedding_validation(params):
    with pytest.raises(ValueError):
        time_delay_embedding(signal, **params)


def test_time_delay_embedding_2D():
    """Testing the return values of time_delay_embedding on 2D input, assuming
    correctness of TakensEmbedding on 1D input."""
    params = {"time_delay": 2, "dimension": 3, "stride": 3}
    signals = np.tile(signal, (3, 1)) * np.arange(3)[:, None]
    signals_emb = time_delay_embedding(signals, **params)
    for i, X in enumerate(signals):
        embedder = TakensEmbedding(parameters_type='fixed', **params)
        assert np.array_equal(embedder.fit_transform(X), signals_emb[i])
