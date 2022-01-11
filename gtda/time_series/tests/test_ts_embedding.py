"""Testing for time series embedding."""
# License: GNU AGPLv3

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.time_series import SlidingWindow, \
    takens_embedding_optimal_parameters, SingleTakensEmbedding, TakensEmbedding

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


def test_takens_embedding_optimal_parameters_validate():
    time_delay = -1
    dimension = 2
    with pytest.raises(ValueError):
        takens_embedding_optimal_parameters(signal, time_delay, dimension,
                                            validate=True)


def test_embedder_params():
    parameters_type = 'not_defined'
    embedder = SingleTakensEmbedding(parameters_type=parameters_type)
    with pytest.raises(ValueError):
        embedder.fit(signal)


def test_embedder_not_fitted():
    embedder = SingleTakensEmbedding()
    with pytest.raises(NotFittedError):
        embedder.transform(signal)


@pytest.mark.parametrize("parameters_type, expected",
                         [('search', signal_embedded_search),
                          ('fixed', signal_embedded_fixed)])
def test_embedder_transform(parameters_type, expected):
    embedder = SingleTakensEmbedding(parameters_type=parameters_type)

    assert_almost_equal(embedder.fit_transform(signal), expected)


def test_embedder_resample():
    embedder = SingleTakensEmbedding(parameters_type='fixed', time_delay=3,
                                     dimension=2, stride=3)
    embedder.fit(signal)
    y_resampled = embedder.resample(y)
    assert_almost_equal(y_resampled, y[np.arange(4, 20, 3)])


@pytest.mark.parametrize("size", [0, -1])
def test_window_params(size):
    windows = SlidingWindow(size=size)
    with pytest.raises(ValueError):
        windows.fit(signal)


def test_window_transform():
    windows = SlidingWindow(size=4, stride=2)
    X_windows = windows.fit_transform(signal_embedded_search)
    assert (X_windows.shape == (8, 4, 2))


def test_window_resample():
    windows = SlidingWindow(size=4, stride=2)
    windows.fit(y)
    y_resampled = windows.resample(y)
    assert_almost_equal(y_resampled, y[np.arange(3, 20, 2)])


def test_window_slice_windows():
    windows = SlidingWindow(size=4, stride=2)
    X = signal_embedded_search
    X_windows = windows.fit_transform(X)
    slice_idx = windows.slice_windows(X)
    assert_almost_equal(
        np.stack([X[begin:end] for begin, end in slice_idx]), X_windows
        )


@pytest.mark.parametrize('time_delay', list(range(1, 5)))
@pytest.mark.parametrize('dimension', list(range(1, 5)))
@pytest.mark.parametrize('stride', list(range(1, 5)))
def test_takens_embedding_consistent_with_single_takens_embedding(
        time_delay, dimension, stride
        ):
    """Test that TakensEmbedding and SingleTakensEmbedding give identical
    outputs (up to shape) on arrays representing single univariate time
    series."""
    n_points = (len(signal) - time_delay * (dimension - 1) - 1) // stride + 1
    single_embedder = SingleTakensEmbedding(parameters_type='fixed',
                                            time_delay=time_delay,
                                            dimension=dimension, stride=stride)
    embedder = TakensEmbedding(time_delay=time_delay, dimension=dimension,
                               stride=stride)
    if n_points <= 0:
        with pytest.raises(ValueError):
            single_embedder.fit_transform(signal)
        with pytest.raises(ValueError):
            embedder.fit_transform(signal[None, :])
    else:
        single_embedder_res = single_embedder.fit_transform(signal)
        embedder_res = embedder.fit_transform(signal[None, :])[0]
        assert np.array_equal(single_embedder_res, embedder_res)


@pytest.mark.parametrize("params",
                         [{"time_delay": 0}, {"time_delay": -1},
                          {"dimension": 0}, {"dimension": -1},
                          {"stride": 0}, {"stride": -1},
                          {"flatten": "foo"}, {"ensure_last_value": "bar"}])
def test_takens_embedding_validation(params):
    if "flatten" not in params and "ensure_last_value" not in params:
        exception_type = ValueError
    else:
        exception_type = TypeError
    with pytest.raises(exception_type):
        TE = TakensEmbedding(**params)
        TE.fit(signal[None, :])


def test_takens_embedding_2D():
    """Test the return values of TakensEmbedding on 2D input or list of 1D
     input, with default parameters."""
    signals = np.arange(10).reshape(2, 5)
    TE = TakensEmbedding()
    signals_emb = TE.fit_transform(signals)
    signals_emb_list = TE.fit_transform(list(signals))
    signals_emb_exp = np.array([[[0, 1],
                                 [1, 2],
                                 [2, 3],
                                 [3, 4]],
                                [[5, 6],
                                 [6, 7],
                                 [7, 8],
                                 [8, 9]]])
    assert np.array_equal(signals_emb, signals_emb_exp)
    assert np.array_equal(np.asarray(signals_emb_list), signals_emb_exp)


def test_takens_embedding_3D_default():
    """Test the return values of TakensEmbedding on 3D input or list of 2D
    input, with default parameters."""
    signals = np.arange(20).reshape(2, 2, 5)
    TE = TakensEmbedding()
    signals_emb = TE.fit_transform(signals)
    signals_emb_list = TE.fit_transform(list(signals))
    signals_emb_exp = np.array([[[0, 1, 5, 6],
                                 [1, 2, 6, 7],
                                 [2, 3, 7, 8],
                                 [3, 4, 8, 9]],
                                [[10, 11, 15, 16],
                                 [11, 12, 16, 17],
                                 [12, 13, 17, 18],
                                 [13, 14, 18, 19]]])
    assert np.array_equal(signals_emb, signals_emb_exp)
    assert np.array_equal(np.asarray(signals_emb_list), signals_emb_exp)


def test_takens_embedding_3D_no_flatten():
    """Test the return values of TakensEmbedding on 3D input or list of 2D
    input, with `flatten` set to ``False``."""
    signals = np.arange(20).reshape(2, 2, 5)
    TE = TakensEmbedding(flatten=False)
    signals_emb = TE.fit_transform(signals)
    signals_emb_list = TE.fit_transform(list(signals))
    signals_emb_exp = np.array([[[[0, 1],
                                  [1, 2],
                                  [2, 3],
                                  [3, 4]],
                                 [[5, 6],
                                  [6, 7],
                                  [7, 8],
                                  [8, 9]]],
                                [[[10, 11],
                                  [11, 12],
                                  [12, 13],
                                  [13, 14]],
                                 [[15, 16],
                                  [16, 17],
                                  [17, 18],
                                  [18, 19]]]])
    assert np.array_equal(signals_emb, signals_emb_exp)
    assert np.array_equal(np.asarray(signals_emb_list), signals_emb_exp)


def test_takens_embedding_plot():
    trace_params = {"mode": "lines+markers"}
    layout_params = {"title": "New title"}
    TE = TakensEmbedding()
    plotly_params = {"trace": trace_params, "layout": layout_params}
    TE.fit_transform_plot([np.arange(20)], sample=0,
                          plotly_params=plotly_params)
