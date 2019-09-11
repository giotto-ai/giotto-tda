"""Testing for TakensEmbedder"""

import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from giotto.time_series import TakensEmbedder

signal = np.asarray([np.sin(x / 2) + 2 for x in range(0, 20)])

signal_embedded_search = np.array([[[2., 2.47942554],
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
                                    [2.41211849, 1.92484888]]])

signal_embedded_fixed = \
    np.array([[[2., 2.47942554, 2.84147098, 2.99749499, 2.90929743],
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
               [2.93799998, 2.98935825, 2.79848711, 2.41211849, 1.92484888]]])


def test_embedder_params():
    embedding_parameters_type = 'not_defined'
    embedder = TakensEmbedder(
        embedding_parameters_type=embedding_parameters_type)
    msg = 'The embedding parameters type %s is not supported'
    with pytest.raises(ValueError, match=msg % embedding_parameters_type):
        embedder.fit(signal)

    embedder = TakensEmbedder(outer_window_duration=signal.shape[0] + 1)
    msg = 'Not enough data to have a single outer window.'

    with pytest.raises(ValueError, match=msg):
        embedder.fit(signal)


def test_embedder_not_fitted():
    embedder = TakensEmbedder()

    with pytest.raises(NotFittedError):
        embedder.transform(signal)


@pytest.mark.parametrize("embedding_parameters_type, expected",
                         [('search', signal_embedded_search),
                          ('fixed', signal_embedded_fixed)])
def test_embedder_transform(embedding_parameters_type, expected):
    embedder = TakensEmbedder(
        embedding_parameters_type=embedding_parameters_type)

    assert_almost_equal(embedder.fit_transform(signal), expected)
