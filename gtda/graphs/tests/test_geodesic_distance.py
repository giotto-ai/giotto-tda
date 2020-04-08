"""Testing for GraphGeodesicDistance."""

import numpy as np
import plotly.io as pio
import pytest
from numpy.testing import assert_almost_equal
from sklearn.exceptions import NotFittedError

from gtda.graphs import GraphGeodesicDistance

pio.renderers.default = 'plotly_mimetype'

X_ggd = np.array([
    np.array(
        [[0, 1, 3, 0, 0],
         [1, 0, 5, 0, 0],
         [3, 5, 0, 4, 0],
         [0, 0, 4, 0, 0],
         [0, 0, 0, 0, 0]]),
    np.array(
        [[0, 1, 3, 0, 0],
         [1, 0, 1, 0, 0],
         [3, 1, 0, 4, 0],
         [0, 0, 4, 0, 0],
         [0, 0, 0, 0, 0]])])


def test_ggd_not_fitted():
    ggd = GraphGeodesicDistance()

    with pytest.raises(NotFittedError):
        ggd.transform(X_ggd)


def test_ggd_fit_transform_plot():
    GraphGeodesicDistance().fit_transform_plot(X_ggd, sample=0)


def test_ggd_transform():
    X_ggd_res = np.array([
        [[0., 1., 3., 7., np.inf],
         [1., 0., 4., 8., np.inf],
         [3., 4., 0., 4., np.inf],
         [7., 8., 4., 0., np.inf],
         [np.inf, np.inf, np.inf, np.inf, 0.]],

        [[0., 1., 2., 6., np.inf],
         [1., 0., 1., 5., np.inf],
         [2., 1., 0., 4., np.inf],
         [6., 5., 4., 0., np.inf],
         [np.inf, np.inf, np.inf, np.inf, 0.]]
    ])
    ggd = GraphGeodesicDistance()

    assert_almost_equal(ggd.fit_transform(X_ggd), X_ggd_res)


def test_parallel_ggd_transform():
    ggd = GraphGeodesicDistance(n_jobs=1)
    ggd_parallel = GraphGeodesicDistance(n_jobs=2)

    assert_almost_equal(ggd.fit_transform(X_ggd), ggd_parallel.fit_transform(
        X_ggd))
