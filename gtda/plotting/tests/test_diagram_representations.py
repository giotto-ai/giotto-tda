"""Testing for plot_betti_curves and plot_betti_surfaces."""
# License: GNU AGPLv3

import numpy as np
import pytest

from gtda.plotting import plot_betti_curves, plot_betti_surfaces

n_samples = 10
n_homology_dimensions = 3
n_bins = 20

X = np.random.randint(0, 20, n_samples * n_homology_dimensions * n_bins).\
    reshape(n_samples, n_homology_dimensions, n_bins)
samplings = np.vstack([
    np.linspace(0, 10, num=n_bins),
    np.linspace(5, 15, num=n_bins),
    np.linspace(10, 20, num=n_bins)
    ])
plotly_params_curves = {"layout": {"xaxis1": {"title": "New title"}}}
plotly_params_surfaces = {
    "layout": {"scene": {"xaxis": {"title": "New title"}}}
    }


@pytest.mark.parametrize("homology_dimensions", [None, [0], [0, 1], [0, 1, 2]])
def test_plot_betti_curves(homology_dimensions):
    fig = plot_betti_curves(X[0], samplings=samplings,
                            homology_dimensions=homology_dimensions,
                            plotly_params=plotly_params_curves)

    if homology_dimensions is None:
        _homology_dimensions = list(range(X.shape[1]))
    else:
        _homology_dimensions = homology_dimensions
    traces_xy = all([
        np.array_equal(fig.data[i].x, samplings[i])
        and np.array_equal(fig.data[i].y, X[0][i])
        for i in _homology_dimensions
        ])
    assert traces_xy

    assert fig.layout.xaxis1.title.text == "New title"


@pytest.mark.parametrize("homology_dimensions", [None, [0], [0, 1], [0, 1, 2]])
def test_plot_betti_surfaces(homology_dimensions):
    fig = plot_betti_surfaces(X, samplings=samplings,
                              homology_dimensions=homology_dimensions,
                              plotly_params=plotly_params_surfaces)

    if homology_dimensions is None:
        _homology_dimensions = list(range(X.shape[1]))
    else:
        _homology_dimensions = homology_dimensions
    traces_xyz = all([
        np.array_equal(fig[i].data[0].x, samplings[i])
        and np.array_equal(fig[i].data[0].y, np.arange(X.shape[0]))
        and np.array_equal(fig[i].data[0].z, X[:, i])
        for i in _homology_dimensions
        ])
    assert traces_xyz

    assert [fig[i].layout.scene.xaxis.title.text == "New title"
            for i in _homology_dimensions]


def test_plot_betti_surfaces_reduces_to_curves():
    fig = plot_betti_surfaces(X[[0]], samplings=samplings,
                              plotly_params=plotly_params_curves)

    _homology_dimensions = range(X.shape[1])
    traces_xy = all([
        np.array_equal(fig.data[i].x, samplings[i])
        and np.array_equal(fig.data[i].y, X[0][i])
        for i in _homology_dimensions
        ])
    assert traces_xy
