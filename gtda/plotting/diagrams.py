"""Plotting functions for featurizations or representations of persistence
diagrams, coming in particular from the outputs of some classes in
:mod:`gtda.diagrams`."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj
from ._plot import _plot_image


def plot_heat_kernel(heat_kernel, samplings, homology_dimension=0):
    """Plot the heat kernel of a single persistence diagram in a single dimension.

    Parameters
    ----------
    heat_kernel : ndarray of shape (homology_dimensions, n_bins, \
        n_bins)
        Raster image: one image per homology dimension. Index i along axis 0
        corresponds to the i-th homology dimension in
        :attr:`homology_dimensions_`.

    homology_dimension : int or None, default: ``0``
        Homology dimension for which the heat kernel should be plotted.

    samplings : ndarray of shape (n_homology_dimensions, n_bins), \
        default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x- and y-axes.

    """
    if homology_dimension is None:
        homology_dimension = 0
    samplings_x = samplings[homology_dimension]
    fig = _plot_image(heat_kernel[homology_dimension, ::-1, :],
                      samplings_x, samplings_x)
    fig.update_layout({'title': 'Heat Kernel'})
    fig.show()


def plot_silhouettes(silhouettes, samplings, homology_dimensions=None):
    """Plot the silhouettes of a single persistence diagram by homology
    dimension.

    Parameters
    ----------
    silhouettes : ndarray of shape (n_homology_dimensions, n_bins)
        Collection of ``n_homology_dimension`` discretised Silhouettes.
        Entry i along axis 0 should be the silhouette in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the silhouettes should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimensions, n_bins)
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `silhouettes` on
        the y-axis.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, silhouettes.shape[0])
    layout = {
        "title": "Silhouette",
        "xaxis1": {
            "title": "Epsilon",
            "side": "bottom",
            "type": "linear",
            "ticks": "outside",
            "anchor": "x1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "yaxis1": {
            "title": "Silhouette",
            "side": "left",
            "type": "linear",
            "ticks": "outside",
            "anchor": "y1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "plot_bgcolor": "white"
    }
    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)

    for i, dimension in enumerate(homology_dimensions):
        fig.add_trace(gobj.Scatter(x=samplings[dimension],
                                   y=silhouettes[i, :],
                                   mode='lines', showlegend=True,
                                   hoverinfo='none',
                                   name=f'dim{dimension}'))

    fig.show()


def plot_betti_curves(betti_curves, samplings, homology_dimensions=None):
    """Plot the Betti curves of a single persistence diagram by homology
    dimension.

    Parameters
    ----------
    betti_curves : ndarray of shape (n_homology_dimensions, n_bins)
        Collection of ``n_homology_dimension`` discretised Betti curves.
        Entry i along axis 0 should be the Betti curve in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti curves should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimensions, n_bins),
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on
        the y-axis.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, betti_curves.shape[0])
    layout = {
        "title": "Betti curves",
        "xaxis1": {
            "title": "Epsilon",
            "side": "bottom",
            "type": "linear",
            "ticks": "outside",
            "anchor": "x1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "yaxis1": {
            "title": "Betti number",
            "side": "left",
            "type": "linear",
            "ticks": "outside",
            "anchor": "y1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "plot_bgcolor": "white"
    }
    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)

    for i, dimension in enumerate(homology_dimensions):
        fig.add_trace(gobj.Scatter(x=samplings[dimension],
                                   y=betti_curves[i, :],
                                   mode='lines', showlegend=True,
                                   hoverinfo='none',
                                   name=f'dim{dimension}'))

    fig.show()


def plot_betti_surfaces(betti_curves, samplings=None,
                        homology_dimensions=None):
    """Plots the Betti surfaces (Betti numbers against time and filtration
    parameter) by homology dimension.

    Parameters
    ----------
    betti_curves : ndarray of shape (n_samples, n_homology_dimensions, \
        n_bins)
        ``n_samples`` collections of discretised Betti curves. There are
        ``n_homology_dimension`` curves in each collection. Index i along axis
        1 should yield all Betti curves in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti surfaces should be plotted.
        If ``None``, all available dimensions will be used. If int, betti
        curves are plotted instead.

    samplings : ndarray of shape (n_homology_dimensions, n_bins),
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on the
        y-axis.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, betti_curves.shape[1])

    scene = {
        "xaxis": {
            "title": "Epsilon",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e"
        },
        "yaxis": {
            "title": "Time",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e"
        },
        "zaxis": {
            "title": "Betti number",
            "type": "linear",
            "showexponent": "all",
            "exponentformat": "e"
        }
    }
    if betti_curves.shape[0] == 1:
        plot_betti_curves(betti_curves[0], samplings, homology_dimensions)
    else:
        for i, dimension in enumerate(homology_dimensions):
            fig = gobj.Figure()
            fig.update_layout(scene=scene,
                              title="Betti surface for homology "
                              "dimension {}".format(int(dimension)))
            fig.add_trace(gobj.Surface(x=samplings[i],
                                       y=np.arange(betti_curves.shape[0]),
                                       z=betti_curves[:, i, :],
                                       connectgaps=True, hoverinfo='none'))

            fig.show()


def plot_landscapes(landscapes, samplings, homology_dimensions=None):
    """Plot landscapes by homology dimension.

    Parameters
    ----------
    landscapes : ndarray of shape (n_homology_dimensions, n_layers, n_bins)
        Collection of ``n_homology_dimension`` discretised persistence
        landscapes. Each landscape contains ``n_layers`` layers. Entry i along
        axis 0 should be the persistence landscape in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti curves should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimensions, n_layers, n_bins), \
        default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `landscapes` on
        the y-axis.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, landscapes.shape[0])
    layout = {
        "xaxis1": {
            "side": "bottom",
            "type": "linear",
            "ticks": "outside",
            "anchor": "y1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "yaxis1": {
            "side": "left",
            "type": "linear",
            "ticks": "outside",
            "anchor": "x1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "plot_bgcolor": "white"
    }

    for i, dimension in enumerate(homology_dimensions):
        layout_dim = layout.copy()
        layout_dim['title'] = "Persistence landscape for homology " + \
                              "dimension {}".format(int(dimension))
        fig = gobj.Figure(layout=layout_dim)
        fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                         mirror=False)
        fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                         mirror=False)

        n_layers = landscapes.shape[1]
        for layer in range(n_layers):
            fig.add_trace(gobj.Scatter(x=samplings[dimension],
                                       y=landscapes[i, layer, :],
                                       mode='lines', showlegend=True,
                                       hoverinfo='none',
                                       name=f"layer {layer + 1}"))

        fig.show()


def plot_persistence_image(images, samplings, homology_dimension=None):
    """Plot persistence images by homology dimension.

    Parameters
    ----------
    images : ndarray of shape (n_homology_dimensions, n_bins, n_bins)
        Collection of ``n_homology_dimension`` discretised persistence
        images. Each image is an an array of shape ``(n_bins, n_bins)``.
        Entry i along axis 0 should be the persistence image in homology
        dimension i.

    samplings : ndarray of shape (n_homology_dimensions, 2, n_bins),
        For each homology dimension, (filtration parameter and persistence)
        values to be used on the x-axis against the corresponding values in
        `images` on the y-axis.

    homology_dimension : int or None, default: ``None``
        Homology dimension for which the persistence image should be plotted.
        If ``None``, the first available dimension will be used.

    """
    if homology_dimension is None:
        homology_dimension = 0
    samplings_x, samplings_y = samplings[homology_dimension]
    fig = _plot_image(images[homology_dimension], samplings_x, samplings_y)
    fig.update_layout({'title': 'Persistence Image'})
    fig.show()
