"""Persistence diagrams-retated plotting functions """
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj


def plot_betti_curves(betti_curves, homology_dimensions=None, samplings=None):
    """Plot the Betti curves of a single persistence diagram by homology
    dimension.

    Parameters
    ----------
    betti_curves : ndarray of shape (n_homology_dimension, n_values)
        Collection of ``n_homology_dimension`` discretised Betti curves.
        Entry i along axis 0 should be the Betti curve in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti curves should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimension, n_values), \
        default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on
        the y-axis. If ``None``, the samplings will start at 0 with step 1.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, betti_curves.shape[0])
    if samplings is None:
        samplings = np.arange(0, betti_curves.shape[1])
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
        fig.add_trace(gobj.Scatter(x=samplings,
                                   y=betti_curves[i, :],
                                   mode='lines', showlegend=False,
                                   hoverinfo='none'))

    fig.show()


def plot_betti_surfaces(betti_curves, samplings=None,
                        homology_dimensions=None):
    """Plots the Betti surfaces (Betti numbers against time and filtration
    parameter) by homology dimension.

    Parameters
    ----------
    betti_curves : ndarray of shape (n_samples, n_homology_dimensions, \
        n_values)
        ``n_samples`` collections of discretised Betti curves. There are
        ``n_homology_dimension`` curves in each collection. Index i along axis
        1 should yield all Betti curves in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti surfaces should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimension, n_values), \
        default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on the
        y-axis. If ``None``, the samplings will start at 0 with step 1.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, betti_curves.shape[1])
    if samplings is None:
        samplings = np.arange(0, betti_curves.shape[2])

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
            fig.add_trace(gobj.Surface(x=samplings,
                                       y=np.arange(betti_curves.shape[0]),
                                       z=betti_curves[:, i, :],
                                       connectgaps=True, hoverinfo='none'))

            fig.show()


def plot_landscapes(landscapes, homology_dimensions=None, samplings=None):
    """Plot landscapes by homology dimension.

    Parameters
    ----------
    landscapes : ndarray of shape (n_homology_dimension, n_layers, n_values)
        Collection of ``n_homology_dimension`` discretised persistence
        landscapes. Each landscape contains ``n_layers`` layers. Entry i along
        axis 0 should be the persistence landscape in homology dimension i.

    homology_dimensions : list of int or None, default: ``None``
        Homology dimensions for which the Betti curves should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimension, n_layers, n_values), \
        default: ``None``
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `landscapes` on
        the y-axis. If ``None``, the samplings will start at 0 with step 1.

    """
    if homology_dimensions is None:
        homology_dimensions = np.arange(0, landscapes.shape[0])
    if samplings is None:
        samplings = np.arange(0, landscapes.shape[2])
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
            fig.add_trace(gobj.Scatter(x=samplings,
                                       y=landscapes[i, layer, :],
                                       mode='lines', showlegend=False,
                                       hoverinfo='none',
                                       name="layer {}".format(layer + 1)))

        fig.show()
