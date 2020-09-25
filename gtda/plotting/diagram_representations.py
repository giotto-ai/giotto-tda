"""Plotting functions for (vector) representations of persistence diagrams."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj


def plot_betti_curves(betti_numbers, samplings, homology_dimensions=None,
                      plotly_params=None):
    """Plot Betti curves by homology dimension.

    Parameters
    ----------
    betti_numbers : ndarray of shape (n_homology_dimensions, n_bins)
        Betti numbers, i.e. the y-coordinates of Betti curves. Entry i along
        axis 0 is assumed to contain the Betti numbers for a discretised Betti
        curve in homology dimension i.

    samplings : ndarray of shape (n_homology_dimensions, n_bins)
        Filtration parameter values to be used as the x-coordinates of the
        Betti curves.

    homology_dimensions : list, tuple or None, optional, default: ``None``
        Which homology dimensions to include in the plot. If ``None``,
        all available homology dimensions will be used.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the Betti curves.

    """
    if homology_dimensions is None:
        _homology_dimensions = list(range(betti_numbers.shape[0]))
    else:
        _homology_dimensions = homology_dimensions

    layout = {
        "xaxis1": {
            "title": "Filtration parameter",
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
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor="black",
                     mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor="black",
                     mirror=False)

    for dim in _homology_dimensions:
        fig.add_trace(gobj.Scatter(x=samplings[dim],
                                   y=betti_numbers[dim],
                                   mode="lines", showlegend=True,
                                   hoverinfo="none",
                                   name=f"H{int(dim)}"))

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig


def plot_betti_surfaces(betti_curves, samplings=None,
                        homology_dimensions=None, plotly_params=None):
    """Plot Betti surfaces (Betti numbers against "time" and filtration
    parameter) by homology dimension.

    Parameters
    ----------
    betti_curves : ndarray of shape (n_samples, n_homology_dimensions, \
        n_bins)
        Collection whose each entry contains the Betti numbers for
        ``n_homology_dimensions`` discretised Betti curves. Index i along axis
        1 is assumed to correspond to homology dimension i.

    samplings : ndarray of shape (n_homology_dimensions, n_bins)
        Filtration parameter values to be used as one of the independent
        variables when plotting the Betti surfaces. The other independent
        variable is "time", i.e. the sample index.

    homology_dimensions : list, tuple or None, optional, default: ``None``
        Homology dimensions for which the Betti surfaces should be plotted.
        If ``None``, all available dimensions will be used.

    samplings : ndarray of shape (n_homology_dimensions, n_bins)
        For each homology dimension, (filtration parameter) values to be used
        on the x-axis against the corresponding values in `betti_curves` on the
        y-axis.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    figs/fig : tuple of :class:`plotly.graph_objects.Figure`/\
        :class:`plotly.graph_objects.Figure` object
        If ``n_samples > 1``, a tuple of figures representing the Betti
        surfaces, with one figure per dimension in `homology_dimensions`.
        Otherwise, a single figure representing the Betti curve of the
        single sample present.

    """
    if homology_dimensions is None:
        _homology_dimensions = list(range(betti_curves.shape[1]))
    else:
        _homology_dimensions = homology_dimensions

    scene = {
        "xaxis": {
            "title": "Filtration parameter",
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
        return plot_betti_curves(
            betti_curves[0], samplings,
            homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )
    else:
        figs = []
        for dim in _homology_dimensions:
            fig = gobj.Figure()
            fig.update_layout(scene=scene,
                              title=f"Betti surface for homology "
                                    f"dimension {int(dim)}")
            fig.add_trace(gobj.Surface(x=samplings[dim],
                                       y=np.arange(betti_curves.shape[0]),
                                       z=betti_curves[:, dim],
                                       connectgaps=True, hoverinfo="none"))

            # Update traces and layout according to user input
            if plotly_params:
                fig.update_traces(plotly_params.get("traces", None))
                fig.update_layout(plotly_params.get("layout", None))

            figs.append(fig)

        return tuple(figs)
