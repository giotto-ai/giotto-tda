"""Persistent-homology–related plotting functions and classes."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from plotly.colors import DEFAULT_PLOTLY_COLORS as default_colors


def plot_diagram(diagram, homology_dimensions=None, plotly_params=None):
    """Plot a single persistence diagram.

    Parameters
    ----------
    diagram : ndarray of shape (n_points, 3)
        The persistence diagram to plot, where the third dimension along axis 1
        contains homology dimensions, and the first two contain (birth, death)
        pairs to be used as coordinates in the two-dimensional plot.

    homology_dimensions : list of int or None, optional, default: ``None``
        Homology dimensions which will appear on the plot. If ``None``, all
        homology dimensions which appear in `diagram` will be plotted.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the persistence diagram.

    """
    # TODO: increase the marker size
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    diag = diagram[diagram[:, 0] != diagram[:, 1]]
    diag_no_dims = diag[:, :2]
    posinfinite_mask = np.isposinf(diag_no_dims)
    neginfinite_mask = np.isneginf(diag_no_dims)
    max_val = np.max(np.where(posinfinite_mask, -np.inf, diag_no_dims))
    min_val = np.min(np.where(neginfinite_mask, np.inf, diag_no_dims))
    parameter_range = max_val - min_val
    extra_space_factor = 0.02
    has_posinfinite_death = np.any(posinfinite_mask[:, 1])
    if has_posinfinite_death:
        posinfinity_val = max_val + 0.1 * parameter_range
        extra_space_factor += 0.1
    extra_space = extra_space_factor * parameter_range
    min_val_display = min_val - extra_space
    max_val_display = max_val + extra_space

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[min_val_display, max_val_display],
        y=[min_val_display, max_val_display],
        mode="lines",
        line={"dash": "dash", "width": 1, "color": "black"},
        showlegend=False,
        hoverinfo="none"
        ))

    for dim in homology_dimensions:
        name = f"H{int(dim)}" if dim != np.inf else "Any homology dimension"
        subdiag = diag[diag[:, 2] == dim]
        unique, inverse, counts = np.unique(
            subdiag, axis=0, return_inverse=True, return_counts=True
            )
        hovertext = [
            f"{tuple(unique[unique_row_index][:2])}" +
            (
                f", multiplicity: {counts[unique_row_index]}"
                if counts[unique_row_index] > 1 else ""
            )
            for unique_row_index in inverse
            ]
        y = subdiag[:, 1]
        if has_posinfinite_death:
            y[np.isposinf(y)] = posinfinity_val
        fig.add_trace(go.Scatter(
            x=subdiag[:, 0], y=y, mode="markers",
            hoverinfo="text", hovertext=hovertext, name=name
        ))

    fig.update_layout(
        width=500,
        height=500,
        xaxis1={
            "title": "Birth",
            "side": "bottom",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            },
        yaxis1={
            "title": "Death",
            "side": "left",
            "type": "linear",
            "range": [min_val_display, max_val_display],
            "autorange": False, "scaleanchor": "x", "scaleratio": 1,
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            },
        plot_bgcolor="white"
        )

    # Add a horizontal dashed line for points with infinite death
    if has_posinfinite_death:
        fig.add_trace(go.Scatter(
            x=[min_val_display, max_val_display],
            y=[posinfinity_val, posinfinity_val],
            mode="lines",
            line={"dash": "dash", "width": 0.5, "color": "black"},
            showlegend=True,
            name=u"\u221E",
            hoverinfo="none"
        ))

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig


def plot_extended_diagram(diagram, homology_dimensions=None,
                          plotly_params=None):
    """Plot a single extended persistence diagram.

    Parameters
    ----------
    diagram : ndarray of shape (n_points, 4)
        The persistence diagram to plot, where the third dimension along axis 1
        contains homology dimensions, the fourth is either 1 or -1 depending on
        whether the feature was born and died during the same phase (upwards or
        downwards sweep) and the first two contain (birth, death) pairs to be
        used as coordinates in the two-dimensional plot.

    homology_dimensions : list of int or None, optional, default: ``None``
        Homology dimensions which will appear on the plot. If ``None``, all
        homology dimensions which appear in `diagram` will be plotted.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"traces"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing the persistence diagram.

    """
    # TODO: increase the marker size
    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    nontrivial_mask = np.logical_not(
        np.logical_and(diagram[:, 0] == diagram[:, 1], diagram[:, 3] == 1.)
        )
    diag = diagram[nontrivial_mask]
    diag_no_dims = diag[:, :2]
    posinfinite_mask = np.isposinf(diag_no_dims)
    neginfinite_mask = np.isneginf(diag_no_dims)
    max_val = np.max(np.where(posinfinite_mask, -np.inf, diag_no_dims))
    min_val = np.min(np.where(neginfinite_mask, np.inf, diag_no_dims))
    parameter_range = max_val - min_val
    extra_space_factor = 0.02
    extra_space = extra_space_factor * parameter_range
    min_val_display = min_val - extra_space
    max_val_display = max_val + extra_space

    subplot_titles = ["Same sweep", "Different sweep"]
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=subplot_titles)

    for row in [1, 2]:
        fig.add_trace(go.Scatter(
            x=[min_val_display, max_val_display],
            y=[min_val_display, max_val_display],
            mode="lines",
            line={"dash": "dash", "width": 1, "color": "black"},
            showlegend=False,
            hoverinfo="none"
            ), row=row, col=1)

    for i, sweep in enumerate([1., -1.]):
        diag_sweep = diag[diag[:, 3] == sweep]
        for j, dim in enumerate(homology_dimensions):
            name = f"H{int(dim)}" \
                if dim != np.inf else "Any homology dimension"
            subdiag = diag_sweep[diag_sweep[:, 2] == dim]
            unique, inverse, counts = np.unique(
                subdiag, axis=0, return_inverse=True, return_counts=True
                )
            hovertext = [
                f"{tuple(unique[unique_row_index][:2])}" +
                (
                    f", multiplicity: {counts[unique_row_index]}"
                    if counts[unique_row_index] > 1 else ""
                )
                for unique_row_index in inverse
                ]
            fig.add_trace(go.Scatter(
                x=subdiag[:, 0], y=subdiag[:, 1], mode="markers",
                marker={"color": default_colors[j]}, hoverinfo="text",
                hovertext=hovertext, name=name
                ), row=i + 1, col=1)

    fig.update_layout(width=500,
                      height=1000,
                      plot_bgcolor="white")
    fig.update_xaxes(title="Birth",
                     side="bottom",
                     type="linear",
                     range=[min_val_display, max_val_display],
                     autorange=False,
                     ticks="outside",
                     showline=True,
                     zeroline=True,
                     linewidth=1,
                     linecolor="black",
                     mirror=False,
                     showexponent="all",
                     exponentformat="e")
    fig.update_yaxes(title="Death",
                     side="left",
                     type="linear",
                     range=[min_val_display, max_val_display],
                     autorange=False,
                     scaleanchor="x",
                     scaleratio=1,
                     ticks="outside",
                     showline=True,
                     zeroline=True,
                     linewidth=1,
                     linecolor="black",
                     mirror=False,
                     showexponent="all",
                     exponentformat="e")

    # Update traces and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig
