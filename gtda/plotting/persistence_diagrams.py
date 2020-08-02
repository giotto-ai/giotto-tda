"""Persistent-homology–related plotting functions and classes."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj


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

    diagram_no_dims = diagram[:, :2]
    max_birth, max_death = np.where(
        np.isposinf(diagram_no_dims), -np.inf, diagram_no_dims
    ).max(axis=0)
    min_birth, min_death = np.where(
        np.isneginf(diagram_no_dims), np.inf, diagram_no_dims
    ).min(axis=0)

    fig = gobj.Figure()
    fig.add_trace(gobj.Scatter(x=[100 * min(-np.abs(max_death), min_birth),
                                  100 * max_death],
                               y=[100 * min(-np.abs(max_death), min_birth),
                                  100 * max_death],
                               mode='lines',
                               line=dict(dash='dash', width=1, color='black'),
                               showlegend=False, hoverinfo='none'))

    for dim in homology_dimensions:
        name = f'H{int(dim)}' if dim != np.inf else 'Any homology dimension'
        subdiagram = diagram[diagram[:, 2] == dim]
        diff = (subdiagram[:, 1] != subdiagram[:, 0])
        subdiagram = subdiagram[diff]
        fig.add_trace(gobj.Scatter(x=subdiagram[:, 0], y=subdiagram[:, 1],
                                   mode='markers', name=name))

    range = max_death - min_birth
    extra_space = 0.02 * range

    fig.update_layout(
        width=500,
        height=500,
        xaxis1=dict(
            title='Birth',
            side='bottom',
            type='linear',
            range=[min_birth - extra_space, max_death + extra_space],
            autorange=False,
            ticks='outside',
            showline=True,
            zeroline=True,
            linewidth=1,
            linecolor='black',
            mirror=False,
            showexponent='all',
            exponentformat='e'
        ),
        yaxis1=dict(
            title='Death',
            side='left',
            type='linear',
            range=[min_birth - extra_space, max_death + extra_space],
            autorange=False,
            scaleanchor="x",
            scaleratio=1,
            ticks='outside',
            showline=True,
            zeroline=True,
            linewidth=1,
            linecolor='black',
            mirror=False,
            showexponent='all',
            exponentformat='e'
        ),
        plot_bgcolor='white'
    )

    # Update trace and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("traces", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig
