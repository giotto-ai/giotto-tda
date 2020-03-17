"""Persistent-homology–related plotting functions and classes."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj


def plot_diagram(diagram, homology_dimensions=None, **input_layout):
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

    """
    # TODO: increase the marker size
    from ..diagrams._utils import _subdiagrams

    if homology_dimensions is None:
        homology_dimensions = np.unique(diagram[:, 2])

    max_filt_param = np.where(np.isinf(diagram), -np.inf, diagram).max()

    layout = {
        "title": "Persistence diagram",
        "width": 500,
        "height": 500,
        "xaxis1": {
            "title": "Birth",
            "side": "bottom",
            "type": "linear",
            "range": [0, 1.1 * max_filt_param],
            "ticks": "outside",
            "anchor": "y1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "yaxis1": {
            "title": "Death",
            "side": "left",
            "type": "linear",
            "range": [0, 1.1 * max_filt_param],
            "ticks": "outside",
            "anchor": "x1",
            "showline": True,
            "zeroline": True,
            "showexponent": "all",
            "exponentformat": "e"
        },
        "plot_bgcolor": "white"
    }

    layout.update(input_layout)

    fig = gobj.Figure(layout=layout)
    fig.update_xaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)
    fig.update_yaxes(zeroline=True, linewidth=1, linecolor='black',
                     mirror=False)

    fig.add_trace(gobj.Scatter(x=np.array([-100 * max_filt_param,
                                           100 * max_filt_param]),
                               y=np.array([-100 * max_filt_param,
                                           100 * max_filt_param]),
                               mode='lines',
                               line=dict(dash='dash', width=1, color='black'),
                               showlegend=False, hoverinfo='none'))

    for i, dimension in enumerate(homology_dimensions):
        name = "H{}".format(int(dimension))
        subdiagram = _subdiagrams(np.asarray([diagram]), [dimension],
                                  remove_dim=True)[0]
        diff = (subdiagram[:, 1] != subdiagram[:, 0])
        subdiagram = subdiagram[diff]
        fig.add_trace(gobj.Scatter(x=subdiagram[:, 0], y=subdiagram[:, 1],
                                   mode='markers', name=name))

    fig.show()
