"""Point-cloud–related plotting functions and classes."""
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj

from ..utils.validation import validate_params


def plot_point_cloud(point_cloud, dimension=None, plotly_params=None):
    """Plot the first 2 or 3 coordinates of a point cloud.

    Note: this function does not work on 1D arrays.

    Parameters
    ----------
    point_cloud : ndarray of shape (n_samples, n_dimensions)
        Data points to be represented in a 2D or 3D scatter plot. Only the
        first 2 or 3 dimensions will be considered for plotting.

    dimension : int or None, default: ``None``
        Sets the dimension of the resulting plot. If ``None``, the dimension
        will be chosen between 2 and 3 depending on the shape of `point_cloud`.

    plotly_params : dict or None, optional, default: ``None``
        Custom parameters to configure the plotly figure. Allowed keys are
        ``"trace"`` and ``"layout"``, and the corresponding values should be
        dictionaries containing keyword arguments as would be fed to the
        :meth:`update_traces` and :meth:`update_layout` methods of
        :class:`plotly.graph_objects.Figure`.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure` object
        Figure representing a point cloud in 2D or 3D.

    """
    # TODO: increase the marker size
    validate_params({"dimension": dimension},
                    {"dimension": {"type": (int, type(None)), "in": [2, 3]}})
    if dimension is None:
        dimension = np.min((3, point_cloud.shape[1]))

    # Check consistency between point_cloud and dimension
    if point_cloud.shape[1] < dimension:
        raise ValueError("Not enough dimensions available in the input point "
                         "cloud.")

    elif dimension == 2:
        layout = {
            "width": 600,
            "height": 600,
            "xaxis1": {
                "title": "0th",
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
                "title": "1st",
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

        fig.add_trace(gobj.Scatter(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            mode="markers",
            marker={"size": 4,
                    "color": list(range(point_cloud.shape[0])),
                    "colorscale": "Viridis",
                    "opacity": 0.8}
            ))

    elif dimension == 3:
        scene = {
            "xaxis": {
                "title": "0th",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
                },
            "yaxis": {
                "title": "1st",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
                },
            "zaxis": {
                "title": "2nd",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
                }
            }

        fig = gobj.Figure()
        fig.update_layout(scene=scene)

        fig.add_trace(gobj.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode="markers",
            marker={"size": 4,
                    "color": list(range(point_cloud.shape[0])),
                    "colorscale": "Viridis",
                    "opacity": 0.8}
            ))

    # Update trace and layout according to user input
    if plotly_params:
        fig.update_traces(plotly_params.get("trace", None))
        fig.update_layout(plotly_params.get("layout", None))

    return fig
