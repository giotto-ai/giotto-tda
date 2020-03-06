"""Point clouds-related plotting functions """
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj


def plot_point_cloud(point_cloud, dimension=None):
    """Plot the first 2 or 3 coordinates of the point cloud.

     This function will not work on 1-dimensional arrays.

    Parameters
    ----------
    point_cloud : ndarray of shape (n_samples, n_dimensions)
        Data points to be represented in a 2D or 3D scatter plot. Only the
        first 2 or 3 dimensions will be considered for plotting.

    dimension : int or None, default : ``None``
        This parameter sets the dimension of the resulting plot. If ``None``,
        the dimension will be chosen between 2 and 3 depending on
        ``n_dimensions`` see Input).

    """
    if dimension is None:
        dimension = np.min((3, point_cloud.shape[1]))

    # Check consistency between point_cloud and dimension
    if point_cloud.shape[1] < dimension:
        raise ValueError("Not enough dimensions available in the input point"
                         "cloud.")

    if dimension == 2:
        layout = {
            "title": "Point Cloud",
            "width": 800,
            "height": 800,
            "xaxis1": {
                "title": "First coordinate",
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
                "title": "Second coordinate",
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

        fig.add_trace(gobj.Scatter(x=point_cloud[:, 0],
                                   y=point_cloud[:, 1],
                                   mode='markers',
                                   marker=dict(size=4,
                                               color=list(range(
                                                   point_cloud.shape[0])),
                                               colorscale='Viridis',
                                               opacity=0.8)))
        fig.show()
    elif dimension == 3:

        scene = {
            "xaxis": {
                "title": "First coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
            },
            "yaxis": {
                "title": "Second coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
            },
            "zaxis": {
                "title": "Third coordinate",
                "type": "linear",
                "showexponent": "all",
                "exponentformat": "e"
            }
        }

        fig = gobj.Figure()
        fig.update_layout(scene=scene, title="Point cloud")

        fig.add_trace(gobj.Scatter3d(x=point_cloud[:, 0],
                                     y=point_cloud[:, 1],
                                     z=point_cloud[:, 2],
                                     mode='markers',
                                     marker=dict(size=4,
                                                 color=list(range(
                                                     point_cloud.shape[0])),
                                                 colorscale='Viridis',
                                                 opacity=0.8)))

        fig.show()
    else:
        raise ValueError("The value of the dimension is different from 2 or 3")
