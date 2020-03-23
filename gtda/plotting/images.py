"""Image-related plotting functions and classes."""
# License: GNU AGPLv3

import plotly.graph_objects as gobj


def plot_heatmap(data, x=None, y=None, colorscale='greys', origin='upper',
                 title=None):
    """Plot a 2D single-channel image, as a heat map from 2D array data.

    Parameters
    ----------
    data : ndarray of shape (n_pixels_x, n_pixels_y)
        Data describing the heat map value-to-color mapping.

    x : ndarray of shape (n_pixels_x,) or None, optional, default: ``None``
        Horizontal coordinates of the pixels in `data`.

    y : ndarray of shape (n_pixels_y,) or None, optional, default: ``None``
        Vertical coordinates of the pixels in `data`.

    colorscale : str, optional, default: ``'greys'``
        Color scale to be used in the heat map. Can be anything allowed by
        :class:`plotly.graph_objects.Heatmap`.

    origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
        Position of the [0, 0] pixel of `data`, in the upper left or lower
        left corner. The convention ``'upper'`` is typically used for
        matrices and images.

    title : str or None, optional, default: ``None``
        Title of the resulting figure.

    """
    autorange = True if origin == 'lower' else 'reversed'
    layout = dict(
        xaxis=dict(scaleanchor='y', constrain='domain'),
        yaxis=dict(autorange=autorange, constrain='domain'),
        plot_bgcolor='white',
        title=title
    )
    fig = gobj.Figure(layout=layout)
    fig.add_trace(gobj.Heatmap(
        z=data, x=x, y=y, colorscale=colorscale
    ))

    fig.show()
