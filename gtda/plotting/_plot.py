
import plotly.graph_objects as gobj
from sklearn.utils.validation import check_array
import numpy as np


def _plot_image(image, samplings_x=None, samplings_y=None):
    """Low-level function to plot a 2d image.

    Parameters
    ----------
    image : ndarray of shape (n_pixels_x, n_pixels_y)
        A single image to plot.

    Returns
    -------
    fig : :class:`plotly.graph_objects.Figure`
        An object representing a plot of an image.

    """
    check_array(image, ensure_2d=True)

    if samplings_x is None:
        samplings_x = np.arange(image.shape[0])-0.5
    if samplings_y is None:
        samplings_y = np.arange(image.shape[1])-0.5
    layout = {
        "title": "Image",
        "plot_bgcolor": "white"
    }
    fig = gobj.Figure(layout=layout)

    fig.add_trace(gobj.Heatmap(z=image,
                               x=samplings_x,
                               y=samplings_y,
                               colorscale='blues'
                               ),
                  )
    return fig
