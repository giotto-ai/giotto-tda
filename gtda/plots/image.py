"""Image-retated plotting functions """
# License: GNU AGPLv3

import numpy as np
import plotly.graph_objs as gobj
from sklearn.utils.validation import check_array


def plot_image(image):
    """Plot a 2d image.

    Parameters
    ----------
    image : ndarray of shape (n_pixels_x, n_pixels_y)s
        A single image to plot.

    """
    check_array(image, ensure_2d=True)
    layout = {
        "title": "Image",
        "plot_bgcolor": "white"
    }
    fig = gobj.Figure(layout=layout)

    fig.add_trace(gobj.Heatmap(z=image,
                               x0=0, dx=1,
                               y0=0, dy=1,
                               colorscale='blues'),
                  )
    fig.show()

