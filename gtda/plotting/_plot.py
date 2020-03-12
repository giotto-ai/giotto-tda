
import plotly.graph_objects as gobj
from sklearn.utils.validation import check_array


def _plot_image(image):
    """Plot a 2d image.

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
    return fig
