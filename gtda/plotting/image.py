"""Image-retated plotting functions """
# License: GNU AGPLv3

from ._plot import _plot_image
from ..base import PlotterMixin


def plot_image(image):
    """Plot a 2d image.

    Parameters
    ----------
    image : ndarray of shape (n_pixels_x, n_pixels_y)
        A single image to plot.

    """
    fig = _plot_image(image)
    fig.show()


class ImagePlotterMixin(PlotterMixin):
    """Mixin class for modules that return images"""

    def plot(self, Xt, sample=0, **layout):
        """Plot a single persistence diagram.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        layout : dict
            Dict of string/value properties that will be passed to the
            `plotly.graph_objs.Layout` constructor. For supported properties,
            see :class:`plotly.graph_objects.Layout`
        """
        return plot_image(Xt[sample])
