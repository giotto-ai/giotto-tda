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

    def plot(self, Xt, sample=0):
        """Plot a single image.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Transformed collection of images. Each entry along axis 0 is a
            2D image.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        """
        X_to_plot = Xt[sample]
        if X_to_plot.ndim > 3:
            raise RuntimeError("Plotting images in more than 2 dimensions"
                               " is not supported.")
        return plot_image(X_to_plo)
