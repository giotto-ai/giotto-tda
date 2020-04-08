"""Image preprocessing module."""
# License: GNU AGPLv3

from functools import reduce
from operator import iconcat
from numbers import Real
from warnings import warn

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_array, check_is_fitted

from ..base import PlotterMixin
from ..plotting import plot_point_cloud, plot_heatmap
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class Binarizer(BaseEstimator, TransformerMixin, PlotterMixin):
    """Binarize all 2D/3D greyscale images in a collection.

    Parameters
    ----------
    threshold : float, default: 0.5
        Fraction of the maximum pixel value `max_value_` from which to
        binarize.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : int
        Dimension of the images. Set in meth:`fit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of the collection.
        Set in meth:`fit`.

    See also
    --------
    gtda.homology.CubicalPersistence

    References
    ----------
    [1] A. Garin and G. Tauzin, "A topological reading lesson: Classification
        of MNIST  using  TDA"; 19th International IEEE Conference on Machine
        Learning and Applications (ICMLA 2020), 2019; arXiv: `1910.08345 \
        <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'threshold': {'type': Real, 'in': Interval(0, 1, closed='right')}
    }

    def __init__(self, threshold=0.5, n_jobs=None):
        self.threshold = threshold
        self.n_jobs = n_jobs

    def _binarize(self, X):
        Xbin = X / self.max_value_ > self.threshold

        return Xbin

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_` and :attr:`max_value_` from the
        collection of greyscale images. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            greyscale image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_array(X, allow_nd=True)
        self.n_dimensions_ = X.ndim - 1
        if (self.n_dimensions_ < 2) or (self.n_dimensions_ > 3):
            warn(f"Input of `fit` contains arrays of dimension "
                 f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self.max_value_ = np.max(X)

        return self

    def transform(self, X, y=None):
        """For each greyscale image in the collection `X`, calculate a
        corresponding binary image by applying the `threshold`. Return the
        collection of binary images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            greyscale image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D binary image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._binarize)(Xt[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        if self.n_dimensions_ == 2:
            Xt = Xt.reshape(X.shape)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper'):
        """Plot a sample from a collection of 2D binary images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D binary images, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of the [0, 0] pixel of `data`, in the upper left or lower
            left corner. The convention ``'upper'`` is typically used for
            matrices and images.

        """
        return plot_heatmap(
            Xt[sample] * 1, colorscale=colorscale, origin=origin)


@adapt_fit_transform_docs
class Inverter(BaseEstimator, TransformerMixin, PlotterMixin):
    """Invert all 2D/3D binary images in a collection.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    References
    ----------
    [1] A. Garin and G. Tauzin, "A topological reading lesson: \
        Classification  of MNIST  using  TDA"; 19th International \
        IEEE Conference on Machine Learning and Applications (ICMLA 2020), \
        2019; arXiv: `1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate its negation.
        Return the collection of negated binary images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D binary image.

        """
        check_is_fitted(self, ['_is_fitted'])
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            np.logical_not)(Xt[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper'):
        """Plot a sample from a collection of 2D binary images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D binary images, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of the [0, 0] pixel of `data`, in the upper left or lower
            left corner. The convention ``'upper'`` is typically used for
            matrices and images.

        """
        return plot_heatmap(
            Xt[sample] * 1, colorscale=colorscale, origin=origin)


@adapt_fit_transform_docs
class Padder(BaseEstimator, TransformerMixin, PlotterMixin):
    """Pad all 2D/3D binary images in a collection.

    Parameters
    ----------
    paddings : int ndarray of shape (padding_x, padding_y [, padding_z]) or \
        None, optional, default: ``None``
        Number of pixels to pad the images along each axis and on both side of
        the images. By default, a frame of a single pixel width is added
        around the image (``1 = padding_x = padding_y [= padding_z]``).

    activated : bool, optional, default: ``False``
        If ``True``, the padded pixels are activated. If ``False``, they are
        deactivated.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    paddings_ : int ndarray of shape (padding_x, padding_y [, padding_z])
       Effective padding along each of the axis. Set in :meth:`fit`.

    References
    ----------
    [1] A. Garin and G. Tauzin, "A topological reading lesson: Classification
        of MNIST  using  TDA"; 19th International IEEE Conference on Machine
        Learning and Applications (ICMLA 2020), 2019; arXiv: `1910.08345 \
        <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'paddings': {
            'type': (np.ndarray, type(None)),
            'of': {'type': int}},
        'activated': {'type': bool}
    }

    def __init__(self, paddings=None, activated=False, n_jobs=None):
        self.paddings = paddings
        self.activated = activated
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Calculate :attr:`paddings_` from a collection of binary images.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_array(X, allow_nd=True)
        n_dimensions = X.ndim - 1
        if n_dimensions < 2 or n_dimensions > 3:
            warn(f"Input of `fit` contains arrays of dimension "
                 f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.paddings is None:
            self.paddings_ = np.ones((n_dimensions,), dtype=np.int)
        elif len(self.paddings) != n_dimensions:
            raise ValueError(
                f"`paddings` has length {self.paddings} while the input "
                f"data requires it to have length equal to {n_dimensions}.")
        else:
            self.paddings_ = self.paddings

        self._pad_width = ((0, 0),
                           *[(self.paddings_[axis], self.paddings_[axis])
                             for axis in range(n_dimensions)])

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, adds a padding.
        Return the collection of padded binary images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x + 2 * padding_x, \
            n_pixels_y + 2 * padding_y [, n_pixels_z + 2 * padding_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D binary image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            np.pad)(Xt[s], pad_width=self._pad_width,
                    constant_values=self.activated)
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper'):
        """Plot a sample from a collection of 2D binary images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D binary images, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        colorscale : str, optional, default: ``'greys'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        origin : ``'upper'`` | ``'lower'``, optional, default: ``'upper'``
            Position of the [0, 0] pixel of `data`, in the upper left or lower
            left corner. The convention ``'upper'`` is typically used for
            matrices and images.

        """
        return plot_heatmap(
            Xt[sample] * 1, colorscale=colorscale, origin=origin)


@adapt_fit_transform_docs
class ImageToPointCloud(BaseEstimator, TransformerMixin, PlotterMixin):
    """Represent active pixels in 2D/3D binary images as points in 2D/3D space.

    The coordinates of each point is calculated as follows. For each activated
    pixel, assign coordinates that are the pixel index on this image, after
    flipping the rows and then swapping between rows and columns.

    This transformer is meant to transform a collection of images to a
    collection of point clouds so that persistent homology calculations can be
    performed.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    See also
    --------
    gtda.homology.VietorisRipsPersistence, gtda.homology.SparseRipsPersistence,
    gtda.homology.EuclideanCechPersistence

    References
    ----------
    [1] A. Garin and G. Tauzin, "A topological reading lesson: Classification
        of MNIST  using  TDA"; 19th International IEEE Conference on Machine
        Learning and Applications (ICMLA 2020), 2019; arXiv: `1910.08345 \
        <https://arxiv.org/abs/1910.08345>`_.

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _embed(self, X):
        return [np.argwhere(x) for x in X]

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)

        n_dimensions = X.ndim - 1
        if n_dimensions < 2 or n_dimensions > 3:
            warn(f"Input of `fit` contains arrays of dimension "
                 f"{self.n_dimensions_}.")

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each collection of binary images, calculate the corresponding
        collection of point clouds based on the coordinates of activated
        pixels.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_pixels_x * n_pixels_y [* \
            n_pixels_z],
            n_dimensions)
            Transformed collection of images. Each entry along axis 0 is a
            point cloud in ``n_dimensions``-dimensional space.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X, allow_nd=True)

        Xt = np.swapaxes(np.flip(Xt, axis=1), 1, 2)
        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._embed)(Xt[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = reduce(iconcat, Xt, [])
        return Xt

    @staticmethod
    def plot(Xt, sample=0):
        """Plot a sample from a collection of point clouds. If the point cloud
        is in more than three dimensions, only the first three are plotted.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, n_dimensions)
            Collection of point clouds in ``n_dimension``-dimensional space,
            such as returned by :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        """
        return plot_point_cloud(Xt[sample])
