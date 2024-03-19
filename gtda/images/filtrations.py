"""Filtrations of 2D/3D binary images."""
# License: GNU AGPLv3

from numbers import Real, Integral
from typing import Callable
import itertools

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_array, check_is_fitted

from ._utils import _dilate, _erode
from .preprocessing import Padder
from ..base import PlotterMixin
from ..plotting import plot_heatmap
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class HeightFiltration(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtrations of 2D/3D binary images based on distances to lines/planes.

    The height filtration assigns to each activated pixel of a binary image a
    greyscale value equal to the distance between the pixel and the hyperplane
    defined by a direction vector and the first seen edge of the image
    following that direction. Deactivated pixels are assigned the value of the
    maximum distance between any pixel of the image and the hyperplane, plus
    one.

    Parameters
    ----------
    direction : ndarray of shape (n_dimensions,) or None, optional, default: \
        ``None``
        Direction vector of the height filtration in
        ``n_dimensions``-dimensional space, where ``n_dimensions`` is the
        dimension of the images of the collection (2 or 3). ``None`` is
        equivalent to passing ``numpy.ones(n_dimensions)``.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    direction_ : ndarray of shape (:attr:`n_dimensions_`,)
        Effective direction of the height filtration. Set in :meth:`fit`.

    mesh_ : ndarray of shape ( n_pixels_x, n_pixels_y [, n_pixels_z])
        greyscale image corresponding to the height filtration of a binary
        image where each pixel is activated. Set in :meth:`fit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    RadialFiltration, DilationFiltration, ErosionFiltration, \
    SignedDistanceFiltration, DensityFiltration, \
    gtda.homology.CubicalPersistence

    References
    ----------
    .. [1] A. Garin and G. Tauzin, "A topological reading lesson:
           Classification of MNIST using TDA"; 19th International IEEE
           Conference on Machine Learning and Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'direction': {'type': (np.ndarray, type(None)), 'of': {'type': Real}}
        }

    def __init__(self, direction=None, n_jobs=None):
        self.direction = direction
        self.n_jobs = n_jobs

    def _calculate_height(self, X):
        Xh = np.full(X.shape, self.max_value_)

        for i in range(len(Xh)):
            Xh[i][np.where(X[i])] = np.dot(self.mesh_[np.where(X[i])],
                                           self.direction_).reshape((-1,))

        return Xh

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`direction_`, :attr:`mesh_`
        and :attr:`max_value_` from a collection of binary images. Then,
        return the estimator.

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
        self.n_dimensions_ = X.ndim - 1
        if self.n_dimensions_ > 3:
            raise ValueError(f"Input of `fit` contains arrays of dimension "
                             f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.direction is None:
            self.direction_ = np.ones(self.n_dimensions_,)
        else:
            self.direction_ = np.copy(self.direction)
        self.direction_ = self.direction_ / np.linalg.norm(self.direction_)

        axis_order = [2, 1, 3]
        mesh_range_list = \
            [np.arange(X.shape[order]) if self.direction_[i] >= 0
             else -np.flip(np.arange(X.shape[order])) for i, order
             in enumerate(axis_order[: self.n_dimensions_])]

        self.mesh_ = np.stack(np.meshgrid(*mesh_range_list, indexing='xy'),
                              axis=self.n_dimensions_)

        self.max_value_ = 0.
        self.max_value_ = np.max(self._calculate_height(
            np.ones((1, *X.shape[1:])))) + 1

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding greyscale image based on the distance of its pixels to
        the hyperplane defined by the `direction` vector and the first seen
        edge of the images following that `direction`. Return the collection
        of greyscale images.

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
            2D or 3D greyscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_height)(X[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample from a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
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

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Height filtration of image {sample}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class RadialFiltration(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtrations of 2D/3D binary images based on distances to a reference
    pixel.

    The radial filtration assigns to each pixel of a binary image a greyscale
    value computed as follows in terms of a reference pixel, called the
    "center", and of a "radius": if the binary pixel is active and lies
    within a ball defined by this center and this radius, then the assigned
    value equals this distance. In all other cases, the assigned value equals
    the maximum distance between any pixel of the image and the center
    pixel, plus one.

    Parameters
    ----------
    center : ndarray of shape (:attr:`n_dimensions_`,) or None, optional,\
        default: ``None``
        Coordinates of the center pixel, where ``n_dimensions`` is the
        dimension of the images of the collection (2 or 3). ``None`` is
        equivalent to passing ``np.zeros(n_dimensions,)```.

    radius : float or None, default: ``None``
        The radius of the ball centered in `center` inside which activated
        pixels are included in the filtration.

    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, each entry in `X` along axis 0 is
        interpreted to be a distance matrix. Otherwise, entries are
        interpreted as feature arrays, and `metric` determines a rule with
        which to calculate distances between pairs of instances (i.e. rows)
        in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan" or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    metric_params : dict or None, optional, default: ``{}``
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    center_ : ndarray of shape (:attr:`n_dimensions_`,)
        Effective center of the radial filtration. Set in :meth:`fit`.

    mesh_ : ndarray of shape ( n_pixels_x, n_pixels_y [, n_pixels_z])
        greyscale image corresponding to the radial filtration of a binary
        image where each pixel is activated. Set in :meth:`fit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    HeightFiltration, DilationFiltration, ErosionFiltration, \
    SignedDistanceFiltration, DensityFiltration, \
    gtda.homology.CubicalPersistence

    References
    ----------
    .. [1] A. Garin and G. Tauzin, "A topological reading lesson:
           Classification of MNIST using TDA"; 19th International IEEE
           Conference on Machine Learning and Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'center': {'type': (np.ndarray, type(None)), 'of': {'type': Integral}},
        'radius': {'type': Real, 'in': Interval(0, np.inf, closed='right')},
        'metric': {'type': (str, Callable)},
        'metric_params': {'type': dict}
        }

    def __init__(self, center=None, radius=np.inf, metric='euclidean',
                 metric_params={}, n_jobs=None):
        self.center = center
        self.radius = radius
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _calculate_radial(self, X):
        Xr = np.nan_to_num(self.mesh_ * X, nan=np.inf, posinf=np.inf)
        Xr = np.nan_to_num(Xr, posinf=-1)

        Xr[X == 0] = self.max_value_
        Xr[Xr == -1] = self.max_value_

        return Xr

    def fit(self, X, y=None):
        """Calculate :attr:`center_`, :attr:`n_dimensions_`, :attr:`mesh_` and
        :attr:`max_value_` from a collection of binary images. Then, return the
        estimator.

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
        self.n_dimensions_ = X.ndim - 1
        if self.n_dimensions_ > 3:
            raise ValueError(f"Input of `fit` contains arrays of dimension "
                             f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.center is None:
            self.center_ = np.zeros(self.n_dimensions_)
        else:
            self.center_ = np.copy(self.center)
        self.center_ = self.center_.reshape((1, -1))

        axis_order = [2, 1, 3]
        mesh_range_list = [np.arange(0, X.shape[i])
                           for i in axis_order[:self.n_dimensions_]]

        self.mesh_ = np.stack(
            np.meshgrid(*mesh_range_list),
            axis=self.n_dimensions_).reshape((-1, self.n_dimensions_))
        self.mesh_ = pairwise_distances(
            self.center_, self.mesh_, metric=self.metric,
            n_jobs=1, **self.metric_params).reshape(X.shape[1:])
        self.mesh_[self.mesh_ > self.radius] = np.inf

        self.max_value_ = 0.
        self.max_value_ = \
            np.max(self._calculate_radial(np.ones((1, *X.shape[1:])))) + 1

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding greyscale image based on the distance of its pixels to
        the center. Return the collection of greyscale images.

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
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_radial)(X[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample from a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
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

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Radial filtration of image {sample}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class DilationFiltration(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtrations of 2D/3D binary images based on the dilation of activated
    regions.

    Binary dilation is a morphological operator commonly used in
    image processing and relies on the `scipy.ndimage \
    <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_ module.

    This filtration assigns to each pixel in an image a greyscale value
    calculated as follows. If the minimum Manhattan distance between the
    pixel and any activated pixel in the image is less than or equal to
    the parameter `n_iterations`, the assigned value is this distance –
    in particular, activated pixels are assigned a value of 0.
    Otherwise, the assigned greyscale value is the sum of the lengths
    along all axes of the image – equivalently, it is the maximum
    Manhattan distance between any two pixels in the image. The name of
    this filtration comes from the fact that these values can be computed
    by iteratively dilating activated regions, thickening them by a total
    amount `n_iterations`.

    Parameters
    ----------
    n_iterations : int or None, optional, default: ``None``
        Number of iterations in the dilation process. ``None`` means dilation
        reaches all deactivated pixels.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    n_iterations_ : int
        Effective number of iterations in the dilation process. Set in
        :meth:`fit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    HeightFiltration, RadialFiltration, ErosionFiltration, \
    SignedDistanceFiltration, DensityFiltration, \
    gtda.homology.CubicalPersistence

    References
    ----------
    .. [1] A. Garin and G. Tauzin, "A topological reading lesson:
           Classification of MNIST using TDA"; 19th International IEEE
           Conference on Machine Learning and Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'n_iterations': {'type': (int, type(None)),
                         'in': Interval(1, np.inf, closed='left')}
        }

    def __init__(self, n_iterations=None, n_jobs=None):
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs

    def _calculate_dilation(self, X):
        Xd = _dilate(X, 1, self.n_iterations_, 1, self.max_value_)

        mask_undilated = Xd == 0
        Xd -= 1
        Xd[mask_undilated] = self.max_value_
        return Xd

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`n_iterations_` and
        :attr:`max_value_` from a collection of binary images. Then, return the
        estimator.

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
        self.n_dimensions_ = X.ndim - 1
        if self.n_dimensions_ > 3:
            raise ValueError(f"Input of `fit` contains arrays of dimension "
                             f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self.max_value_ = np.sum(X.shape[1:])

        if self.n_iterations is None:
            self.n_iterations_ = int(self.max_value_)
        else:
            self.n_iterations_ = self.n_iterations

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding greyscale image based on the distance of its pixels to
        their closest activated neighboring pixel. Return the collection
        of greyscale images.

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
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_dilation)(X[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample from a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
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

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Dilation filtration of image {sample}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class ErosionFiltration(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtrations of 2D/3D binary images based on the erosion of activated
    regions.

    Binary erosion is a morphological operator commonly used in
    image processing and relies on the `scipy.ndimage \
    <https://docs.scipy.org/doc/scipy/reference/ndimage.html>`_ module.

    This filtration assigns to each pixel in an image a greyscale value
    calculated as follows. If the minimum Manhattan distance between the
    pixel and any deactivated pixel in the image is less than or equal to
    the parameter `n_iterations`, the assigned value is this distance –
    in particular, deactivated pixels are assigned a value of 0.
    Otherwise, the assigned greyscale value is the sum of the lengths
    along all axes of the image – equivalently, it is the maximum
    Manhattan distance between any two pixels in the image. The name of
    this filtration comes from the fact that these values can be computed
    by iteratively eroding activated regions, shrinking them by a total
    amount `n_iterations`.

    Parameters
    ----------
    n_iterations : int or None, optional, default: ``None``
        Number of iterations in the erosion process. ``None`` means erosion
        reaches all activated pixels.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    n_iterations_ : int
        Effective number of iterations in the erosion process. Set in
        :meth:`fit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    HeightFiltration, RadialFiltration, DilationFiltration, \
    SignedDistanceFiltration, DensityFiltration, \
    gtda.homology.CubicalPersistence

    References
    ----------
    .. [1] A. Garin and G. Tauzin, "A topological reading lesson:
           Classification of MNIST using TDA"; 19th International IEEE
           Conference on Machine Learning and Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'n_iterations': {'type': (int, type(None)),
                         'in': Interval(1, np.inf, closed='left')}
        }

    def __init__(self, n_iterations=None, n_jobs=None):
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs

    def _calculate_erosion(self, X):
        Xe = _erode(X, 1, self.n_iterations_, 1, self.max_value_)

        mask_uneroded = Xe == 0
        Xe -= 1
        Xe[mask_uneroded] = self.max_value_
        return Xe

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`n_iterations_` and
        :attr:`max_value_` from a collection of binary images. Then, return the
        estimator.

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
        self.n_dimensions_ = X.ndim - 1
        if self.n_dimensions_ > 3:
            raise ValueError(f"Input of `fit` contains arrays of dimension "
                             f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self.max_value_ = np.sum(X.shape[1:])

        if self.n_iterations is None:
            self.n_iterations_ = int(self.max_value_)
        else:
            self.n_iterations_ = self.n_iterations

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding greyscale image based on the distance of its pixels to
        their closest activated neighboring pixel. Return the collection
        of greyscale images.

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
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_erosion)(X[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample from a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
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

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Erosion filtration of image {sample}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class SignedDistanceFiltration(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtrations of 2D/3D binary images based on the dilation and the erosion
    of activated regions.

    This filtration assigns to each pixel in an image a greyscale value
    calculated as follows. For activated pixels, if the minimum Manhattan
    distance between the pixel and any deactivated pixel in the image is less
    than or equal to the parameter `n_iterations`, the assigned value is
    this distance minus 1. Otherwise, the assigned greyscale value is the sum
    of the lengths along all axes of the image – equivalently, it is the
    maximum Manhattan distance between any two pixels in the image, minus 1.
    For deactivated pixels, if the minimum Manhattan distance between the pixel
    and any activated pixel in the image is less than or equal to the parameter
    `n_iterations`, the assigned value is the opposite of this distance.
    Otherwise, the assigned greyscale value is the opposite of the maximum
    Manhattan distance between any two pixels in the image.

    The name of this filtration comes from the fact that it is a a negatively
    signed dilation plus a positively signed erosion, minus 1 on the activated
    pixels. Therefore, pixels the activated pixels at the boundary of the
    activated regions always have a pixel value of 0.

    Parameters
    ----------
    n_iterations : int or None, optional, default: ``None``
        Number of iterations in the dilation process. ``None`` means dilation
        over the full image.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    n_iterations_ : int
        Effective number of iterations in the dilation process. Set in
        :meth:`fit`.

    max_value_ : float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    HeightFiltration, RadialFiltration, DilationFiltration, \
    ErosionFiltration, DensityFiltration, gtda.homology.CubicalPersistence

    References
    ----------
    .. [1] A. Garin and G. Tauzin, "A topological reading lesson:
           Classification of MNIST using TDA"; 19th International IEEE
           Conference on Machine Learning and Applications (ICMLA 2020), 2019;
           `arXiv:1910.08345 <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'n_iterations': {'type': (int, type(None)),
                         'in': Interval(1, np.inf, closed='left')}
        }

    def __init__(self, n_iterations=None, n_jobs=None):
        self.n_iterations = n_iterations
        self.n_jobs = n_jobs

    def _calculate_signed_distance(self, X):
        mask = X == 1

        Xd = -_dilate(X, 1, self.n_iterations_, 0, self.max_value_)
        Xe = _erode(X, 0, self.n_iterations_, 0, self.max_value_)

        mask_e = Xe == 0
        mask_d = Xd == 0
        Xe[np.logical_not(mask)] = 0
        Xe[mask] -= 1
        Xd[mask] = 0
        Xd[mask_d] = -self.max_value_
        Xe[mask_e] = self.max_value_
        return (Xd + Xe)

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_`, :attr:`n_iterations_` and
        :attr:`max_value_` from a collection of binary images. Then, return the
        estimator.

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
        self.n_dimensions_ = X.ndim - 1
        if self.n_dimensions_ > 3:
            raise ValueError(f"Input of `fit` contains arrays of dimension "
                             f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        self.max_value_ = np.sum(X.shape[1:])

        if self.n_iterations is None:
            self.n_iterations_ = int(self.max_value_)
        else:
            self.n_iterations_ = self.n_iterations

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding greyscale image based on the distance of its pixels to
        their closest activated neighboring pixel. Return the collection
        of greyscale images.

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
        Xt : ndarray of shape (n_samples, n_pixels_x,
            n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D greyscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_signed_distance)(X[s])
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample from a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
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

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Signed-distance filtration of image {sample}",
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class DensityFiltration(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtrations of 2D/3D binary images based on the number of activated
    neighboring pixels.

    The density filtration assigns to each pixel of a binary image a greyscale
    value equal to the number of activated pixels within a ball centered around
    it.

    Parameters
    ----------
    radius : float, optional, default: ``1.``
        The radius of the ball within which the number of activated pixels is
        considered.

    metric : string or callable, optional, default: ``'euclidean'``
        Determines a rule with which to calculate distances between
        pairs of pixels.
        If ``metric`` is a string, it must be one of the options allowed by
        ``scipy.spatial.distance.pdist`` for its metric parameter, or a metric
        listed in ``sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS``, including
        "euclidean", "manhattan", or "cosine".
        If ``metric`` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    metric_params : dict, optional, default: ``{}``
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    mask_ : ndarray of shape (radius, radius [, radius])
        The mask applied around each pixel to calculate the weighted number of
        its activated neighbors. Set in :meth:`fit`.

    See also
    --------
    HeightFiltration, RadialFiltration, DilationFiltration, \
    ErosionFiltration, SignedDistanceFiltration, \
    gtda.homology.CubicalPersistence

    References
    ----------
    [1] A. Garin and G. Tauzin, "A topological reading lesson: Classification
        of MNIST  using  TDA"; 19th International IEEE Conference on Machine
        Learning and Applications (ICMLA 2020), 2019; arXiv: `1910.08345 \
        <https://arxiv.org/abs/1910.08345>`_.

    """

    _hyperparameters = {
        'radius': {'type': Real, 'in': Interval(0, np.inf, closed='right')},
        'metric': {'type': (str, Callable)},
        'metric_params': {'type': dict},
        }

    def __init__(self, radius=3, metric='euclidean', metric_params={},
                 n_jobs=None):
        self.radius = radius
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def _calculate_density(self, X):
        Xd = np.zeros(X.shape)

        # The idea behind this is to sum up pixel values of the image
        # rolled according to the 3D mask
        for i, j, k in self._iterator:
            Xd += np.roll(np.roll(
                np.roll(X, k, axis=3), j, axis=2), i, axis=1) \
                * self.mask_[self._size + i, self._size + j,
                             self._size + k]
        return Xd

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_` and :attr:`mask_` from a collection
        of binary images. Then, return the estimator.

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
        self.n_dimensions_ = X.ndim - 1
        if self.n_dimensions_ > 3:
            raise ValueError(f"Input of `fit` contains arrays of dimension "
                             f"{self.n_dimensions_}.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        # Determine the size of the mask based on the radius and metric
        self._size = int(np.ceil(
            pairwise_distances([[0]], [[self.radius]], metric=self.metric,
                               **self.metric_params)
            ))
        # The mask is always 3D but not the iterator.
        self.mask_ = np.ones(tuple(2 * self._size + 1 for _ in range(3)),
                             dtype=bool)

        # Create an iterator for applying the mask to every pixel at once
        iterator_size_list = \
            [range(-self._size, self._size + 1)] * self.n_dimensions_ + \
            [[0] for _ in range(3 - self.n_dimensions_)]
        self._iterator = tuple(itertools.product(*iterator_size_list))

        # We create a mesh so that we have an array with coordinates and we can
        # calculate the distance of each point to the center
        mesh_size_list = [np.arange(0, 2 * self._size + 1)] * 3
        self.mesh_ = np.stack(
            np.meshgrid(*mesh_size_list), axis=3).reshape((-1, 3))

        # Calculate those distances to the center and use them to set the mask
        # values so that it corresponds to a ball
        center = self._size * np.ones((1, 3))
        self.mask_ = pairwise_distances(
            center, self.mesh_, metric=self.metric,
            n_jobs=1, **self.metric_params).reshape(self.mask_.shape)

        self.mask_ = self.mask_ <= self.radius

        # Instantiate a padder to pad all images with 0 so that the rolling of
        # the mask also works at the boundary of the images
        padding = np.asarray([*[self._size] * self.n_dimensions_,
                              *[0] * (3 - self.n_dimensions_)])
        self._padder = Padder(padding=padding)
        self._padder.fit(X.reshape((*X.shape[:3], -1)))

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding greyscale image based on the density of its pixels.
        Return the collection of greyscale images.

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
            2D or 3D greyscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True, copy=True)

        # Reshape the images to 3D so that they can be rolled according to the
        # 3D mask
        Xt = Xt.reshape((*X.shape[:3], -1))
        Xt = self._padder.transform(Xt)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_density)(Xt[s])
            for s in gen_even_slices(Xt.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        Xt = Xt[:, self._size: -self._size, self._size: -self._size]

        if self.n_dimensions_ == 3:
            Xt = Xt[:, :, :, self._size: -self._size]

        Xt = Xt.reshape(X.shape)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='greys', origin='upper',
             plotly_params=None):
        """Plot a sample from a collection of 2D greyscale images.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_pixels_x, n_pixels_y)
            Collection of 2D greyscale images, such as returned by
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

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale, origin=origin,
            title=f"Density filtration of image {sample}",
            plotly_params=plotly_params
            )
