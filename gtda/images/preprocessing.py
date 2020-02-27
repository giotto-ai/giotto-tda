"""Image preprocessing."""
# License: GNU AGPLv3

import numbers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted, check_array
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class Binarizer(BaseEstimator, TransformerMixin):
    """Binarize all 2D/3D grayscale images in a collection.

    Parameters
    ----------
    threshold : float, default: 0.5
        Percentage of the maximum pixel value `max_value_` from which to
        binarize.

    normalize: bool, optional, default: ``False``
        If ``True``, divide every pixel value by `max_value_`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_dimensions_ : int
        Dimension of the images. Set in meth:`fit`.

    max_value_: float
        Maximum pixel value among all pixels in all images of the collection.
        Set in meth:`fit`.

    See also
    --------
    gtda.homology.CubicalPersistence

    """
    _hyperparameters = {'threshold': [numbers.Number, (1e-16, 1)],
                        'normalize': [bool, [False, True]]}

    def __init__(self, threshold=0.5, normalize=False, n_jobs=None):
        self.threshold = threshold
        self.normalize = normalize
        self.n_jobs = n_jobs

    def _binarize(self, X):
        Xbin = X / self.max_value_ > self.threshold

        if self.normalize:
            Xbin = Xbin * self.max_value_

        return Xbin

    def fit(self, X, y=None):
        """Calculate :attr:`n_dimensions_` and :attr:`max_value_` from the
        collection of grayscale images. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y \
            [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            grayscale image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)
        X = check_array(X, allow_nd=True)

        self.n_dimensions_ = len(X.shape) - 1
        self.max_value_ = np.max(X)

        return self

    def transform(self, X, y=None):
        """For each grayscale image in the collection `X`, calculate a
        corresponding binary image by applying the `threshold`. Return the
        collection of binary images.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            grayscale image.

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
        Xt = check_array(X, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._binarize)(Xt[s])
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        if self.n_dimensions_ == 2:
            Xt = Xt.reshape(X.shape)

        return Xt


@adapt_fit_transform_docs
class Inverter(BaseEstimator, TransformerMixin):
    """Invert all 2D/3D binary images in a collection.


    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

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
        X = check_array(X, allow_nd=True)

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
        Xt = check_array(X, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            np.logical_not)(Xt[s])
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt


@adapt_fit_transform_docs
class Padder(BaseEstimator, TransformerMixin):
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

    """
    _hyperparameters = {'paddings_': [np.ndarray, (int, None)],
                        'activated': [bool, [True, False]]}

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
        if self.paddings is None:
            self.paddings_ = np.ones(len(X.shape[1:]), dtype=np.int)
        else:
            self.paddings_ = self.paddings

        n_dimensions = len(X.shape) - 1

        validate_params({**self.get_params(),
                         'paddings_': self.paddings_,
                         'paddings_dim': len(self.paddings_)},
                        {**self._hyperparameters,
                         'paddings_dim': [int, [n_dimensions]]})

        check_array(X, allow_nd=True)

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
        Xt = check_array(X, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            np.pad)(Xt[s], pad_width=self._pad_width,
                    constant_values=self.activated)
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt


@adapt_fit_transform_docs
class ImageToPointCloud(BaseEstimator, TransformerMixin):
    """Represent active pixels in 2D/3D binary images as points in 2D/3D space.

    The coordinates of each point is calculated as follows. For each activated
    pixel, assign coordinates that are the pixel position on this image. All
    deactivated pixels are given infinite coordinates in that space.
    This transformer is meant to transform a collection of images to a point
    cloud so that collection of point clouds-based persistent homology module
    can be applied.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    mesh_ : ndarray, shape (n_pixels_x * n_pixels_y [* n_pixels_z], \
        n_dimensions)
        Mesh image for which each pixel value is its coordinates in a
        `n_dimensions` space, where `n_dimensions` is the dimension of the
        images of the input collection. Set in meth:`fit`.

    See also
    --------
    gtda.homology.VietorisRipsPersistence, gtda.homology.SparseRipsPersistence,
    gtda.homology.EuclideanCechPersistence

    """
    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _embed(self, X):
        Xpts = np.stack([self.mesh_ for _ in range(X.shape[0])]) * 1.0
        Xpts[np.logical_not(X.reshape((X.shape[0], -1))), :] += np.inf
        return Xpts

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
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

        n_dimensions = len(X.shape) - 1
        axis_order = [2, 1, 3]
        mesh_range_list = [np.arange(0, X.shape[i])
                           for i in axis_order[:n_dimensions]]

        self.mesh_ = np.flip(np.stack(np.meshgrid(*mesh_range_list),
                                      axis=n_dimensions),
                             axis=0).reshape((-1, n_dimensions))

        return self

    def transform(self, X, y=None):
        """For each collection of binary images, calculate the corresponding
        collection of point clouds based on the coordinates of activated
        pixels.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            binary image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_pixels_x * n_pixels_y [* n_pixels_z],
            n_dimensions)
            Transformed collection of images. Each entry along axis 0 is a
            point cloud in a `n_dimensions` dimensional space.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            self._embed)(Xt[s])
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)
        return Xt
