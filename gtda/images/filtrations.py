"""Binary image filtration."""
# License: GNU AGPLv3

import numbers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.metrics import pairwise_distances
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted, check_array
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class HeightFiltration(BaseEstimator, TransformerMixin):
    """Filtrations of 2D/3D binary images based on distances to lines/planes.

    The height filtration assigns to each activated pixel of a binary image a
    grayscale value equal to the distance between the pixel and the hyperplane
    defined by a direction vector and the first seen edge of the image
    following that direction. Deactivated pixels are assigned the value of the
    maximum distance between any pixel of the image and the hyperplane, plus
    one.

    Parameters
    ----------
    direction : ndarray of shape (n_dimensions, 1), optional, default:
        ``np.ones(n_dimensions, 1)``
        Direction of the height filtration, where ``n_dimensions`` is the
        dimension of the images of the collection.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    direction_ : ndarray of shape (n_dimensions_, 1)
        Effective direction of the height filtration. Set in :meth:`fit`.

    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    mesh_ : ndarray of shape ( n_pixels_x, n_pixels_y [, n_pixels_z])
        Grayscale image corresponding to the height filtration of a binary
        image where each pixel is activated. Set in :meth:`fit`.

    max_value_: float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    gtda.homology.CubicalPersistence, Binarizer

    """

    _hyperparameters = {'n_dimensions_': [int, [2, 3]],
                        'direction_': [np.ndarray, (numbers.Number, None)]}

    def __init__(self, direction=None, n_jobs=None):
        self.direction = direction
        self.n_jobs = n_jobs

    def _calculate_height(self, X):
        Xh = np.full(X.shape, self.max_value_)

        for i in range(Xh.shape[0]):
            Xh[i][np.where(X[i])] = np.dot(self.mesh_[np.where(X[i])],
                                           self.direction_).reshape((-1,))

        return Xh

    def fit(self, X, y=None):
        """Calculate `direction_`, `n_dimensions_`, 'mesh_' and `max_value_`
        from a collection of binary images. Then, return the estimator.

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
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        self.n_dimensions_ = len(X.shape) - 1

        if self.direction is None:
            self.direction_ = np.ones((self.n_dimensions_, ))
        else:
            self.direction_ = np.copy(self.direction)

        validate_params({**self.get_params(), 'direction_': self.direction_,
                         'n_dimensions_': self.n_dimensions_},
                        self._hyperparameters)
        print(self.direction_)
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
        corresponding grayscale image based on the distance of its pixels to
        the hyperplane defined by the `direction` vector and the first seen
        edge of the images following that `direction`. Return the collection
        of grayscale images.

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
            2D or 3D grayscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X,  ensure_2d=False, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_height)(X[s])
            for s in gen_even_slices(Xt.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt


@adapt_fit_transform_docs
class RadialFiltration(BaseEstimator, TransformerMixin):
    """Filtrations of 2D/3D binary images based on distances to reference
    pixels.

    The radial filtration assigns to each pixel of a binary image a grayscale
    value computed as follows in terms of a reference pixel, called the
    "center", and of a "radius": if the binary pixel is active and lies
    within a ball defined by this center and this radius, then the assigned
    value equals this distance. In all other cases, the assigned value equals
    the maximum distance between any pixel of the image and the center
    pixel, plus one.

    Parameters
    ----------
    center : ndarray of shape (n_dimensions, ), optional, default:
        ``np.zeros(n_dimensions, )``
        Coordinates of the center pixel, where ``n_dimensions`` is the
        dimension of the images of the collection.

    radius : float or None, default: ``None``
        The radius of the ball centered in `center` inside which
        activated pixels are included in the filtration.

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

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    center_ : ndarray of shape (n_dimensions_, 1)
        Effective center of the radial filtration. Set in :meth:`fit`.

    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params`.
        If `metric_params` is ``None``, it is set to the empty dictionary.

    n_dimensions_ : ``2`` or ``3``
        Dimension of the images. Set in :meth:`fit`.

    mesh_ : ndarray of shape ( n_pixels_x, n_pixels_y [, n_pixels_z])
        Grayscale image corresponding to the radial filtration of a binary
        image where each pixel is activated. Set in :meth:`fit`.

    max_value_: float
        Maximum pixel value among all pixels in all images of the collection.
        Set in :meth:`fit`.

    See also
    --------
    gtda.homology.CubicalPersistence, Binarizer

    """

    _hyperparameters = {'n_dimensions_': [int, [2, 3]],
                        'center_': [np.ndarray, (int, None)]}

    def __init__(self, center=None, radius=np.inf, metric='euclidean',
                 metric_params=None, n_jobs=None):
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
        """Calculate `center_`, 'effective_metric_params_', `n_dimensions_`,
        'mesh_' and `max_value_` from a collection of binary images. Then,
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
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        self.n_dimensions_ = len(X.shape) - 1

        if self.center is None:
            self.center_ = np.zeros((self.n_dimensions_, ))
        else:
            self.center_ = np.copy(self.center)

        validate_params({**self.get_params(), 'center_': self.center_,
                         'n_dimensions_': self.n_dimensions_},
                        self._hyperparameters)

        self.center_ = self.center_.reshape((1, -1))

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        axis_order = [2, 1, 3]
        mesh_range_list = [np.arange(0, X.shape[i])
                           for i in axis_order[:self.n_dimensions_]]

        self.mesh_ = np.stack(
            np.meshgrid(*mesh_range_list),
            axis=self.n_dimensions_).reshape((-1, self.n_dimensions_))
        self.mesh_ = pairwise_distances(
            self.center_, self.mesh_, metric=self.metric,
            n_jobs=1, **self.effective_metric_params_).reshape(X.shape[1:])
        self.mesh_[self.mesh_ > self.radius] = np.inf

        self.max_value_ = 0.
        self.max_value_ = np.max(self._calculate_radial(
            np.ones((1, *X.shape[1:])))) + 1

        return self

    def transform(self, X, y=None):
        """For each binary image in the collection `X`, calculate a
        corresponding grayscale image based on the distance of its pixels to
        the center. Return the collection of grayscale images.

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
            2D or 3D grayscale image.

        """
        check_is_fitted(self)
        Xt = check_array(X,  ensure_2d=False, allow_nd=True, copy=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._calculate_radial)(X[s])
            for s in gen_even_slices(Xt.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        return Xt
