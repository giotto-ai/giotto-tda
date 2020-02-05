"""Binary image filtration."""
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
class HeightFiltration(BaseEstimator, TransformerMixin):
    """Transformer returning a collection of grayscale images
    from a collection of 2D or 3D binary images.

    The height filtration assigns to each activated pixel of an image a pixel
    value corresponding to the distance between the pixel and the hyperplane
    defined by a direction vector and the first seen edge of the image
    following that direction. Deactivated pixels are assigned the value of the
    maximum distance between any pixel of the image and the hyperplane plus
    one.

    Parameters
    ----------
    direction : ndarray of shape (n_dimensions, ), optional, default:
        ``np.ones(n_dimensions, )``
        Direction of the height filtration, where ``n_dimensions`` is the
        dimension of the images of the collection.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    direction_ : ndarray of shape (n_dimensions_, )
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
        """Calculate `n_dimensions_`, 'mesh_' and `max_value_` from a
        collection of binary image. Then, return the estimator.

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
        the hyperplane defined by the ``direction`` vector and the first seen
        edge of the images following that ``direction``. Return the collection
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
