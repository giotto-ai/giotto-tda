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
    """Transformer returning a collection of binary images from an input
    collection of 2D or 3D grayscale images.

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
        """Calculate `n_dimensions` and `max_value_` of the collection of grayscale
        image. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
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
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        self.n_dimensions_ = len(X.shape) - 1
        self.max_value_ = np.max(X)

        return self

    def transform(self, X, y=None):
        """For each collection of grayscale images, calculate the corresponding
        collection of binary images based on the threshold.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            grayscale image.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Transformed collection of images. Each entry along axis 0 is a
            2D or 3D binary images.
        """

        check_is_fitted(self)
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(self._binarize)(X[s])
            for s in gen_even_slices(X.shape[0], effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        if self.n_dimensions_ == 2:
            Xt = Xt.reshape((*X.shape))

        return Xt
