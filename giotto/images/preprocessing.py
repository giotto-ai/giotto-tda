"""Image preprocessing."""
# License: Apache 2.0

import numbers
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils.validation import check_is_fitted, check_array
from ..utils.validation import validate_params


class Binarizer(BaseEstimator, TransformerMixin):
    """Transformer returning a collection of binary image from an input
    collection of 2D or 3D boolean images.

    Parameters
    ----------
    threshold : float, default: 0.5
        Percentage of the maximum pixel value of the collection of image from
        which to binarize. The maximum pixel value is calculated in the meth:`fit`.

    normalize: bool, optional, default: ``False``
        If set to ``True``, divide every pixel values by the maximum pixel value
        of the collection of image from which to binarize. The maximum pixel
        value is calculated in the meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """
    _hyperparameters = {'threshold': [numbers.Number, (1e-16, 1)]}

    def __init__(self, threshold=0.5, normalize=False, n_jobs=None):
        self.threshold = threshold
        self.n_jobs = n_jobs

    def _binarize(self, X):
        Xbin = X / self.max_value_ > self.threshold

        if self.normalize:
            Xbin = X_bin * self.max_value_

        return Xbin

    def fit(self, X, y=None):
        """Calculate the dimension and max value of the image. Then, return the
        estimator.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            boolean images.

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
        """For each collection of images, calculate the corresponding collection
        of binary images based on the threshold.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Input data. Each entry along axis 0 is interpreted as a 2D or 3D
            boolean images.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_pixels_x, n_pixels_y [, n_pixels_z])
            Collection of binary images for which each entry along axis 0 is a
            2D or 3D binary images.
        """

        # Check is fit had been called
        check_is_fitted(self)
        X = check_array(X,  ensure_2d=False, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(self._binarize)(
            X[s]) for s in gen_even_slices(X.shape[0],
                                           effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt)

        if self.n_dimensions_ == 2:
            Xt = Xt.reshape((*X.shape))

        return Xt
