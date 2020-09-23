"""Preprocessing transformers for curves."""

from numbers import Real
from types import FunctionType

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class Derivative(BaseEstimator, TransformerMixin):
    """Computes the derivative of multi-channel curves.

    Given a multi-channel curve computes the corresponding multi-channel
    derivative.

    Parameters
    ----------
    order : int, optional, default: ``1``
        The number of time the multi-channels curves are derived.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """
    _hyperparameters = {
        'order': {'type': int, 'in': Interval(1, np.inf, closed='left')},
    }

    def __init__(self, order=1, n_jobs=None):
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator.

        This function is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input data. Collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_array(X, allow_nd=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.order > X.shape[-1]:
            raise ValueError(f'The number of bins in `X` is not sufficient to '
                             f'calculate its derivative at order {self.order}. '
                             f'It is `n_bins`= {n_bins}.'

        self._is_fitted = True

        return self

    def transform(self, X, y=None):
        """Compute the derivatives of the input multi-channel curves.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_channels, n_bins-order)
            Output collection of the multi-channel curves' derivative.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(np.diff)(Xt[s], n=self.order, axis=-1)
            for s in gen_even_slices(
                    Xt.shape[0], effective_n_jobs(self.n_jobs))
        )
        Xt = np.concatenate(Xt)

        return Xt
