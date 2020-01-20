"""Implements a TransformerResamplerMixin for transformers that have a resample
method."""
# License: GNU AGPLv3


class TransformerResamplerMixin:
    """Mixin class for all transformers-resamplers in giotto-tda."""

    _estimator_type = 'transformer_resampler'

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.
        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : numpy array of shape (n_samples, ...)
            Transformed input.
        """
        # non-optimized default implementation; override when a better
        # method is possible for a given clustering algorithm
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X, y)

    def transform_resample(self, X, y):
        """Fit to data, then transform it.

        Fits transformer to `X` and `y` with optional parameters `fit_params`
        and returns a transformed version of `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.
        y : ndarray of shape (n_samples, )
            Target data.

        Returns
        -------
        Xt : ndarray of shape (n_samples, ...)
            Transformed input.
        """
        return self.transform(X), self.resample(y, X)

    def fit_transform_resample(self, X, y, **fit_params):
        """Fit to data, then transform the input and resample the target.
        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X ans a resampled version of y.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.
        y : ndarray of shape (n_samples, )
            Target data.

        Returns
        -------
        Xt : ndarray of shape (n_samples, ...)
            Transformed input.
        yr : ndarray of shape (n_samples, ...)
            Resampled target.
        """
        return self.fit(X, y, **fit_params).transform_resample(X, y)
