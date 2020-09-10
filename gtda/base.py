"""Implements a TransformerResamplerMixin for transformers that have a resample
method and TransformerPlotterMixin for transformers that have a plot method."""
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
            There is no need for a target, yet the pipeline API requires this
            parameter.

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

        y : ndarray of shape (n_samples,)
            Target data.

        Returns
        -------
        Xt : ndarray of shape (n_samples, ...)
            Transformed input.

        yr : ndarray of shape (n_samples, ...)
            Resampled target.

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

        y : ndarray of shape (n_samples,)
            Target data.

        Returns
        -------
        Xt : ndarray of shape (n_samples, ...)
            Transformed input.

        yr : ndarray of shape (n_samples, ...)
            Resampled target.

        """
        return self.fit(X, y, **fit_params).transform_resample(X, y)


class PlotterMixin:
    """Mixin class for all plotters in giotto-tda."""

    def fit_transform_plot(self, X, y=None, sample=0, **plot_params):
        """Fit to data, then apply :meth:`transform_plot`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.

        y : ndarray of shape (n_samples,) or None
            Target values for supervised problems.

        sample : int
            Sample to be plotted.

        **plot_params
            Optional plotting parameters.

        Returns
        -------
        Xt : ndarray of shape (1, ...)
            Transformed one-sample slice from the input.

        """
        self.fit(X, y)
        Xt = self.transform_plot(X, sample=sample, **plot_params)
        return Xt

    def transform_plot(self, X, sample=0, **plot_params):
        """Take a one-sample slice from the input collection and transform it.
        Before returning the transformed object, plot the transformed sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.

        sample : int
            Sample to be plotted.

        **plot_params
            Optional plotting parameters.

        Returns
        -------
        Xt : ndarray of shape (1, ...)
            Transformed one-sample slice from the input.

        """
        Xt = self.transform(X[sample:sample+1])
        self.plot({sample: Xt[0]}, sample=sample, **plot_params).show()
        return Xt
