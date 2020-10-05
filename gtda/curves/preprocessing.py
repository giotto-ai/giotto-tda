"""Preprocessing transformers for curves."""
# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from plotly.graph_objs import Figure, Scatter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted, check_array

from ..base import PlotterMixin
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class Derivative(BaseEstimator, TransformerMixin, PlotterMixin):
    """Derivatives of multi-channel curves.

    A multi-channel (integer sampled) curve is a 2D array of shape
    ``(n_channels, n_bins)``, where each row represents the y-values in one of
    the channels. This transformer computes the n-th order derivative of each
    channel in each multi-channel curve in a collection, by discrete
    differences. The output is another collection of multi-channel curves.

    Parameters
    ----------
    order : int, optional, default: ``1``
        Order of the derivative to be taken.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    n_channels_ : int
        Number of channels present in the 3D array passed to :meth:`fit`.

    """
    _hyperparameters = {
        'order': {'type': int, 'in': Interval(1, np.inf, closed='left')},
    }

    def __init__(self, order=1, n_jobs=None):
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Compute :attr:`n_channels_`. Then, return the estimator.

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
        check_array(X, ensure_2d=False, allow_nd=True)
        if X.ndim != 3:
            raise ValueError("Input must be 3-dimensional.")
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        n_bins = X.shape[2]
        if self.order >= n_bins:
            raise ValueError(
                f"Input channels have length {n_bins} but they must have at "
                f"least length {self.order + 1} to calculate derivatives of "
                f"order {self.order}."
                )

        self.n_channels_ = X.shape[1]

        return self

    def transform(self, X, y=None):
        """Compute derivatives of multi-channel curves.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_channels, n_bins - order)
            Output collection of multi-channel curves given by taking discrete
            differences of order `order` in each channel in the curves in `X`.

        """
        check_is_fitted(self)
        Xt = check_array(X, ensure_2d=False, allow_nd=True)
        if Xt.ndim != 3:
            raise ValueError("Input must be 3-dimensional.")

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(np.diff)(Xt[s], n=self.order, axis=-1)
            for s in gen_even_slices(len(Xt), effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt)

        return Xt

    def plot(self, Xt, sample=0, channels=None, plotly_params=None):
        """Plot a sample from a collection of derivatives of multi-channel
        curves arranged as in the output of :meth:`transform`.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_channels, n_bins)
            Collection of multi-channel curves, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        channels : list, tuple or None, optional, default: ``None``
            Which channels to include in the plot. ``None`` means plotting the
            first :attr:`n_channels_` channels.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        check_is_fitted(self)

        layout_axes_common = {
            "type": "linear",
            "ticks": "outside",
            "showline": True,
            "zeroline": True,
            "linewidth": 1,
            "linecolor": "black",
            "mirror": False,
            "showexponent": "all",
            "exponentformat": "e"
            }
        layout = {
            "xaxis1": {
                "title": "Sample",
                "side": "bottom",
                "anchor": "y1",
                **layout_axes_common
                },
            "yaxis1": {
                "title": "Derivative",
                "side": "left",
                "anchor": "x1",
                **layout_axes_common
                },
            "plot_bgcolor": "white",
            "title": f"Derivative of sample {sample}"
            }

        fig = Figure(layout=layout)

        if channels is None:
            channels = range(self.n_channels_)

        samplings = np.arange(Xt[sample].shape[0])
        for ix, channel in enumerate(channels):
            fig.add_trace(Scatter(x=samplings,
                                  y=Xt[sample][ix],
                                  mode="lines",
                                  showlegend=True,
                                  name=f"Channel {channel}"))

        # Update traces and layout according to user input
        if plotly_params:
            fig.update_traces(plotly_params.get("traces", None))
            fig.update_layout(plotly_params.get("layout", None))

        return fig
