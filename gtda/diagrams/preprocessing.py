"""Persistence diagram preprocessing."""
# License: GNU AGPLv3

from numbers import Real
from typing import Callable

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _AVAILABLE_AMPLITUDE_METRICS, _parallel_amplitude
from ._utils import _filter, _bin, _homology_dimensions_to_sorted_ints
from ..base import PlotterMixin
from ..plotting.persistence_diagrams import plot_diagram
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import check_diagrams, validate_params


@adapt_fit_transform_docs
class ForgetDimension(BaseEstimator, TransformerMixin, PlotterMixin):
    """Replaces all homology dimensions in persistence diagrams with
    ``numpy.inf``.

    Useful when downstream tasks require the use of topological features all at
    once -- and not separated between different homology dimensions.

    See also
    --------
    PairwiseDistance, Amplitude, Scaler, Filtering

    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_diagrams(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Replace all homology dimensions in `X` with ``numpy.inf``.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Output persistence diagram.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_diagrams(X, copy=True)

        Xt[:, :, 2] = np.inf
        # TODO: for plotting, replace the dimension with a tag
        return Xt

    @staticmethod
    def plot(Xt, sample=0, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

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
        return plot_diagram(
            Xt[sample], homology_dimensions=[np.inf],
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class Scaler(BaseEstimator, TransformerMixin, PlotterMixin):
    """Linear scaling of persistence diagrams.

    A positive scale factor :attr:`scale_` is calculated during :meth:`fit` by
    considering all available persistence diagrams partitioned according to
    homology dimensions. During :meth:`transform`, all birth-death pairs are
    divided by :attr:`scale_`.

    The value of :attr:`scale_` depends on two things:

        - A way of computing, for each homology dimension, the :ref:`amplitude
          <vectorization_amplitude_and_kernel>` in that dimension of a
          persistence diagram consisting of birth-death-dimension triples
          [b, d, q]. Together, `metric` and `metric_params` define this in the
          same way as in :class:`Amplitude`.
        - A scalar-valued function which is applied to the resulting
          two-dimensional array of amplitudes (one per diagram and homology
          dimension) to obtain :attr:`scale_`.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'betti'`` | \
        ``'landscape'`` |``'silhouette'`` |  ``'heat'`` | \
        ``'persistence_image'``, optional, default: ``'bottleneck'``
        See the corresponding parameter in :class:`Amplitude`.

    metric_params : dict or None, optional, default: ``None``
        See the corresponding parameter in :class:`Amplitude`.

    function : callable, optional, default: ``numpy.max``
        Function used to extract a positive scalar from the collection of
        amplitude vectors in :meth:`fit`. Must map 2D arrays to scalars.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params` as
        well as relevant quantities computed in :meth:`fit`.

    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    scale_ : float
        Value by which to rescale diagrams.

    See also
    --------
    PairwiseDistance, ForgetDimension, Filtering, Amplitude

    Notes
    -----
    When `metric` is ``'bottleneck'`` and `function` is ``numpy.max``,
    :meth:`fit_transform` has the effect of making the lifetime of the most
    persistent point across all diagrams and homology dimensions equal to 2.

    To compute scaling factors without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    """

    _hyperparameters = {
        'metric': {'type': str, 'in': _AVAILABLE_AMPLITUDE_METRICS.keys()},
        'metric_params': {'type': (dict, type(None))},
        'function': {'type': (Callable, type(None))}
        }

    def __init__(self, metric='bottleneck', metric_params=None,
                 function=np.max, n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.function = function
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and compute :attr:`scale_`.
        Then, return the estimator.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        validate_params(self.effective_metric_params_,
                        _AVAILABLE_AMPLITUDE_METRICS[self.metric])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)

        self.effective_metric_params_['samplings'], \
            self.effective_metric_params_['step_sizes'] = \
            _bin(X, self.metric, **self.effective_metric_params_)

        if self.metric == 'persistence_image':
            weight_function = self.effective_metric_params_.get(
                'weight_function', None
                )
            weight_function = \
                np.ones_like if weight_function is None else weight_function
            self.effective_metric_params_['weight_function'] = weight_function

        amplitude_array = _parallel_amplitude(X, self.metric,
                                              self.effective_metric_params_,
                                              self.homology_dimensions_,
                                              self.n_jobs)
        self.scale_ = self.function(amplitude_array)

        return self

    def transform(self, X, y=None):
        """Divide all birth and death values in `X` by :attr:`scale_`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xs : ndarray of shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(self)

        Xs = check_diagrams(X, copy=True)
        Xs[:, :, :2] /= self.scale_
        return Xs

    def inverse_transform(self, X):
        """Scale back the data to the original representation. Multiplies by
        the scale found in :meth:`fit`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Data to apply the inverse transform to, c.f. :meth:`transform`.

        Returns
        -------
        Xs : ndarray of shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(self)

        Xs = check_diagrams(X, copy=True)
        Xs[:, :, :2] *= self.scale_
        return Xs

    def plot(self, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` is
            equivalent to passing :attr:`homology_dimensions_`.

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
        if homology_dimensions is None:
            _homology_dimensions = self.homology_dimensions_
        else:
            _homology_dimensions = homology_dimensions

        return plot_diagram(
            Xt[sample], homology_dimensions=_homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class Filtering(BaseEstimator, TransformerMixin, PlotterMixin):
    """Filtering of persistence diagrams.

    Filtering a diagram means discarding all points [b, d, q] representing
    non-trivial topological features whose lifetime d - b is less than or equal
    to a cutoff value. Points on the diagonal (i.e. for which b and d are
    equal) may still appear in the output for padding purposes, but carry no
    information.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    homology_dimensions : list, tuple, or None, optional, default: ``None``
        When set to ``None``, subdiagrams corresponding to all homology
        dimensions seen in :meth:`fit` will be filtered. Otherwise, it contains
        the homology dimensions (as non-negative integers) at which filtering
        should occur.

    epsilon : float, optional, default: ``0.01``
        The cutoff value controlling the amount of filtering.

    Attributes
    ----------
    homology_dimensions_ : tuple
        If `homology_dimensions` is set to ``None``, contains the homology
        dimensions seen in :meth:`fit`, sorted in ascending order. Otherwise,
        it is a similarly sorted version of `homology_dimensions`.

    See also
    --------
    PairwiseDistance, ForgetDimension, Scaler, Amplitude

    """

    _hyperparameters = {
        'homology_dimensions': {
            'type': (list, tuple, type(None)),
            'of': {'type': int, 'in': Interval(0, np.inf, closed='left')}
            },
        'epsilon': {'type': Real, 'in': Interval(0, np.inf, closed='left')}
        }

    def __init__(self, homology_dimensions=None, epsilon=0.01):
        self.homology_dimensions = homology_dimensions
        self.epsilon = epsilon

    def fit(self, X, y=None):
        """Store relevant homology dimensions in
        :attr:`homology_dimensions_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters)

        if self.homology_dimensions is None:
            # Find the unique homology dimensions in the 3D array X passed to
            # `fit` assuming that they can all be found in its zero-th entry
            homology_dimensions = np.unique(X[0, :, 2])
        else:
            homology_dimensions = self.homology_dimensions
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions)

        return self

    def transform(self, X, y=None):
        """Filter all relevant persistence subdiagrams.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features_filtered, 3)
            Filtered persistence diagrams. Only the subdiagrams corresponding
            to dimensions in :attr:`homology_dimensions_` are filtered.
            ``n_features_filtered`` is less than or equal to ``n_features``.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        Xt = _filter(X, self.homology_dimensions_, self.epsilon)
        return Xt

    def plot(self, Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` is
            equivalent to passing :attr:`homology_dimensions_`.

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
        if homology_dimensions is None:
            _homology_dimensions = self.homology_dimensions_
        else:
            _homology_dimensions = homology_dimensions

        return plot_diagram(
            Xt[sample], homology_dimensions=_homology_dimensions,
            plotly_params=plotly_params
            )
