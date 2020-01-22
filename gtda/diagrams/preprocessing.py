"""Persistence diagram preprocessing."""
# License: GNU AGPLv3

import numbers
import types

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _parallel_amplitude
from ._utils import _sort, _filter, _discretize
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import (check_diagram, validate_params,
                                validate_metric_params)


@adapt_fit_transform_docs
class ForgetDimension(BaseEstimator, TransformerMixin):
    """Replaces all homology dimensions in persistence diagrams with
    ``numpy.inf``.

    Useful when downstream tasks require the use of topological features all at
    once -- and not separated between different homology dimensions.

    See also
    --------
    gtda.homology.VietorisRipsPersistence

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
        X = check_diagram(X)

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
        # Check if fit had been called
        check_is_fitted(self, '_is_fitted')
        X = check_diagram(X)

        Xt = X.copy()
        Xt[:, :, 2] = np.inf
        return Xt


@adapt_fit_transform_docs
class Scaler(BaseEstimator, TransformerMixin):
    """Linear scaling of persistence diagrams.

    A positive scale factor is calculated during :meth:`fit` by considering all
    available persistence diagrams and homology dimensions. During
    :meth:`transform`, all birth-death pairs are divided by this factor.

    The value of the scale factor depends on two things:

        - A way of computing, for each homology dimension, the `amplitude
          <https://giotto.ai/theory>`_ in that dimension of a persistence
          diagram consisting of birth-death-dimension triples [b, d, q].
          Together, `metric` and `metric_params` define this in the same way as
          in :class:`Amplitude`.
        - A scalar-valued function which is applied to the resulting
          two-dimensional array of amplitudes.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, default: ``'bottleneck'``
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function:

        - If ``metric == 'bottleneck'`` there are no available arguments.
        - If ``metric == 'wasserstein'`` the only argument is `p` (int,
          default: ``2``).
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_values` (int, default: ``100``).
        - If ``metric == 'landscape'`` the available arguments are `p`
          (float, default: ``2.``), `n_values` (int, default: ``100``) and
          `n_layers` (int, default: ``1``).
        - If ``metric == 'heat'`` the available arguments are `p` (float,
          default: ``2.``), `sigma` (float, default: ``1.``) and `n_values`
          (int, default: ``100``).

    function : callable, optional, default: ``numpy.max``
        Function used to extract a positive scalar from the collection of
        amplitude vectors in :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params` as
        well as on any relevant quantities computed in :meth:`fit`.

    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    scale_ : float
        Value by which to rescale diagrams.

    See also
    --------
    Filtering, Amplitude, PairwiseDistance, \
    gtda.homology.VietorisRipsPersistence

    Notes
    -----
    To compute scaling factors without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    """

    _hyperparameters = {'function': [types.FunctionType]}

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

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(self.get_params(), self._hyperparameters)

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        validate_metric_params(self.metric, self.effective_metric_params_)
        X = check_diagram(X)
        self.homology_dimensions_ = sorted(set(X[0, :, 2]))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)

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

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xs : ndarray of shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(self)

        Xs = check_diagram(X)
        Xs[:, :, :2] /= self.scale_
        return Xs

    def inverse_transform(self, X):
        """Scale back the data to the original representation. Multiplies
        by the scale found in :meth:`fit`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Data to apply the inverse transform to.

        Returns
        -------
        Xs : ndarray of shape (n_samples, n_features, 3)
            Rescaled diagrams.

        """
        check_is_fitted(self)

        Xs = check_diagram(X)
        Xs[:, :, :2] *= self.scale_
        return Xs


@adapt_fit_transform_docs
class Filtering(BaseEstimator, TransformerMixin):
    """Filtering of persistence diagrams.

    Filtering a diagram means removing all points whose distance from the
    diagonal is less than or equal to a certain cutoff value which can be
    interpreted as (:math:`1/\\sqrt{2}` times) the "minimum amount of
    persistence" required from points in the filtered diagram.

    Parameters
    ----------
    homology_dimensions : iterable or None, optional, default: ``None``
        When set to ``None``, subdiagrams corresponding to all homology
        dimensions seen in :meth:`fit` will be filtered.
        Otherwise, it contains the homology dimensions at which filtering
        should occur.

    epsilon : float, optional, default: ``0.01``
        The cutoff value controlling the amount of filtering.

    Attributes
    ----------
    homology_dimensions_ : list
        If `homology_dimensions` is set to ``None``, then this is the list
        of homology dimensions seen in :meth:`fit`, sorted in ascending
        order. Otherwise, it is a similarly sorted version of
        `homology_dimensions`.

    See also
    --------
    Scaling, Amplitude, PairwiseDistance, \
    gtda.homology.VietorisRipsPersistence

    """

    _hyperparameters = {'homology_dimensions_': [list, [int, (0, np.inf)]],
                        'epsilon': [numbers.Number, (0., np.inf)]}

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

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)

        if self.homology_dimensions is None:
            self.homology_dimensions_ = [int(dim) for dim in set(X[0, :, 2])]
        else:
            self.homology_dimensions_ = self.homology_dimensions
        self.homology_dimensions_ = sorted(self.homology_dimensions_)

        validate_params({**self.get_params(),
                         'homology_dimensions_': self.homology_dimensions_},
                        self._hyperparameters)

        self.homology_dimensions_ = \
            [float(dim) for dim in self.homology_dimensions_]

        return self

    def transform(self, X, y=None):
        """Filter all relevant persistence subdiagrams.

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
            Filtered persistence diagrams. Only the subdiagrams corresponding
            to dimensions in :attr:`homology_dimensions_` are filtered.
            Discarded points are replaced by points on the diagonal.

        """
        # Check if fit had been called
        check_is_fitted(self)
        X = check_diagram(X)

        X = _sort(X)
        Xt = _filter(X, self.homology_dimensions_, self.epsilon)
        return Xt
