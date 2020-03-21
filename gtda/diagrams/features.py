"""Feature extraction from persistence diagrams."""
# License: GNU AGPLv3

from numbers import Real

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted

from ._metrics import _AVAILABLE_AMPLITUDE_METRICS, _parallel_amplitude
from ._utils import _subdiagrams, _bin, _calculate_weights
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params, check_diagrams


@adapt_fit_transform_docs
class PersistenceEntropy(BaseEstimator, TransformerMixin):
    """:ref:`Persistence entropies <persistence_entropy>` of persistence
    diagrams.

    Given a persistence diagrams consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and their respective persistence entropies are
    calculated as the (base e) entropies of the collections of differences
    d - b, normalized by the sum of all such differences.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    BettiCurve, PersistenceLandscape, HeatKernel, Amplitude, \
    PersistenceImage, PairwiseDistance, Silhouette, \
    gtda.homology.VietorisRipsPersistence

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _persistence_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape(-1, 1)
        return - np.sum(np.nan_to_num(
            X_normalized * np.log(X_normalized)), axis=1).reshape(-1, 1)

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
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
        X = check_diagrams(X)

        self.homology_dimensions_ = sorted(set(X[0, :, 2]))
        self._n_dimensions = len(self.homology_dimensions_)

        return self

    def transform(self, X, y=None):
        """Compute the persistence entropies of diagrams in `X`.

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
        Xt : ndarray of shape (n_samples, n_homology_dimensions)
            Persistence entropies: one value per sample and per homology
            dimension seen in :meth:`fit`. Index i along axis 1 corresponds
            to the i-th homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        with np.errstate(divide='ignore', invalid='ignore'):
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(self._persistence_entropy)(_subdiagrams(X, [dim])[s])
                for dim in self.homology_dimensions_
                for s in gen_even_slices(
                    X.shape[0], effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt).reshape(self._n_dimensions, X.shape[0]).T
        return Xt


@adapt_fit_transform_docs
class Amplitude(BaseEstimator, TransformerMixin):
    """:ref:`Amplitudes <amplitude>` of persistence diagrams.

    For each persistence diagram in a collection, a vector of amplitudes or a
    single scalar amplitude is calculated according to the following steps:

        1. The diagram is partitioned into subdiagrams according to homology
           dimension.
        2. The amplitude of each subdiagram is calculated according to the
           parameters `metric` and `metric_params`. This gives a vector of
           amplitudes, :math:`\\mathbf{a} = (a_{q_1}, \\ldots, a_{q_n})` where
           the :math:`q_i` range over the available homology dimensions.
        3. The final result is either :math:`\\mathbf{a}` itself or
           a norm of :math:`\\mathbf{a}`, specified by the parameter `order`.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'`` | ``'silhouette'`` | \
        ``'persistence_image'``, optional, default: ``'landscape'``
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the (trivial) diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.
        - ``'silhouette'`` refers to the :math:`L^p` distance between
          silhouettes.
        - ``'persistence_image'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams represented on birth-persistence axes.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function (passing
        ``None`` is equivalent to passing the defaults described below):

        - If ``metric == 'bottleneck'`` there are no available arguments.
        - If ``metric == 'wasserstein'`` the only argument is `p` (float,
          default: ``2.``).
        - If ``metric == 'landscape'`` the available arguments are `p`
          (float, default: ``2.``), `n_bins` (int, default: ``100``) and
          `n_layers` (int, default: ``1``).
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_bins` (int, default: ``100``).
        - If ``metric == 'heat'`` the available arguments are `p` (float,
          default: ``2.``), `sigma` (float, default: ``1.``) and `n_bins`
          (int, default: ``100``).
        - If ``metric == 'silhouette'`` the available arguments are `p`
          (float, default: ``2.``), `order` (float, default: ``1.``) and
          `n_bins` (int, default: ``100``).
        - If ``metric == 'persistence_image'`` the available arguments are `p`
          (float, default: ``2.``), `sigma` (float, default: ``1.``),
          `n_bins` (int, default: ``100``) and `weight_function`
          (callable or None, default: ``None``).

    order : float or None, optional, default: ``2.``
        If ``None``, :meth:`transform` returns for each diagram a vector of
        amplitudes corresponding to the dimensions in
        :attr:`homology_dimensions_`. Otherwise, the :math:`p`-norm of
        these vectors with :math:`p` equal to `order` is taken.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params` as
        well as on any relevant quantities computed in :meth:`fit`.

    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    PairwiseDistance, Scaler, Filtering, \
    BettiCurve, PersistenceLandscape, \
    HeatKernel, Silhouette, \
    gtda.homology.VietorisRipsPersistence

    Notes
    -----
    To compute amplitudes without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    """

    _hyperparameters = {
        'metric': {'type': str, 'in': _AVAILABLE_AMPLITUDE_METRICS.keys()},
        'order': {'type': (Real, type(None)),
                  'in': Interval(0, np.inf, closed='right')},
        'metric_params': {'type': (dict, type(None))}}

    def __init__(self, metric='landscape', metric_params=None, order=2.,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and compute
        :attr:`effective_metric_params`. Then, return the estimator.

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
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        validate_params(self.effective_metric_params_,
                        _AVAILABLE_AMPLITUDE_METRICS[self.metric])

        self.homology_dimensions_ = sorted(set(X[0, :, 2]))

        self.effective_metric_params_['samplings'], \
            self.effective_metric_params_['step_sizes'] = \
            _bin(X, metric=self.metric, **self.effective_metric_params_)

        if self.metric == 'persistence_image':
            self.effective_metric_params_['weights'] = \
                _calculate_weights(X, **self.effective_metric_params_)

        return self

    def transform(self, X, y=None):
        """Compute the amplitudes or amplitude vectors of diagrams in `X`.

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
        Xt : ndarray of shape (n_samples, n_homology_dimensions) if `order` \
            is ``None``, else (n_samples, 1)
            Amplitudes or amplitude vectors of the diagrams in `X`. In the
            second case, index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        Xt = check_diagrams(X, copy=True)

        Xt = _parallel_amplitude(Xt, self.metric,
                                 self.effective_metric_params_,
                                 self.homology_dimensions_,
                                 self.n_jobs)
        if self.order is None:
            return Xt
        Xt = np.linalg.norm(Xt, axis=1, ord=self.order).reshape(-1, 1)
        return Xt
