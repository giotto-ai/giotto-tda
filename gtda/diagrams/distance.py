"""Distance and amplitude calculations for persistence diagrams."""
# License: GNU AGPLv3

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _parallel_pairwise, _parallel_amplitude
from ._utils import _discretize
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import (check_diagram, validate_params,
                                validate_metric_params)


@adapt_fit_transform_docs
class PairwiseDistance(BaseEstimator, TransformerMixin):
    """`Distances <https://giotto.ai/theory>`_ between pairs of persistence
    diagrams, constructed from the distances between their respective
    subdiagrams with constant homology dimension.

    Given two collections of persistence diagrams consisting of
    birth-death-dimension triples [b, d, q], a collection of distance
    matrices or a single distance matrix between pairs of diagrams is
    calculated according to the following steps:

        1. All diagrams are partitioned into subdiagrams corresponding to
           distinct homology dimensions.
        2. Pairwise distances between subdiagrams of equal homology
           dimension are calculated according to the parameters `metric` and
           `metric_params`. This gives a collection of distance matrices,
           :math:`\\mathbf{D} = (D_{q_1}, \\ldots, D_{q_n})`.
        3. The final result is either :math:`\\mathbf{D}` itself as a
           three-dimensional array, or a single distance matrix constructed
           by taking norms of the vectors of distances between diagram pairs.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, default: ``'landscape'``
        Distance or dissimilarity function between subdiagrams:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function:

        - If ``metric == 'bottleneck'`` the only argument is `delta` (float,
          default: ``0.01``). When equal to ``0.``, an exact algorithm is
          used; otherwise, a faster approximate algorithm is used.
        - If ``metric == 'wasserstein'`` the available arguments are `p`
          (int, default: ``2``) and `delta` (float, default: ``0.01``).
          Unlike the case of ``'bottleneck'``, `delta` cannot be set to
          ``0.`` and an exact algorithm is not available.
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_values` (int, default: ``100``).
        - If ``metric == 'landscape'`` the available arguments are `p`
          (float, default: ``2.``), `n_values` (int, default: ``100``) and
          `n_layers` (int, default: ``1``).
        - If ``metric == 'heat'`` the available arguments are `p`
          (float, default: ``2.``), `sigma` (float, default: ``1.``) and
          `n_values` (int, default: ``100``).

    order : float or None, optional, default: ``2.``
        If ``None``, :meth:`transform` returns for each pair of diagrams a
        vector of distances corresponding to the dimensions in
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
    Amplitude, BettiCurve, PersistenceLandscape, HeatKernel, \
    gtda.homology.VietorisRipsPersistence

    Notes
    -----
    To compute distances without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    `Hera <https://bitbucket.org/grey_narn/hera>`_ is used as a C++ backend
    for computing bottleneck and Wasserstein distances between persistence
    diagrams. Python bindings were modified for performance from the
    `Dyonisus 2 <https://mrzv.org/software/dionysus2/>`_ package.

    """

    _hyperparameters = {'order': [float, (1, np.inf)]}

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
        X : ndarray of shape (n_samples_fit, n_features, 3)
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
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        hyperparameters = self.get_params().copy()
        if self.order is not None:
            if isinstance(self.order, int):
                hyperparameters['order'] = float(self.order)
        else:
            hyperparameters['order'] = 1.  # Automatically pass validate_params

        validate_params(hyperparameters, self._hyperparameters)
        validate_metric_params(self.metric, self.effective_metric_params_)

        self.homology_dimensions_ = sorted(set(X[0, :, 2]))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)

        self._X = X
        return self

    def transform(self, X, y=None):
        """Computes a distance or vector of distances between the diagrams in
        `X` and the diagrams seen in :meth:`fit`.

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
        Xt : ndarray of shape (n_samples_fit, n_samples, \
             n_homology_dimensions) if `order` is ``None``, else \
             (n_samples_fit, n_samples)
            Distance matrix or collection of distance matrices between
            diagrams in `X` and diagrams seen in :meth:`fit`. In the
            second case, index i along axis 2 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagram(X)

        if np.array_equal(X, self._X):
            X2 = None
        else:
            X2 = X

        Xt = _parallel_pairwise(self._X, X2, self.metric,
                                self.effective_metric_params_,
                                self.homology_dimensions_,
                                self.n_jobs)
        if self.order is not None:
            Xt = np.linalg.norm(Xt, axis=2, ord=self.order)

        return Xt


@adapt_fit_transform_docs
class Amplitude(BaseEstimator, TransformerMixin):
    """`Amplitudes <https://giotto.ai/theory>`_ of persistence diagrams,
    constructed from the amplitudes of their subdiagrams with constant
    homology dimension.

    Given a single persistence diagram consisting of birth-death-dimension
    triples [b, d, q], a vector of amplitudes or a single scalar amplitude is
    calculated according to the following steps:

        1. All diagrams are partitioned into subdiagrams corresponding to
           distinct homology dimensions.
        2. The amplitude of each subdiagram is calculated according to the
           parameters `metric` and `metric_params`. This gives a vector of
           amplitudes, :math:`\\mathbf{a} = (a_{q_1}, \\ldots, a_{q_n})`.
        3. The final result is either :math:`\\mathbf{a}` itself or
           a norm of :math:`\\mathbf{a}`.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, default: ``'landscape'``
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
    HeatKernel, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    To compute amplitudes without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    """

    _hyperparameters = {'order': [float, (1, np.inf)]}

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
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        hyperparameters = self.get_params().copy()
        if self.order is not None:
            if isinstance(self.order, int):
                hyperparameters['order'] = float(self.order)
        else:
            hyperparameters['order'] = 1.  # Automatically pass validate_params

        validate_params(hyperparameters, self._hyperparameters)
        validate_metric_params(self.metric, self.effective_metric_params_)
        X = check_diagram(X)
        self.homology_dimensions_ = sorted(set(X[0, :, 2]))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)

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
        X = check_diagram(X)

        Xt = _parallel_amplitude(X, self.metric,
                                 self.effective_metric_params_,
                                 self.homology_dimensions_,
                                 self.n_jobs)
        if self.order is None:
            return Xt
        Xt = np.linalg.norm(Xt, axis=1, ord=self.order).reshape(-1, 1)
        return Xt
