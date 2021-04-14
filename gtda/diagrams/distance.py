"""Pairwise distance calculations for persistence diagrams."""
# License: GNU AGPLv3

from numbers import Real

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _AVAILABLE_METRICS, _parallel_pairwise
from ._utils import _bin, _homology_dimensions_to_sorted_ints
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import check_diagrams, validate_params


@adapt_fit_transform_docs
class PairwiseDistance(BaseEstimator, TransformerMixin):
    """:ref:`Distances <wasserstein_and_bottleneck_distance>` between pairs
    of persistence diagrams.

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

    **Important notes**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.
        - The shape of outputs of :meth:`transform` depends on the value of the
          `order` parameter.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'betti'`` | \
        ``'landscape'`` | ``'silhouette'`` | ``'heat'`` | \
        ``'persistence_image'``, optional, default: ``'landscape'``
        Distance or dissimilarity function between subdiagrams:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'silhouette'`` refers to the :math:`L^p` distance between
          silhouettes.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.
        - ``'persistence_image'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams represented on birth-persistence axes.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function (passing
        ``None`` is equivalent to passing the defaults described below):

        - If ``metric == 'bottleneck'`` the only argument is `delta` (float,
          default: ``0.01``). When equal to ``0.``, an exact algorithm is used;
          otherwise, a faster approximate algorithm is used and symmetry is not
          guaranteed.
        - If ``metric == 'wasserstein'`` the available arguments are `p`
          (float, default: ``2.``) and `delta` (float, default: ``0.01``).
          Unlike the case of ``'bottleneck'``, `delta` cannot be set to ``0.``
          and an exact algorithm is not available.
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_bins` (int, default: ``100``).
        - If ``metric == 'landscape'`` the available arguments are `p` (float,
          default: ``2.``), `n_bins` (int, default: ``100``) and `n_layers`
          (int, default: ``1``).
        - If ``metric == 'silhouette'`` the available arguments are `p` (float,
          default: ``2.``), `power` (float, default: ``1.``) and `n_bins` (int,
          default: ``100``).
        - If ``metric == 'heat'`` the available arguments are `p` (float,
          default: ``2.``), `sigma` (float, default: ``0.1``) and `n_bins`
          (int, default: ``100``).
        - If ``metric == 'persistence_image'`` the available arguments are `p`
          (float, default: ``2.``), `sigma` (float, default: ``0.1``), `n_bins`
          (int, default: ``100``) and `weight_function` (callable or None,
          default: ``None``).

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
        well as relevant quantities computed in :meth:`fit`.

    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    Amplitude, Scaler, Filtering, BettiCurve, PersistenceLandscape, \
    PersistenceImage, HeatKernel, Silhouette, \
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

    _hyperparameters = {
        'metric': {'type': str, 'in': _AVAILABLE_METRICS.keys()},
        'order': {'type': (Real, type(None)),
                  'in': Interval(0, np.inf, closed='right')},
        'metric_params': {'type': (dict, type(None))}
        }

    def __init__(self, metric='landscape', metric_params=None, order=2.,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and compute
        :attr:`effective_metric_params_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples_fit, n_features, 3)
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
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        validate_params(
            self.effective_metric_params_, _AVAILABLE_METRICS[self.metric])

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
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_samples_fit, \
            n_homology_dimensions) if `order` is ``None``, else \
            (n_samples, n_samples_fit)
            Distance matrix or collection of distance matrices between
            diagrams in `X` and diagrams seen in :meth:`fit`. In the
            second case, index i along axis 2 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        Xt = check_diagrams(X, copy=True)

        Xt = _parallel_pairwise(Xt, self._X, self.metric,
                                self.effective_metric_params_,
                                self.homology_dimensions_,
                                self.n_jobs)
        if self.order is not None:
            Xt = np.linalg.norm(Xt, axis=2, ord=self.order)

        return Xt
