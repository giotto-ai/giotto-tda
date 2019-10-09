# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _parallel_pairwise, _parallel_amplitude
from ._utils import _discretize
from ..utils.validation import check_diagram, validate_metric_params


class DiagramDistance(BaseEstimator, TransformerMixin):
    """Transformer for calculating distances between collections of
    persistence diagrams.
    In the case in which diagrams in the collection have been
    consistently partitioned
    into one or more subdiagrams (e.g. according to homology dimension),
    the distance between any two diagrams is a *p*-norm of a vector of
    distances between respective subdiagrams of the same kind.

    Parameters
    ----------
    metric : 'bottleneck' | 'wasserstein' | 'landscape' | 'betti' | 'heat',
    optional, default: 'bottleneck'
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to a family of possible (:math:`L^p`-like)
           distances between "persistence landscapes" obtained from persistence
           (sub)diagrams.
        - ``'betti'`` refers to a family of possible (:math:`L^p`-like)
           distances between "Betti curves" obtained from persistence
           (sub)diagrams. A Betti curve simply records the evolution in the
           number of independent topological holes (technically, the number
           of linearly independent homology classes) as can be read from a
           persistence (sub)diagram.
        - ``'heat'`` refers to a family of possible (:math:`L^p`-like)
           distances between "Heat kernels"obtained from persistence
           (sub)diagrams.

    metric_params : dict, optional, default: {'n_values': 100}
        Additional keyword arguments for the metric function:

        - If ``metric == 'bottleneck'`` the only argument is
          ``delta`` (default = ``0.0``).
        - If ``metric == 'wasserstein'`` the available arguments are ``p``
          (default = ``1``) and ``delta`` (default = ``0.0``).
        - If ``metric == 'landscape'`` the available arguments are ``p``
          (default = ``2``), ``n_values`` (default = ``100``) and
          ``n_layers`` (default = ``1``).
        - If ``metric == 'betti'`` the available arguments are ``p``
          (default = ``2``) and ``n_values`` (default = ``100``).
        - If ``metric == 'heat'`` the available arguments are ``p``
          (default = ``2``), ``sigma`` (default = ``1``) and
          ``n_values`` (default = ``100``).

    order : int, optional, default: 2
        Order of the norm used to combine subdiagrams distances into a single
        distance. If set to ``None``, returns one distance matrix per homology
        dimension.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """
    def __init__(self, metric='landscape', metric_params=None, order=2,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the estimator and return it.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which k equals ``np.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        validate_metric_params(self.metric, self.effective_metric_params_)
        X = check_diagram(X)

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)
            if self.metric == 'landscape':
                self.effective_metric_params_['samplings'] = {
                    dim: np.sqrt(2) * sampling for dim, sampling in
                    self.effective_metric_params_['samplings'].items()}
                self.effective_metric_params_['step_sizes'] = {
                    dim: np.sqrt(2) * step_size for dim, step_size in
                    self.effective_metric_params_['step_sizes'].items()}

        self._X = X
        return self

    def transform(self, X, y=None):
        """Computes the distance matrix between the diagrams in X, according to
        the choice of ``metric`` and ``metric_params``.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which k equals ``np.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_samples) if ``order`` is ``None``
        (n_samples, n_samples, n_dimensions) else.
            Distance matrix between diagrams in X.

        """
        check_is_fitted(self, 'effective_metric_params_')
        X = check_diagram(X)

        if np.array_equal(X, self._X):
            X2 = None
        else:
            X2 = X

        Xt = _parallel_pairwise(self._X, X2, self.metric,
                                self.effective_metric_params_,
                                n_jobs=self.n_jobs)
        if self.order is not None:
            Xt = np.linalg.norm(Xt, axis=2, ord=self.order)

        return Xt


class DiagramAmplitude(BaseEstimator, TransformerMixin):
    """Transformer for calculating the amplitude of a collections of
    persistence diagrams. In the case in which diagrams in the collection
    have been consistently partitioned into one or more subdiagrams (e.g.
    according to homology dimension), the amplitude of a diagram is a
    *p*-norm of a vector of distances between respective subdiagrams of
    the same kind.

    Parameters
    ----------
    metric : 'bottleneck' | 'wasserstein' | 'landscape' | 'betti', optional,
        default: 'bottleneck'
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to a family of possible (:math:`L^p`-like)
          distances between "persistence landscapes" obtained from
          persistence (sub)diagrams.
        - ``'betti'`` refers to a family of possible (:math:`L^p`-like)
          distances between "Betti curves" obtained from persistence
          (sub)diagrams. A Betti curve simply records the evolution in the
          number of independent topological holes (technically, the number
          of linearly independent homology classes) as can be read from a
          persistence (sub)diagram.
        - ``'heat'`` refers to the heat kernel

    metric_params : dict, optional, default: {'n_values': 100}
        Additional keyword arguments for the metric function:

        - If ``metric == 'bottleneck'`` the available arguments are ``order``
          (default = ``np.inf``) and ``delta`` (default = ``0.0``).
        - If ``metric == 'wasserstein'`` the only argument is ``order``
          (default = ``1``) and ``delta`` (default = ``0.0``).
        - If ``metric == 'landscape'`` the available arguments are ``order``
          (default = ``2``), ``n_values`` (default = ``100``) and
          ``n_layers`` (default = ``1``).
        - If ``metric == 'betti'`` the available arguments are ``order``
          (default = ``2``) and ``n_values`` (default = ``100``).
        - If ``metric == 'heat'`` the available arguments are ``order``
          (default = ``2``), ``sigma`` (default = ``1``) and
          ``n_values`` (default = ``100``).

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """
    def __init__(self, metric='landscape', metric_params=None, order=2,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the estimator and return it.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``np.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()

        validate_metric_params(self.metric, self.effective_metric_params_)
        X = check_diagram(X)

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)
            if self.metric == 'landscape':
                self.effective_metric_params_['samplings'] = {
                    dim: np.sqrt(2) * sampling for dim, sampling in
                    self.effective_metric_params_['samplings'].items()}
                self.effective_metric_params_['step_sizes'] = {
                    dim: np.sqrt(2) * step_size for dim, step_size in
                    self.effective_metric_params_['step_sizes'].items()}

        return self

    def transform(self, X, y=None):
        """Computes the amplitude of a each diagram in the collection X,
        according to the choice of ``metric`` and ``metric_params``.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``np.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, 1) if ``order`` is ``None``
        (n_samples, n_samples, n_dimensions) else
            Amplitude of the diagrams in X.

        """
        check_is_fitted(self, ['effective_metric_params_'])
        X = check_diagram(X)

        Xt = _parallel_amplitude(X, self.metric,
                                 self.effective_metric_params_,
                                 n_jobs=self.n_jobs)
        if self.order is None:
            return Xt
        return np.linalg.norm(Xt, axis=1, ord=self.order)
