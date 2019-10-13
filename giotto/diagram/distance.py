# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _parallel_pairwise, _parallel_amplitude
from ._utils import _discretize
from ..utils.validation import check_diagram, validate_metric_params


class DiagramDistance(BaseEstimator, TransformerMixin):
    """`Distances <LINK TO GLOSSARY>`_ between pairs of persistence diagrams,
    constructed from the distances between their respective subdiagrams with
    constant homology dimension.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, default: ``'bottleneck'``
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

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function:

        - If ``metric == 'bottleneck'`` the only argument is `delta` (float,
          default: `0.01`).
        - If ``metric == 'wasserstein'`` the available arguments are `p`
          (int, default: ``2``) and `delta` (float, default: ``0.``).
        - If ``metric == 'landscape'`` the available arguments are `p`
          (float, default: ``2.``), `n_values` (int, default: ``100``) and
          `n_layers` (int, default: ``1``).
        - If ``metric == 'betti'`` the available arguments are `p`
          (float, default: ``2.``) and `n_values` (int, default: ``100``).
        - If ``metric == 'heat'`` the available arguments are `p`
          (float, default: ``2.``), `sigma` (float, default: ``1.``) and
          `n_values` (int, default: ``100``).

    order : float, optional, default: ``2.``
        Order of the norm used to combine subdiagrams distances into a single
        distance. If set to ``None``, returns one distance matrix per homology
        dimension.

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
    DiagramAmplitude, BettiCurve, PersistenceLandscape, HeatKernel, \
    giotto.homology.VietorisRipsPersistence

    Notes
    -----
    To compute distances without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetHomologyDimensions`.

    `Hera <https://bitbucket.org/grey_narn/hera>`_ is used as a C++ backend
    for computing bottleneck and Wasserstein distances between persistence
    diagrams.

    """
    def __init__(self, metric='landscape', metric_params=None, order=2.,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the estimator and return it.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

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
        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)

        self._X = X
        return self

    def transform(self, X, y=None):
        """Computes the distance matrix between the diagrams in `X`, according to
        the choice of `metric` and `metric_params`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_samples) if `order` is ``None``, \
            else (n_samples, n_samples, n_dimensions).
            Distance matrix between diagrams in `X`.

        """
        check_is_fitted(self, ['effective_metric_params_',
                               'homology_dimensions_'])
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


class DiagramAmplitude(BaseEstimator, TransformerMixin):
    """`Amplitudes <LINK TO GLOSSARY>`_ of persistence diagrams, constructed
    from the amplitudes of their subdiagrams with constant homology dimension.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], a vector of amplitudes or a single scalar amplitude is
    calculated according to the following steps:

        1. Subdiagrams corresponding to distinct homology dimensions are
           extracted.
        2. The amplitude of each subdiagram is calculated according to the
           parameters `metric` and `metric_params`. The result is a vector of
           amplitudes, :math:`\\mathbf{a} = (a_{q_1}, \\ldots, a_{q_n})`.
        3. The final result  is either :math:`\\mathbf{a}` itself or
           the `order`-`norm <GLOSSARY>`_ of :math:`\\mathbf{a}`.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'`` | ``'heat'``, optional, default: ``'bottleneck'``

        Distance function used to define the amplitude of subdiagrams as their
        distance from the trivial (i.e. diagonal) diagram:

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
        If ``None``, :meth:`transform` returns for each persistence diagram
        a vector of amplitudes corresponding to the homology dimensions in
        :attr:`homology_dimensions_`. Otherwise, the vector `order`-norm of
        these vectors is taken.

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
    DiagramDistance, Scaler, Filtering, \
    BettiCurve, PersistenceLandscape, \
    HeatKernel, giotto.homology.VietorisRipsPersistence

    Notes
    -----
    To compute amplitudes without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetHomologyDimensions`.

    """
    def __init__(self, metric='landscape', metric_params=None, order=2.,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit the estimator and return it.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

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
        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))

        if self.metric in ['landscape', 'heat', 'betti']:
            self.effective_metric_params_['samplings'], \
                self.effective_metric_params_['step_sizes'] = \
                _discretize(X, **self.effective_metric_params_)

        return self

    def transform(self, X, y=None):
        """Compute the amplitudes or amplitude vectors of diagrams in `X`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_homology_dimensions) if `order` is \
             ``None``, else (n_samples, 1)
            Amplitudes or amplitude vectors of the diagrams in `X`. In the
            latter case, index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self, ['effective_metric_params_',
                               'homology_dimensions_'])
        X = check_diagram(X)

        Xt = _parallel_amplitude(X, self.metric,
                                 self.effective_metric_params_,
                                 self.homology_dimensions_,
                                 self.n_jobs)
        if self.order is None:
            return Xt
        Xt = np.linalg.norm(Xt, axis=1, ord=self.order).reshape((-1, 1))
        return Xt
