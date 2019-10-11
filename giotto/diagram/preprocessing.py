# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._metrics import _parallel_amplitude
from ._utils import _sort, _filter, _discretize
from ..utils.validation import check_diagram, validate_metric_params


class Stacking(BaseEstimator, TransformerMixin):
    """Transformer for stacking persistence subdiagrams.

    Useful when topological
    persistence information per sample has been previously separated according
    to some criterion (e.g. by homology dimension if produced by an instance of
    ```VietorisRipsPersistence``).

    """

    def __init__(self):
        pass

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.

        """
        pass

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``numpy.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        self._validate_params()
        X = check_diagram(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Stacks all available persistence subdiagrams corresponding
        to each sample into one persistence diagram.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``numpy.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : dict of None: ndarray
            Dictionary with a single ``None`` key, and corresponding value an
            ndarray of shape (n_samples, :math:`\\sum_{\\mathrm{d}}` M_d, 2).

        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_diagram(X)

        Xt = X.copy()
        Xt[:, :, 2] = np.inf
        return Xt


class Scaler(BaseEstimator, TransformerMixin):
    """Transformer scaling collections of persistence diagrams in which each
    diagram is partitioned into one or more subdiagrams (e.g. according to
    homology dimension).

    A scale factor is calculated during `fit` which depends on the entire
    collection, and it is applied during `transform`. The value of the scale
    factor depends on a chosen norm function which is internally evaluated on
    each persistent diagram separately, and on a function (e.g. `np.max`)
    which is applied to the resulting collection of norms to extract a single
    scale factor.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
        ``'betti'``, optional,  default: ``'bottleneck'``
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to a family of possible (:math:`L^p`-like)
           distances between "persistence landscapes" obtained from
           persistence (sub)diagrams.
        - ``'betti'`` refers to a family of possible (:math:`L^p`-like)
           distances between "Betti curves" obtained from persistence (
           sub)diagrams. A Betti curve simply records the evolution in the
           number of independent topological holes (technically, the number
           of linearly independent homology classes) as can be read from a
           persistence (sub)diagram.
        - ``'heat'`` heat kernel

    metric_params : dict, optional, default: ``{'n_samples': 200}``
        Additional keyword arguments for the norm function:

        - If ``norm == 'bottleneck'`` the only argument is `order`
          (default: ``numpy.inf``).
        - If ``norm == 'wasserstein'`` the only argument is `order`
          (default: ``1.``).
        - If ``norm == 'landscape'`` the available arguments are `order`
          (default: ``2.``), `n_samples` (default: ``200``) and `n_layers`
          (default: ``1``).
        - If ``norm == 'betti'``` the available arguments are `order`
          (default: ``2.``) and `n_samples` (default: ``200``).
        - If ``metric == 'heat'`` the available arguments are `order`
          (default: ``2.``), `sigma` (default: ``1.``), and `n_samples` (
          default: ``200``).

    function : callable, optional, default: numpy.max
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributess
    ----------
    scale_ : float
        The scaling factor used to rescale diagrams.

    """

    def __init__(self, metric='bottleneck', metric_params=None,
                 function=np.max, n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.function = function
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fits the transformer by finding the scale factor according to the
        chosen parameters.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``numpy.inf`` are used for padding and
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

        amplitude_array = _parallel_amplitude(X, self.metric,
                                              self.effective_metric_params_,
                                              n_jobs=self.n_jobs)
        self.scale_ = self.function(amplitude_array)

        return self

    def transform(self, X, y=None):
        """Rescales all persistence diagrams in the collection according to the
        factor computed during `fit`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``numpy.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xs : dict of int: ndarray
            Dictionary of rescaled persistence (sub)diagrams, each with the
            same shape as the corresponding (sub)diagram in `X`.
        """
        check_is_fitted(self, ['scale_', 'effective_metric_params_'])

        Xs = check_diagram(X)
        Xs[:, :, :2] = X[:, :, :2] / self.scale_
        return Xs

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation. Multiplies
        by the scale found in `fit`.

        Parameters
        ----------
        X : dict of int: ndarray
            Data to apply the inverse transform to.

        copy : bool, optional (default: None)
            Copy the input X or not.

        Returns
        -------
        X_scaled : dict of int: ndarray
            Transformed array.
        """
        check_is_fitted(self, ['effective_metric_params_'])

        Xs = check_diagram(X)
        Xs[:, :, :2] = X[:, :, :2] * self.scale_
        return Xs


class Filtering(BaseEstimator, TransformerMixin):
    """Transformer filtering collections of persistence diagrams in which each
    diagram is partitioned into one or more subdiagrams (e.g. according to
    homology dimension).

    Filtering a persistence (sub)diagram means removing
    all points whose distance from the diagonal is less than or equal to a
    certain cutoff value: that is, the cutoff value can be interpreted as the
    "minimum amount of persistence" required from points in the filtered
    diagram.

    Parameters
    ----------
    homology_dimensions : list or None, optional, default: ``None``
        When set to ``None``, all available (sub)diagrams will be filtered.
        When set to a list, it is interpreted as the list of those homology
        dimensions for which (sub)diagrams should be filtered.

    delta : float, optional, default: ``0.``
        The cutoff value controlling the amount of filtering.

    """

    implemented_filtering_parameters_types = ['fixed', 'search']

    def __init__(self, homology_dimensions=None, delta=0.):
        self.homology_dimensions = homology_dimensions
        self.delta = delta

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``numpy.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)

        if self.homology_dimensions is None:
            self.homology_dimensions_ = set(X[0, :, 2])
        else:
            self.homology_dimensions_ = sorted(self.homology_dimensions)

        return self

    def transform(self, X, y=None):
        r"""Filters all relevant persistence (sub)diagrams, and returns them.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            Triples in which q equals ``numpy.inf`` are used for padding and
            carry no information.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : dict of int: ndarray
            Dictionary of filtered persistence (sub)diagrams. The value
            corresponding to key d has shape (n_samples, F_d, 2), where
            :math:`F_\mathrm{d} \leq M_\mathrm{d}` in general, due to
            filtering.
            If `homology_dimensions` was set to be a list not containing all
            keys in `X`, only the corresponding (sub)diagrams are filtered and
            returned.
        """

        # Check if fit had been called
        check_is_fitted(self, ['homology_dimensions_'])
        X = check_diagram(X)

        X = _sort(X)
        Xt = _filter(X, self.homology_dimensions_, self.delta)
        return Xt
