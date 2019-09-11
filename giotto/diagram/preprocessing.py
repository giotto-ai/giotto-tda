# Authors: Guillaume Tauzin <guillaume.tauzin@epfl.ch>
# License: TBD

import math as m
import numpy as np
import sklearn as sk
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from functools import partial
from sklearn.utils._joblib import Parallel, delayed
import itertools

from ._utils import _sort, _filter, _sample
from ._metrics import _parallel_norm
from .distance import DiagramDistance


class DiagramStacker(BaseEstimator, TransformerMixin):
    """Transformer for stacking persistence subdiagrams.

    Useful when topological
    persistence information per sample has been previously separated according
    to some criterion (e.g. by homology dimension if produced by an instance of
    ```VietorisRipsPersistence``).

    """

    def __init__(self):
        pass

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: True
            Behaviour not yet implemented.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {}

    @staticmethod
    def _validate_params():
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
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative
            integers d representing homology dimensions, and whose values
            are ndarrays of shape (n_samples, M_d, 2) whose each entries along
            axis 0 are persistence diagrams with M_d persistent topological
            features. For example, X could be the result of applying the
            ``transform`` method of a ``VietorisRipsPersistence`` transformer
            to a collection of point clouds/distance matrices, but only if
            that transformer was instantiated with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.

        """
        self._validate_params()

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Stacks all available persistence subdiagrams corresponding
        to each sample into one persistence diagram.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative
            integers d representing homology dimensions, and whose values
            are ndarrays of shape (n_samples, M_d, 2) whose each entries
            along axis 0 are persistence diagrams with M_d persistent
            topological features. For example, X could be the result of
            applying the ``transform`` method of a ``VietorisRipsPersistence``
            transformer to a collection of point clouds/distance matrices,
            but only if that transformer was instantiated with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : dict of None: ndarray
            Dictionary with a single ``None`` key, and corresponding value an
            ndarray of shape (n_samples, :math:`\\sum_{\\mathrm{d}}` M_d, 2).

        """
        # Check is fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = {None: np.concatenate(list(X.values()), axis=1)}
        return X_transformed


class DiagramScaler(BaseEstimator, TransformerMixin):
    """Transformer scaling collections of persistence diagrams in which each
    diagram is partitioned into one or more subdiagrams (e.g. according to
    homology dimension).

    A scale factor is calculated during ``fit`` which depends on the entire
    collection, and it is applied during ``transform``. The value of the scale
    factor depends on a chosen norm function which is internally evaluated on
    each persistent diagram separately, and on a function (e.g. ``np.max``)
    which is applied to the resulting collection of norms to extract a single
    scale factor.

    Parameters
    ----------
    metric : 'bottleneck' | 'wasserstein' | 'landscape' | 'betti', optional, default: 'bottleneck'
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to a family of possible (:math:`L^p`-like) distances
           between "persistence landscapes" obtained from persistence (sub)diagrams.
        - ``'betti'`` refers to a family of possible (:math:`L^p`-like) distances
           between "Betti curves" obtained from persistence (sub)diagrams. A Betti
           curve simply records the evolution in the number of independent topological
           holes (technically, the number of linearly independent homology classes)
           as can be read from a persistence (sub)diagram.
        - ``'heat'`` heat kernel

    metric_params : dict, optional, default: {'n_samples': 200}
        Additional keyword arguments for the norm function:

        - If ``norm == 'bottleneck'`` the only argument is ``order``
          (default = ``np.inf``).
        - If ``norm == 'wasserstein'`` the only argument is ``order``
          (default = ``1``).
        - If ``norm == 'landscape'`` the available arguments are ``order``
          (default = ``2``), ``n_samples`` (default = ``200``) and ``n_layers``
          (default = ``1``).
        - If ``norm == 'betti'`` the available arguments are ``order``
          (default = ``2``) and ``n_samples`` (default = ``200``).
        - If ``metric == 'heat'`` the available arguments are ``order`` (default = ``2``)
           ``sigma`` (default = ``1``), and ``n_samples`` (default = ``200``).

    function : callable, optional, default: numpy.max
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    """

    def __init__(self, metric='bottleneck',
                 metric_params={'order': np.inf, 'n_samples': 200},
                 function=np.max, n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.function = function
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: True
            Behaviour not yet implemented.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {'metric': self.metric, 'metric_params': self.metric_params,
                'function': self.function, 'n_jobs': self.n_jobs}

    def fit(self, X, y=None):
        """Fits the transformer by finding the scale factor according to the
        chosen parameters.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative
            integers d representing homology dimensions, and whose values
            are ndarrays of shape (n_samples, M_d, 2) whose each entries
            along axis 0 are persistence diagrams with M_d persistent
            topological features. For example, X could be the result of
            applying the ``transform`` method of a ``VietorisRipsPersistence``
            transformer to a collection of point clouds/distance matrices,
            but only if that transformer was instantiated with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """
        metric_params = self.metric_params.copy()

        sampling = {dimension: None for dimension in X.keys()}

        if 'n_samples' in metric_params.keys():
            n_samples = metric_params.pop('n_samples')

        if self.metric in ['landscape', 'betti', 'heat']:
            metric_params['sampling'] = _sample(X, n_samples)

        norm_array = _parallel_norm(X, self.metric, metric_params, self.n_jobs)
        self._scale = self.function(norm_array)

        self._is_fitted = True
        return self

    # @jit
    def transform(self, X, y=None):
        """Rescales all persistence diagrams in the collection according to the
        factor computed during ``fit``.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative
            integers d representing homology dimensions, and whose values are
            ndarrays of shape (n_samples, M_d, 2) whose each entries along axis
            0 are persistence diagrams with M_d persistent topological
            features. For example, X could be the result of applying the
            ``transform`` method of a ``VietorisRipsPersistence`` transformer
            to a collection of point clouds/distance matrices, but only if
            that transformer was instantiated with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_scaled : dict of int: ndarray
            Dictionary of rescaled persistence (sub)diagrams, each with the
            same shape as the corresponding (sub)diagram in X.
        """
        check_is_fitted(self, ['_is_fitted'])

        X_scaled = {dimension: X / self._scale for dimension, X in X.items()}
        return X_scaled

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation. Multiplies
        by the scale found in ``fit``.

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
        check_is_fitted(self, ['_is_fitted'])

        X_scaled = {dimension: X * self._scale for dimension, X in X.items()}
        return X_scaled


class DiagramFilter(BaseEstimator, TransformerMixin):
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
    homology_dimensions : list or None, optional, default: None
        When set to ``None``, all available (sub)diagrams will be filtered.
        When set to a list, it is interpreted as the list of those homology
        dimensions for which (sub)diagrams should be filtered.

    filtering_parameters_type : str, optional, default: 'fixed'
        When set to ``'fixed'``, ``'epsilon'`` is ignored and filtering
        consists simply in removing from all persistent (sub)diagrams all
        points an absolute distance ``delta`` from the diagonal.

    delta : float, optional, default: 0.
        The cutoff value controlling the amount of filtering.

    metric : 'bottleneck' | 'wasserstein' | 'landscape' | 'betti',
    optional, default: 'bottleneck'
        Behaviour not implemented. This variable is currently ignored.

    metric_params : dict, optional, default: {'n_samples': 200}
        Behaviour not implemented. This variable is currently ignored.

    epsilon : float, optional, default: 1.
        Behaviour not implemented. This variable is currently ignored.

    tolerance : float, optional, default: 1e-2
        Behaviour not implemented. This variable is currently ignored.

    max_iteration : int, optional, default: 20
        Behaviour not implemented. This variable is currently ignored.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means using
        all processors.

    """

    implemented_filtering_parameters_types = ['fixed', 'search']

    def __init__(self, homology_dimensions=None,
                 filtering_parameters_type='fixed', delta=0.,
                 metric='bottleneck', metric_params={'order': np.inf},
                 epsilon=1.,
                 tolerance=1e-2, max_iteration=20, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.filtering_parameters_type = filtering_parameters_type
        self.delta = delta
        self.metric = metric
        self.metric_params = metric_params
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.max_iteration = max_iteration
        self.n_jobs = n_jobs

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional, default: True
            Behaviour not yet implemented.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """
        return {'homology_dimensions': self.homology_dimensions,
                'filtering_parameters_type': self.filtering_parameters_type,
                'delta': self.delta,
                'metric': self.metric,
                'metric_params': self.metric_params,
                'epsilon': self.epsilon,
                'tolerance': self.tolerance,
                'max_iteration': self.max_iteration,
                'n_jobs': self.n_jobs}

    @staticmethod
    def _validate_params(filtering_parameters_type):
        if (filtering_parameters_type not in
                DiagramFilter.implemented_filtering_parameters_types):
            raise ValueError("""The filtering parameters type you specified
                is not implemented""")

    def _bisection(self, X):
        iterator = iter([(i, i) for i in range(len(X))])

        numberPoints = [X[dimension].shape
                        for dimension in self.homologyDimensions]
        XConcatenated = np.concatenate(
            [X[dimension] for dimension in self.homologyDimensions])

        lowerCutoff = 0.
        upperCutoff = 1.

        currentMeanDistance = 0.
        previousMeanDistance = 1.

        while m.abs((currentMeanDistance - previousMeanDistance) >
        self.tolerance and iteration <= self.max_iteration):
            middlePoint = (lowerPoint + upperPoint) // 2.
            middlePointIndex = indices[:, middlePoint]
            # cutoff = m.sqrt(2.)/2. * (XConcatenated[indices[middlePoint][0],
            # indices[middlePoint][0], 1] - XConcatenated[, , 0])
            # XFiltered = _filter(XConcatenated, self.homologyDimensions,
            # middleCutoff)

            # XCOncatenated and XFIltered need to have the same homology
            # dimensions!!!!!
            # distance = _parallel_pairwise(XConcatenated, XFiltered, iterator,
            # self.n_jobs)

            if distance == tolerance:
                return middleCutoff
            elif (distance - tolerance) * () < 0:
                upperCutoff = middleCutoff
            else:
                lowerCutoff = middleCutoff

        return middleCutoff

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative
            integers d representing homology dimensions, and whose values are
            ndarrays of shape (n_samples, M_d, 2) whose each entries along axis
            0 are persistence diagrams with M_d persistent topological
            features. For example, X could be the result of applying the
            ``transform`` method of a ``VietorisRipsPersistence`` transformer
            to a collection of point clouds/distance matrices, but only if that
            transformer was instantiated with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object
            Returns self.
        """

        if not self.homology_dimensions:
            self.homology_dimensions = set(X.keys())

        self._validate_params(self.filtering_parameters_type)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Filters all relevant persistence (sub)diagrams, and returns them.

        Parameters
        ----------
        X : dict of int: ndarray
            Input data. Dictionary whose keys are typically non-negative
            integers d representing homology dimensions, and whose values are
            ndarrays of shape (n_samples, M_d, 2) whose each entries along axis
            0 are persistence diagrams with M_d persistent topological
            features. For example, X could be the result of applying the
            ``transform`` method of a ``VietorisRipsPersistence`` transformer
            to a collection of point clouds/distance matrices, but only if that
            transformer was instantiated with ``pad=True``.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_filtered : dict of int: ndarray
            Dictionary of filtered persistence (sub)diagrams. The value
            corresponding to key d has shape (n_samples, F_d, 2), where
            :math:`F_\mathrm{d} \leq M_\mathrm{d}` in general, due to
            filtering.
            If ``homology_dimensions`` was set to be a list not containing all
            keys in X, only the corresponding (sub)diagrams are filtered and
            returned.

        """

        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X = _sort(X, self.homology_dimensions)

        X_filtered = _filter(X, self.homology_dimensions, self.delta)
        return X_filtered
