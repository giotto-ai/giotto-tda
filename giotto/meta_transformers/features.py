"""Feature generation using meta transformers."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from giotto.pipeline import Pipeline
from ..utils import validate_params, validate_metric_params
from giotto import homology as hl
from giotto import diagrams as diag
import numbers


class EntropyGenerator(BaseEstimator, TransformerMixin):
    r"""Meta transformer that returns the persistent entropy.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and ``metric`` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If ``metric`` is a string, it must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric
        listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS, including "euclidean",
        "manhattan", or "cosine".
        If ``metric`` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in X as input, and return a value indicating
        the distance between them.

    scaler_metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
                    ``'betti'`` | ``'heat'``, optional, default: \
                    ``'bottleneck'``
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    scaler_metric_params : dict, optional, default: {'n_samples': 200}
        Additional keyword arguments for the norm function:

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

    max_edge_length : float, optional, default: np.inf
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter.
        Points whose distance is greater than this value will never be
        connected by an edge, and topological features at scales larger than
        this value will not be detected.

    function : callable, optional, default: numpy.max
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    homology_dimensions : list or None, optional, default: None
        When set to ``None``, all available (sub)diagrams will be filtered.
        When set to a list, it is interpreted as the list of those homology
        dimensions for which (sub)diagrams should be filtered.

    epsilon : float, optional, default: 0.
        The cutoff value controlling the amount of filtering.

    len_vector : int, optional, default: 8
        Used for performance optimization by exploiting numpy's vectorization
        capabilities.

    Attributes
    ----------
    _n_features : int
        Number of features (i.e. number of time series) passed as an input
        of the resampler.

    Examples
    --------
    >>> from giotto.meta_transformers import EntropyGenerator as eg
    >>> import numpy as np
    >>> ent = eg()
    >>> X = np.asarray([[[1,2],[2,1],[1,1]]])
    >>> X_tr = ent.fit_transform(X)
    >>> X_tr
    ... array([[ 0.69314718, -0.        ]])

    """

    _hyperparameters = {'max_edge_length': [numbers.Number, (0, np.inf)],
                        'epsilon': [numbers.Number, (0, 1)]}

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=None, scaler_metric='bottleneck',
                 scaler_metric_params=None, epsilon=0., n_jobs=None,
                 function=np.max):
        self.steps = None
        self.metric = metric
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.function = function
        self.scaler_metric = scaler_metric
        self.scaler_metric_params = scaler_metric_params

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.

        """
        # still need to check the homology dimensions with utils fnc
        validate_params(self.get_params(), self._hyperparameters)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape: (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of the
            point cloud (i.e. the dimension of the point cloud space).

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.

        """

        if self.scaler_metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.scaler_metric_params.copy()

        validate_metric_params(self.scaler_metric,
                               self.effective_metric_params_)

        if self.homology_dimensions is None:
            self.effective_homology_dimensions_ = (0, 1)
        else:
            self.effective_homology_dimensions_ = \
                self.homology_dimensions.copy()

        self._validate_params()

        self._is_fitted = True

        self.steps = [
            ('diagram', hl.VietorisRipsPersistence(
                metric=self.metric,
                max_edge_length=self.max_edge_length,
                homology_dimensions=self.effective_homology_dimensions_,
                n_jobs=self.n_jobs)),
            ('rescaler', diag.Scaler(
                metric=self.scaler_metric,
                metric_params=self.effective_metric_params_,
                function=self.function,
                n_jobs=self.n_jobs)),
            ('filter', diag.Filtering(
                epsilon=self.epsilon,
                homology_dimensions=self.effective_homology_dimensions_)),
            ('entropy', diag.PersistenceEntropy(n_jobs=self.n_jobs))]

        Pipeline(self.steps).fit(X)

        return self

    def transform(self, X, y=None):
        """Extract the persistent entropy from X.

        Parameters
        ----------
        X : ndarray, shape: (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of the
            point cloud (i.e. the dimension of the point cloud space)

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape: (n_samples, n_homology_dimensions)
            The Persistent Entropy. ``n_samples`` is not modified by the
            algorithm: the Persistent Entropy is computed per point cloud.
            ``n_homology_dimensions`` is the number of homology
            dimensions considered, i.e. the length of ``homology_dimensions``.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = Pipeline(self.steps).transform(X)
        return X_transformed


class BettiCurveGenerator(BaseEstimator, TransformerMixin):
    """Meta_transformer that returns the sampled Betti curves.

    Parameters
    ----------
    metric : string or callable, optional, default: 'euclidean'
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and ``metric`` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If ``metric`` is a string, it must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric
        listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS, including "euclidean",
        "manhattan", or "cosine".
        If ``metric`` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in X as input, and return a value indicating
        the distance between them.

    scaler_metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
                    ``'betti'`` | ``'heat'``, optional, default: \
                    ``'bottleneck'``
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    scaler_metric_params : dict, optional, default: {'n_samples': 200}
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

    max_edge_length : float, optional, default: np.inf
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter.
        Points whose distance is greater than this value will never be
        connected by an edge, and topological features at scales larger than
        this value will not be detected.

    function : callable, optional, default: numpy.max
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    homology_dimensions : list or None, optional, default: None
        When set to ``None``, all available (sub)diagrams will be filtered.
        When set to a list, it is interpreted as the list of those homology
        dimensions for which (sub)diagrams should be filtered.

    epsilon : float, optional, default: 0.
        The cutoff value controlling the amount of filtering.

    n_values : int, optional, default: 100
        Used to sample the Betti curves to extract a finite dimensional
        array.

    """
    _hyperparameters = {'max_edge_length': [numbers.Number, (0, np.inf)],
                        'epsilon': [numbers.Number, (0, 1)],
                        'n_values': [int, (1, np.inf)]
                        }

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=None, scaler_metric='bottleneck',
                 scaler_metric_params=None, epsilon=0., n_jobs=None,
                 function=np.max, n_values=100):
        self.steps = None
        self.metric = 'euclidean'
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.scaler_metric_params = scaler_metric_params
        self.scaler_metric = scaler_metric
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.function = function
        self.n_values = n_values

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:`fit` are valid.

        """
        # still need to check the homology dimensions with utils fnc
        validate_params(self.get_params(), self._hyperparameters)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape: (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of the
            point cloud (i.e. the dimension of the point cloud space)

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.

        """

        if self.scaler_metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.scaler_metric_params.copy()

        validate_metric_params(self.scaler_metric,
                               self.effective_metric_params_)

        self._validate_params()

        if self.homology_dimensions is None:
            self.effective_homology_dimensions_ = (0, 1)
        else:
            self.effective_homology_dimensions_ = \
                self.homology_dimensions.copy()

        self.steps = [
            ('diagram', hl.VietorisRipsPersistence(
                metric=self.metric,
                max_edge_length=self.max_edge_length,
                homology_dimensions=self.effective_homology_dimensions_,
                n_jobs=self.n_jobs)),
            ('rescaler', diag.Scaler(
                metric=self.scaler_metric,
                metric_params=self.effective_metric_params_,
                function=self.function,
                n_jobs=self.n_jobs)),
            ('filter', diag.Filtering(
                epsilon=self.epsilon,
                homology_dimensions=self.effective_homology_dimensions_)),
            ('betticurve', diag.BettiCurve(n_values=self.n_values))]

        Pipeline(self.steps).fit(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Extract the persistent entropy from X.

        Parameters
        ----------
        X : ndarray, shape: (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of
            the point cloud (i.e. the dimension of the point cloud space).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape: (n_samples, n_homology_dimensions, \
                        n_sampled_values)
            The Betti Curves. ``n_samples`` is not modified by the
            algorithm: the Bettiv curves are computed per point cloud.
            ``n_homology_dimensions`` is the number of homology
            dimensions considered, i.e. the length of ``homology_dimensions``.
            The parameter ``n_sampled_values`` is the number of points
            used to sample the continuous Betti Curves into arrays.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = Pipeline(self.steps).transform(X)
        return X_transformed


class LandscapeGenerator(BaseEstimator, TransformerMixin):
    """Meta_transformer that returns the sampled Persistence Landscape.

    Parameters
    ----------
    metric : string or callable, optional, default: 'euclidean'
        If set to ``'precomputed'``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and ``metric`` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If ``metric`` is a string, it must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric
        listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS, including "euclidean",
        "manhattan", or "cosine".
        If ``metric`` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in X as input, and return a value indicating
        the distance between them.

    scaler_metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
                    ``'betti'`` | ``'heat'``, optional, default: \
                    ``'bottleneck'``
        Which notion of distance between (sub)diagrams to use:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
           perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    scaler_metric_params : dict, optional, default: {'n_samples': 200}
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

    max_edge_length : float, optional, default: np.inf
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter.
        Points whose distance is greater than this value will never be
        connected by an edge, and topological features at scales larger than
        this value will not be detected.

    function : callable, optional, default: numpy.max
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    homology_dimensions : list or None, optional, default: None
        When set to ``None``, all available (sub)diagrams will be filtered.
        When set to a list, it is interpreted as the list of those homology
        dimensions for which (sub)diagrams should be filtered.

    epsilon : float, optional, default: 0.
        The cutoff value controlling the amount of filtering.

    n_sampled_values : int, optional, default: 100
        Used to sample the Betti curves to extract a finite dimensional array.

    n_layers : int, optional, default: 1
        Used to specify which of the persistence landscape profiles to
        consider.

    """
    _hyperparameters = {'max_edge_length': [numbers.Number, (0, np.inf)],
                        'epsilon': [numbers.Number, (0, 1)],
                        'n_values': [int, (1, np.inf)],
                        'n_layers': [int, (1, np.inf)]
                        }

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=None, scaler_metric='bottleneck',
                 scaler_metric_params=None, epsilon=0., n_jobs=None,
                 function=np.max, n_values=100, n_layers=1):
        self.steps = None
        self.metric = 'euclidean'
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.scaler_metric_params = scaler_metric_params
        self.scaler_metric = scaler_metric
        self.n_jobs = n_jobs
        self.epsilon = epsilon
        self.function = function
        self.n_values = n_values
        self.n_layers = n_layers

    def _validate_params(self):
        """A class method that checks whether the hyperparameters and the
        input parameters of the :meth:``fit`` are valid.

        """
        # still need to check the homology dimensions with utils fnc
        validate_params(self.get_params(), self._hyperparameters)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is there to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape: (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of
            the point cloud (i.e. the dimension of the point cloud space).

        y : None
            Ignored.

        Returns
        -------
        self : object
            Returns self.

        """

        if self.scaler_metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.scaler_metric_params.copy()

        validate_metric_params(self.scaler_metric,
                               self.effective_metric_params_)

        self._validate_params()

        if self.homology_dimensions is None:
            self.effective_homology_dimensions_ = [0, 1]
        else:
            self.effective_homology_dimensions_ = \
                self.homology_dimensions.copy()

        self.steps = [
            ('diagram', hl.VietorisRipsPersistence(
                metric=self.metric,
                max_edge_length=self.max_edge_length,
                homology_dimensions=self.effective_homology_dimensions_,
                n_jobs=self.n_jobs)),
            ('rescaler', diag.Scaler(
                metric=self.scaler_metric,
                metric_params=self.effective_metric_params_,
                function=self.function,
                n_jobs=self.n_jobs)),
            ('filter', diag.Filtering(
                epsilon=self.epsilon,
                homology_dimensions=self.effective_homology_dimensions_)),
            ('landscape', diag.PersistenceLandscape(
                n_values=self.n_values, n_layers=self.n_layers))]

        Pipeline(self.steps).fit(X)
        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Extract the persistent entropy from X.

        Parameters
        ----------
        X : ndarray, shape: (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of
            the point cloud (i.e. the dimension of the point cloud space).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        X_transformed : ndarray, shape: (n_samples, n_homology_dimensions, \
                        n_values)
            The Persistence Landscape. ``n_samples`` is not modified by the
            algorithm: the Betti curves are computed per point cloud.
            ``n_homology_dimensions`` is the number of homology
            dimensions considered, i.e. the length of ``homology_dimensions``.
            The parameter ``n_values`` is the number of points
            used to sample the continuous persistence landscape into arrays.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])

        X_transformed = Pipeline(self.steps).transform(X)
        return X_transformed
