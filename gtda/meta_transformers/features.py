"""Feature generation using meta transformers."""
# License: GNU AGPLv3

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .. import diagrams as diag
from .. import homology as hl
from ..pipeline import Pipeline
from ..utils._docs import adapt_fit_transform_docs


@adapt_fit_transform_docs
class EntropyGenerator(BaseEstimator, TransformerMixin):
    """Persistence entropies directly from point clouds.

    Implements a feature generation pipeline which computes persistence
    diagrams, scales and filters them, and then computes their persistence
    entropies.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, each entry in `X` along axis 0 is
        interpreted to be a distance matrix. Otherwise, entries are
        interpreted as feature arrays, and `metric` determines a rule with
        which to calculate distances between pairs of instances (i.e. rows)
        in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan" or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    scaler_metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
                    ``'betti'`` | ``'heat'``, optional, default: \
                    ``'bottleneck'``
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    scaler_metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for `scaler_metric`:

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

    scaler_function : callable, optional, default: ``numpy.max``
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    filter_epsilon : float, optional, default: ``0.``
        The cutoff value controlling the amount of filtering.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Examples
    --------
    >>> from gtda.meta_transformers import EntropyGenerator as eg
    >>> import numpy as np
    >>> ent = eg()
    >>> X = np.asarray([[[1, 2], [2, 1], [1, 1]]])
    >>> Xt = ent.fit_transform(X)
    >>> print(Xt)
    [[0.69314718, -0.]]

    """

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=(0, 1), scaler_metric='bottleneck',
                 scaler_metric_params=None,
                 scaler_function=np.max, filter_epsilon=0.,
                 n_jobs=None):
        self.metric = metric
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.scaler_metric = scaler_metric
        self.scaler_metric_params = scaler_metric_params
        self.scaler_function = scaler_function
        self.filter_epsilon = filter_epsilon
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of the
            point cloud (i.e. the dimension of the point cloud space).

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        steps = [
            ('diagram', hl.VietorisRipsPersistence(
                metric=self.metric,
                max_edge_length=self.max_edge_length,
                homology_dimensions=self.homology_dimensions,
                n_jobs=self.n_jobs)),
            ('scaler', diag.Scaler(
                metric=self.scaler_metric,
                metric_params=self.scaler_metric_params,
                function=self.scaler_function,
                n_jobs=self.n_jobs)),
            ('filter', diag.Filtering(
                epsilon=self.filter_epsilon)),
            ('entropy', diag.PersistenceEntropy(n_jobs=self.n_jobs))]

        self._pipeline = Pipeline(steps).fit(X)

        return self

    def transform(self, X, y=None):
        """Extract persistence entropies from the sample point clouds in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of the
            point cloud (i.e. the dimension of the point cloud space).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions)
            For each point cloud in `X`, one persistence entropy per homology
            dimension in `homology_dimensions`.

        """
        Xt = self._pipeline.transform(X)
        return Xt


@adapt_fit_transform_docs
class BettiCurveGenerator(BaseEstimator, TransformerMixin):
    """Meta transformer returning Betti curves directly from point clouds.

    Implements a feature generation pipeline which computes persistence
    diagrams, scales and filters them, and then computes their Betti curves.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, each entry in `X` along axis 0 is
        interpreted to be a distance matrix. Otherwise, entries are
        interpreted as feature arrays, and `metric` determines a rule with
        which to calculate distances between pairs of instances (i.e. rows)
        in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan" or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    scaler_metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
                    ``'betti'`` | ``'heat'``, optional, default: \
                    ``'bottleneck'``
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    scaler_metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for `scaler_metric`:

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

    scaler_function : callable, optional, default: ``numpy.max``
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    filter_epsilon : float, optional, default: ``0.``
        The cutoff value controlling the amount of filtering.

    n_values : int, optional, default: ``100``
        Length of array used to sample the continuous Betti curves.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    """

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=(0, 1), scaler_metric='bottleneck',
                 scaler_metric_params=None, scaler_function=np.max,
                 filter_epsilon=0., n_values=100, n_jobs=None):
        self.metric = metric
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.scaler_metric_params = scaler_metric_params
        self.scaler_metric = scaler_metric
        self.scaler_function = scaler_function
        self.filter_epsilon = filter_epsilon
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Create a :class:`gtda.pipeline.Pipeline` object and fit it. Then,
        return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of the
            point cloud (i.e. the dimension of the point cloud space)

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        steps = [
            ('diagram', hl.VietorisRipsPersistence(
                metric=self.metric,
                max_edge_length=self.max_edge_length,
                homology_dimensions=self.homology_dimensions,
                n_jobs=self.n_jobs)),
            ('scaler', diag.Scaler(
                metric=self.scaler_metric,
                metric_params=self.scaler_metric_params,
                function=self.scaler_function,
                n_jobs=self.n_jobs)),
            ('filter', diag.Filtering(
                epsilon=self.filter_epsilon)),
            ('betticurve', diag.BettiCurve(n_values=self.n_values))]

        self._pipeline = Pipeline(steps).fit(X)
        return self

    def transform(self, X, y=None):
        """Extract Betti curves from the sample point clouds in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of
            the point cloud (i.e. the dimension of the point cloud space).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, \
             n_values)
            For each point cloud in `X`, one discretised Betti curve
            per homology dimension in `homology_dimensions`.

        """
        Xt = self._pipeline.transform(X)
        return Xt


@adapt_fit_transform_docs
class LandscapeGenerator(BaseEstimator, TransformerMixin):
    """Meta transformer returning persistence landscapes directly from point
    clouds.

    Implements a feature generation pipeline which computes persistence
    diagrams, scales and filters them, and then computes their persistence
    landscapes.

    Parameters
    ----------
    metric : string or callable, optional, default: ``'euclidean'``
        If set to ``'precomputed'``, each entry in `X` along axis 0 is
        interpreted to be a distance matrix. Otherwise, entries are
        interpreted as feature arrays, and `metric` determines a rule with
        which to calculate distances between pairs of instances (i.e. rows)
        in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan" or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in `X` as input, and return a value
        indicating the distance between them.

    max_edge_length : float, optional, default: ``numpy.inf``
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    homology_dimensions : iterable, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    scaler_metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'landscape'`` | \
                    ``'betti'`` | ``'heat'``, optional, default: \
                    ``'bottleneck'``
        Distance or dissimilarity function used to define the amplitude of
        a subdiagram as its distance from the diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.

    scaler_metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for `scaler_metric`:

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

    scaler_function : callable, optional, default: ``numpy.max``
        Function used to extract a single positive scalar from the collection
        of norms of diagrams.

    filter_epsilon : float, optional, default: ``0.``
        The cutoff value controlling the amount of filtering.

    n_layers : int, optional, default: ``1``
        How many layers to consider in the persistence landscape.

    n_values : int, optional, default: ``100``
        Length of array used to sample the continuous persistence landscapes.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    """

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=(0, 1), scaler_metric='bottleneck',
                 scaler_metric_params=None, scaler_function=np.max,
                 filter_epsilon=0., n_layers=1, n_values=100, n_jobs=None):
        self.metric = metric
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.scaler_metric = scaler_metric
        self.scaler_metric_params = scaler_metric_params
        self.scaler_function = scaler_function
        self.filter_epsilon = filter_epsilon
        self.n_layers = n_layers
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Create a :class:`gtda.pipeline.Pipeline` object and fit it. Then,
        return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of
            the point cloud (i.e. the dimension of the point cloud space).

        y : None
            Ignored.

        Returns
        -------
        self : object

        """
        steps = [
            ('diagram', hl.VietorisRipsPersistence(
                metric=self.metric,
                max_edge_length=self.max_edge_length,
                homology_dimensions=self.homology_dimensions,
                n_jobs=self.n_jobs)),
            ('scaler', diag.Scaler(
                metric=self.scaler_metric,
                metric_params=self.scaler_metric_params,
                function=self.scaler_function,
                n_jobs=self.n_jobs)),
            ('filter', diag.Filtering(
                epsilon=self.filter_epsilon)),
            ('landscape', diag.PersistenceLandscape(
                n_values=self.n_values, n_layers=self.n_layers))]

        self._pipeline = Pipeline(steps).fit(X)
        return self

    def transform(self, X, y=None):
        """Extract persistence landscapes from the sample point clouds in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_points, n_dimensions)
            Input data. ``n_samples`` is the number of point clouds,
            ``n_points`` is the number of points per point cloud and
            ``n_dimensions`` is the number of features for each point of
            the point cloud (i.e. the dimension of the point cloud space).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, \
             n_layers, n_values)
            For each point cloud in `X`, one discretised persistence landscape
            per homology dimension in `homology_dimensions`, consisting of
            `n_layers` layers.

        """
        Xt = self._pipeline.transform(X)
        return Xt
