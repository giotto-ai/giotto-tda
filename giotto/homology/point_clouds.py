"""Persistent homology on point clouds or finite metric spaces."""
# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors.base import VALID_METRICS
from joblib import Parallel, delayed
from sklearn.utils.validation import check_is_fitted
from ._utils import _pad_diagram

from ..externals.python import ripser


class VietorisRipsPersistence(BaseEstimator, TransformerMixin):
    """`Persistence diagrams <LINK TO GLOSSARY>`_ (equivalently,
    `persistence barcodes <LINK TO GLOSSARY>`_) resulting from
    `Vietoris-Rips filtrations <LINK TO GLOSSARY>`_.

    Given a `point cloud <LINK TO GLOSSARY>`_ in Euclidean space, or an
    abstract `metric space <LINK TO GLOSSARY>`_ encoded by a distance matrix,
    information about the appearance and disappearance of topological features
    (technically, `homology classes <LINK TO GLOSSARY>`_) of various
    dimensions and at different scales is summarised in the corresponding
    persistence diagram.

    Parameters
    ----------
    metric : string or callable, optional, default: 'euclidean'
        If set to `'precomputed'`, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays),
        and `metric` determines a rule with which to calculate distances
        between pairs of instances (i.e. rows) in these arrays.
        If `metric` is a string, it must be one of the options allowed by
        scipy.spatial.distance.pdist for its metric parameter, or a metric
        listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS, including "euclidean",
        "manhattan", or "cosine".
        If `metric` is a callable function, it is called on each pair of
        instances and the resulting value recorded. The callable should take
        two arrays from the entry in X as input, and return a value
        indicating the distance between them.

    max_edge_length : float, optional, default: np.inf
        Upper bound on the maximum value of the Vietoris-Rips filtration
        parameter. Points whose distance is greater than this value will
        never be connected by an edge, and topological features at scales
        larger than this value will not be detected.

    homology_dimensions : iterable, optional, default: (0, 1)
        Dimensions (non-negative integers) of the topological voids to be
        detected.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    """

    def __init__(self, metric='euclidean', max_edge_length=np.inf,
                 homology_dimensions=(0, 1), n_jobs=None):
        self.metric = metric
        self.max_edge_length = max_edge_length
        self.homology_dimensions = homology_dimensions
        self.n_jobs = n_jobs

    @staticmethod
    def _validate_params(metric):
        """A class method that checks whether the hyperparameters and the input
        parameters of the :meth:`fit` are valid.
        """
        implemented_metric_types = set(['precomputed'] +
                                       [met for i in VALID_METRICS.values()
                                        for met in i])

        if metric not in implemented_metric_types:
            raise ValueError('The metric %s is not supported' % metric)

    def _ripser_diagram(self, X, is_distance_matrix, metric):
        Xds = ripser(X[X[:, 0] != np.inf], distance_matrix=is_distance_matrix,
                     metric=metric, maxdim=self._max_homology_dimension,
                     thresh=self.max_edge_length)['dgms']

        if 0 in self._homology_dimensions:
            Xds[0] = Xds[0][:-1, :]

        Xds = {dim: np.hstack([Xds[dim], dim * np.ones((Xds[dim].shape[0], 1),
                                                       dtype=Xds[dim].dtype)])
               for dim in self._homology_dimensions}
        return Xds

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.
        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, n_points) or
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            (n_points, n_points). Otherwise, each such entry will be
            interpreted as an ndarray of `n_points` in Euclidean space of
            dimension `n_dimensions`.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        self._validate_params(self.metric)

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """Compute, for each point cloud or distance matrix in X,
        the relevant persistence diagram as an array of triples [b, d,
        q]. When q is not equal to ``np.inf``, each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Triples
        `[0., 0., np.inf]` are used for padding, as the number of persistent
        topological features is generally different between different entries
        in X.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_points, n_points) or
            (n_samples, n_points, n_dimensions)
            Input data. If ``metric == 'precomputed'``, the input should be an
            ndarray whose each entry along axis 0 is a distance matrix of shape
            (n_points, n_points). Otherwise, each such entry will be
            interpreted as an ndarray of `n_points` in Euclidean space of
            dimension `n_dimensions`.

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. `n_features` is the maximum number of
            topological features across all samples in `X`, and padding by `[0,
            0, np.inf]` is performed when necessary.

        """
        # Check if fit had been called
        check_is_fitted(self, ['_homology_dimensions',
                               '_max_homology_dimension'])

        is_distance_matrix = (self.metric == 'precomputed')

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ripser_diagram)(X[i, :, :], is_distance_matrix,
                                          self.metric)
            for i in range(X.shape[0]))

        max_n_points = {dim: max(1, np.max([Xt[i][dim].shape[0]
                                            for i in range(len(Xt))]))
                        for dim in self.homology_dimensions}
        min_values = {dim: min([np.min(Xt[i][dim][:, 0]) if Xt[i][dim].size
                                else np.inf for i in range(len(Xt))])
                      for dim in self.homology_dimensions}

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(_pad_diagram)(Xt[i], self._homology_dimensions,
                                  max_n_points, min_values)
            for i in range(len(Xt)))
        Xt = np.stack(Xt)
        return Xt
