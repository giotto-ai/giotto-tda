"""Persistent homology on point clouds or finite metric spaces."""
# License: GNU AGPLv3

from numbers import Real, Integral
from typing import Callable

import numpy as np
from gph import ripser_parallel as ripser
from joblib import Parallel, delayed
from pyflagser import flagser_weighted
from scipy.sparse import coo_matrix
from scipy.spatial import Delaunay
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_is_fitted

from ._utils import _postprocess_diagrams
from ..base import PlotterMixin
from ..externals.python import SparseRipsComplex, CechComplex
from ..plotting import plot_diagram
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params, check_point_clouds

_AVAILABLE_RIPS_WEIGHTS = {
    "DTM": {
        "p": {"type": Real, "in": [1, 2, np.inf]},
        "r": {"type": Real, "in": Interval(0, np.inf, closed="right")},
        "n_neighbors": {"type": Integral,
                        "in": Interval(1, np.inf, closed="left")}
        },
    "general": {
        "p": {"type": Real, "in": [1, 2, np.inf]},
        }
    }


@adapt_fit_transform_docs
class VietorisRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`Vietoris–Rips filtrations
    <vietoris-rips_complex_and_vietoris-rips_persistence>`.

    Given a :ref:`point cloud <distance_matrices_and_point_clouds>` in
    Euclidean space, an abstract :ref:`metric space
    <distance_matrices_and_point_clouds>` encoded by a distance matrix, or the
    adjacency matrix of a weighted undirected graph, information about the
    appearance and disappearance of topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimensions
    and at different scales is summarised in the corresponding persistence
    diagram.

    **Important note**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.

    Parameters
    ----------
    metric : string or callable, optional, default: ``"euclidean"``
        If set to ``"precomputed"``, input data is to be interpreted as a
        collection of distance matrices or of adjacency matrices of weighted
        undirected graphs. Otherwise, input data is to be interpreted as a
        collection of point clouds (i.e. feature arrays), and `metric`
        determines a rule with which to calculate distances between pairs of
        points (i.e. row vectors). If `metric` is a string, it must be one of
        the options allowed by :func:`scipy.spatial.distance.pdist` for its
        metric parameter, or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``"euclidean"``, ``"manhattan"`` or ``"cosine"``. If `metric` is a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    metric_params : dict, optional, default: ``{}``
        Additional parameters to be passed to the distance function.

    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    collapse_edges : bool, optional, default: ``False``
        Whether to run the edge collapse algorithm in [2]_ prior to the
        persistent homology computation (see the Notes). Can reduce the runtime
        dramatically when the data or the maximum homology dimensions are
        large.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Vietoris–Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge,
        and topological features at scales larger than this value will not be
        detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    reduced_homology : bool, optional, default: ``True``
       If ``True``, the earliest-born triple in homology dimension 0 which has
       infinite death is discarded from each diagram computed in
       :meth:`transform`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    WeightedRipsPersistence, FlagserPersistence, SparseRipsPersistence,
    WeakAlphaPersistence, EuclideanCechPersistence, ConsistentRescaling,
    ConsecutiveRescaling

    Notes
    -----
    `giotto-ph <https://github.com/giotto-ai/giotto-ph>`_ [1]_ is used as a C++
    backend for computing Vietoris–Rips persistent homology and edge collapses.

    References
    ----------
    .. [1] J. Burella Pérez et al, "giotto-ph: A Python Library for
           High-Performance Computation of Persistent Homology of Vietoris–Rips
           Filtrations", 2021; `arXiv:2107.05412
           <https://arxiv.org/abs/2107.05412>`_.

    .. [2] J.-D. Boissonnat and S. Pritam, "Edge Collapse and Persistence of
           Flag Complexes"; in *36th International Symposium on Computational
           Geometry (SoCG 2020)*, pp. 19:1–19:15,
           Schloss Dagstuhl-Leibniz–Zentrum für Informatik, 2020;
           `DOI: 10.4230/LIPIcs.SoCG.2020.19
           <https://doi.org/10.4230/LIPIcs.SoCG.2020.19>`_.

    """

    _hyperparameters = {
        "metric": {"type": (str, Callable)},
        "metric_params": {"type": dict},
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
            },
        "collapse_edges": {"type": bool},
        "coeff": {"type": int, "in": Interval(2, np.inf, closed="left")},
        "max_edge_length": {"type": Real},
        "infinity_values": {"type": (Real, type(None))},
        "reduced_homology": {"type": bool}
        }

    def __init__(self, metric="euclidean", metric_params={},
                 homology_dimensions=(0, 1), collapse_edges=False, coeff=2,
                 max_edge_length=np.inf, infinity_values=None,
                 reduced_homology=True, n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.homology_dimensions = homology_dimensions
        self.collapse_edges = collapse_edges
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.n_jobs = n_jobs

    def _ripser_diagram(self, X):
        Xdgms = ripser(
            X, maxdim=self._max_homology_dimension,
            thresh=self.max_edge_length, coeff=self.coeff, metric=self.metric,
            metric_params=self.metric_params,
            collapse_edges=self.collapse_edges
            )["dgms"]

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``"precomputed"``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays/sparse matrices.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``"precomputed"``, then:

                - Diagonal entries indicate vertex weights, i.e. the filtration
                  parameters at which vertices appear.
                - If entries of `X` are dense, only their upper diagonal
                  portions (including the diagonal) are considered.
                - If entries of `X` are sparse, they do not need to be upper
                  diagonal or symmetric. If only one of entry (i, j) and (j, i)
                  is stored, its value is taken as the weight of the undirected
                  edge {i, j}. If both are stored, the value in the upper
                  diagonal is taken. Off-diagonal entries which are not
                  explicitly stored are treated as infinite, indicating absent
                  edges.
                - Entries of `X` should be compatible with a filtration, i.e.
                  the value at index (i, j) should be no smaller than the
                  values at diagonal indices (i, i) and (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        self._is_precomputed = self.metric == "precomputed"
        check_point_clouds(X, accept_sparse=True,
                           distance_matrices=self._is_precomputed)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]

        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``"precomputed"``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays/sparse matrices.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``"precomputed"``, then:

                - Diagonal entries indicate vertex weights, i.e. the filtration
                  parameters at which vertices appear.
                - If entries of `X` are dense, only their upper diagonal
                  portions (including the diagonal) are considered.
                - If entries of `X` are sparse, they do not need to be upper
                  diagonal or symmetric. If only one of entry (i, j) and (j, i)
                  is stored, its value is taken as the weight of the undirected
                  edge {i, j}. If both are stored, the value in the upper
                  diagonal is taken. Off-diagonal entries which are not
                  explicitly stored are treated as infinite, indicating absent
                  edges.
                - Entries of `X` should be compatible with a filtration, i.e.
                  the value at index (i, j) should be no smaller than the
                  values at diagonal indices (i, i) and (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True,
                               distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ripser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(
            Xt, "ripser", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
            )
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class WeightedRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`weighted Vietoris–Rips filtrations <TODO>` as in [3]_.

    Given a :ref:`point cloud <distance_matrices_and_point_clouds>` in
    Euclidean space, an abstract :ref:`metric space
    <distance_matrices_and_point_clouds>` encoded by a distance matrix, or the
    adjacency matrix of a weighted undirected graph, information about the
    appearance and disappearance of topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimensions
    and at different scales is summarised in the corresponding persistence
    diagram.

    Weighted (Vietoris–)Rips filtrations can be useful to highlight topological
    features against outliers and noise. Among them, the distance-to-measure
    (DTM) filtration is particularly suited to point clouds due to several
    favourable properties. This implementation follows the general framework
    described in [3]_. The idea is that, starting from a way to compute vertex
    weights :math:`\\{w_i\\}_i` from an input point cloud/distance
    matrix/adjacency matrix, a modified adjacency matrix is determined whose
    diagonal entries are the :math:`\\{w_i\\}_i`, and whose edge weights are

    .. math:: w_{ij} = \\begin{cases} \\max\\{ w_i, w_j \\} &\\text{if }
       2\\mathrm{dist}_{ij} \\leq |w_i^p - w_j^p|^{\\frac{1}{p}}, \\\\
       t &\\text{otherwise} \\end{cases}

    where :math:`t` is the only positive root of

    .. math:: 2 \\mathrm{dist}_{ij} = (t^p - w_i^p)^\\frac{1}{p} +
       (t^p - w_j^p)^\\frac{1}{p}

    and :math:`p` is a parameter (see `metric_params`). The modified adjacency
    matrices are then treated exactly as in :class:`VietorisRipsPersistence`.

    **Important notes**:

        - Vertex and edge weights are twice the ones in [3]_ so that the same
          results as :class:`VietorisRipsPersistence` are obtained when all
          vertex weights are zero.
        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.

    Parameters
    ----------
    metric : string or callable, optional, default: ``"euclidean"``
        If set to ``"precomputed"``, input data is to be interpreted as a
        collection of distance matrices or of adjacency matrices of weighted
        undirected graphs. Otherwise, input data is to be interpreted as a
        collection of point clouds (i.e. feature arrays), and `metric`
        determines a rule with which to calculate distances between pairs of
        points (i.e. row vectors). If `metric` is a string, it must be one of
        the options allowed by :func:`scipy.spatial.distance.pdist` for its
        metric parameter, or a metric listed in
        :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`, including
        ``"euclidean"``, ``"manhattan"`` or ``"cosine"``. If `metric` is a
        callable, it should take pairs of vectors (1D arrays) as input and, for
        each two vectors in a pair, it should return a scalar indicating the
        distance/dissimilarity between them.

    metric_params : dict, optional, default: ``{}``
        Additional parameters to be passed to the distance function.

    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    weights : ``"DTM"`` or callable, optional, default: ``"DTM"``
        Function that will be applied to each input point cloud/distance
        matrix/adjacency matrix to compute a 1D array of vertex weights for the
        the modified adjacency matrices. The default ``"DTM"`` denotes the
        empirical distance-to-measure function defined, following [3]_, by

        .. math:: w(x) = 2\\left(\\frac{1}{n+1} \\sum_{k=1}^n
           \\mathrm{dist}(x, x_k)^r\\right)^{1/r}.

        Here, :math:`\\mathrm{dist}` is the distance metric used, :math:`x_k`
        is the :math:`k`-th :math:`\\mathrm{dist}`-nearest neighbour of
        :math:`x` (:math:`x` is not considered a neighbour of itself),
        :math:`n` is the number of nearest neighbors to include, and :math:`r`
        is a parameter (see `weight_params`). If a callable, it must return
        non-negative 1D arrays.

    weight_params : dict, optional, default: ``{}``
        Additional parameters for the weighted filtration. ``"p"`` determines
        the power to be used in computing edge weights from vertex weights. It
        can be one of ``1``, ``2`` or ``np.inf`` and defaults to ``1``. If
        `weights` is ``"DTM"``, the additional keys ``"r"`` (default: ``2``)
        and ``"n_neighbors"`` (default: ``3``) are available (see `weights`,
        where the latter corresponds to :math:`n`).

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    collapse_edges : bool, optional, default: ``False``
        Whether to run the edge collapse algorithm in [2]_ prior to the
        persistent homology computation (see the Notes). Can reduce the runtime
        dramatically when the data or the maximum homology dimensions are
        large.

    max_edge_weight : float, optional, default: ``numpy.inf``
        Maximum value of the filtration parameter in the modified adjacency
        matrix. Edges with weight greater than this value will be considered
        absent.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_weight`. ``None`` means that this death
        value is declared to be equal to `max_edge_weight`.

    reduced_homology : bool, optional, default: ``True``
       If ``True``, the earliest-born triple in homology dimension 0 which has
       infinite death is discarded from each diagram computed in
       :meth:`transform`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_weight`.

    effective_weight_params_ : dict
        Effective parameters involved in computing the weighted Rips
        filtration.

    See also
    --------
    VietorisRipsPersistence, SparseRipsPersistence, FlagserPersistence,
    WeakAlphaPersistence, EuclideanCechPersistence, ConsistentRescaling,
    ConsecutiveRescaling

    Notes
    -----
    `giotto-ph <https://github.com/giotto-ai/giotto-ph>`_ [1]_ is used as a C++
    backend for computing Vietoris–Rips persistent homology and edge collapses.

    References
    ----------
    .. [1] J. Burella Pérez et al, "giotto-ph: A Python Library for
           High-Performance Computation of Persistent Homology of Vietoris–Rips
           Filtrations", 2021; `arXiv:2107.05412
           <https://arxiv.org/abs/2107.05412>`_.

    .. [2] J.-D. Boissonnat and S. Pritam, "Edge Collapse and Persistence of
           Flag Complexes"; in *36th International Symposium on Computational
           Geometry (SoCG 2020)*, pp. 19:1–19:15,
           Schloss Dagstuhl-Leibniz–Zentrum für Informatik, 2020;
           `DOI: 10.4230/LIPIcs.SoCG.2020.19
           <https://doi.org/10.4230/LIPIcs.SoCG.2020.19>`_.

    .. [3] H. Anai et al, "DTM-Based Filtrations"; in *Topological Data
           Analysis* (Abel Symposia, vol 15), Springer, 2020;
           `DOI: 10.1007/978-3-030-43408-3_2
           <https://doi.org/10.1007/978-3-030-43408-3_2>`_.

    """

    _hyperparameters = {
        "metric": {"type": (str, Callable)},
        "metric_params": {"type": dict},
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
            },
        "weights": {"type": (str, Callable)},
        "weight_params": {"type": dict},
        "collapse_edges": {"type": bool},
        "coeff": {"type": int, "in": Interval(2, np.inf, closed="left")},
        "max_edge_weight": {"type": Real},
        "infinity_values": {"type": (Real, type(None))},
        "reduced_homology": {"type": bool}
        }

    def __init__(self, metric="euclidean", metric_params={},
                 homology_dimensions=(0, 1), weights="DTM", weight_params={},
                 collapse_edges=False, coeff=2, max_edge_weight=np.inf,
                 infinity_values=None, reduced_homology=True, n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.homology_dimensions = homology_dimensions
        self.weights = weights
        self.weight_params = weight_params
        self.collapse_edges = collapse_edges
        self.coeff = coeff
        self.max_edge_weight = max_edge_weight
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.n_jobs = n_jobs

    def _ripser_diagram(self, X):
        if isinstance(self.weights, Callable):
            weights = self.weights(X)
        else:
            weights = self.weights
        Xdgms = ripser(
            X, maxdim=self._max_homology_dimension,
            thresh=self.max_edge_weight, coeff=self.coeff, metric=self.metric,
            metric_params=self.metric_params, weights=weights,
            weight_params=self.effective_weight_params_,
            collapse_edges=self.collapse_edges
            )["dgms"]

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``"precomputed"``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays/sparse matrices.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``"precomputed"``, then:

                - All entries of `X` should not contain infinities or negative
                  values (contrary to :class:`VietorisRipsPersistence`).
                - The diagonals of entries of `X` are ignored (after the vertex
                  weights are computed, when `weights` is a callable).
                - If entries of `X` are dense, only their upper diagonal
                  portions are considered.
                - If entries of `X` are sparse, they do not need to be upper
                  diagonal or symmetric. If only one of entry (i, j) and (j, i)
                  is stored, its value is taken as the weight of the undirected
                  edge {i, j}. If both are stored, the value in the upper
                  diagonal is taken. Off-diagonal entries which are not
                  explicitly stored are treated as infinite, indicating absent
                  edges.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])
        if isinstance(self.weights, str) and self.weights != "DTM":
            raise ValueError(f"'{self.weights}' passed for `weights` but the "
                             f"only allowed string is 'DTM'.")
        self.effective_weight_params_ = {"p": 1}
        if self.weights == "DTM":
            key = "DTM"
            self.effective_weight_params_.update({"n_neighbors": 3, "r": 2})
        else:
            key = "general"
        if self.weight_params:
            self.effective_weight_params_.update(self.weight_params)
            validate_params(self.effective_weight_params_,
                            _AVAILABLE_RIPS_WEIGHTS[key])

        self._is_precomputed = self.metric == "precomputed"
        check_point_clouds(X, accept_sparse=True, force_all_finite=True,
                           distance_matrices=self._is_precomputed)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_weight
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]

        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``"precomputed"``, and of distance matrices or
            adjacency matrices of weighted undirected graphs otherwise. Can be
            either a 3D ndarray whose zeroth dimension has size ``n_samples``,
            or a list containing ``n_samples`` 2D ndarrays/sparse matrices.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``"precomputed"``, then:

                - All entries of `X` should not contain infinities or negative
                  values (contrary to :class:`VietorisRipsPersistence`).
                - The diagonals of entries of `X` are ignored (after the vertex
                  weights are computed, when `weights` is a callable).
                - If entries of `X` are dense, only their upper diagonal
                  portions are considered.
                - If entries of `X` are sparse, they do not need to be upper
                  diagonal or symmetric. If only one of entry (i, j) and (j, i)
                  is stored, its value is taken as the weight of the undirected
                  edge {i, j}. If both are stored, the value in the upper
                  diagonal is taken. Off-diagonal entries which are not
                  explicitly stored are treated as infinite, indicating absent
                  edges.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True, force_all_finite=True,
                               distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._ripser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(
            Xt, "ripser", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
            )
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class SparseRipsPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`Sparse Vietoris–Rips filtrations
    <vietoris-rips_complex_and_vietoris-rips_persistence>`.

    Given a :ref:`point cloud <distance_matrices_and_point_clouds>` in
    Euclidean space, or an abstract :ref:`metric space
    <distance_matrices_and_point_clouds>` encoded by a distance matrix,
    information about the appearance and disappearance of topological features
    (technically, :ref:`homology classes <homology_and_cohomology>`) of various
    dimensions and at different scales is summarised in the corresponding
    persistence diagram.

    **Important note**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.

    Parameters
    ----------
    metric : string or callable, optional, default: ``"euclidean"``
        If set to ``"precomputed"``, input data is to be interpreted as a
        collection of distance matrices. Otherwise, input data is to be
        interpreted as a collection of point clouds (i.e. feature arrays), and
        `metric` determines a rule with which to calculate distances between
        pairs of instances (i.e. rows) in these arrays. If `metric` is a
        string, it must be one of the options allowed by
        :func:`scipy.spatial.distance.pdist` for its metric parameter, or a
        metric listed in :obj:`sklearn.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`,
        including "euclidean", "manhattan", or "cosine". If `metric` is a
        callable, it is called on each pair of instances and the resulting
        value recorded. The callable should take two arrays from the entry in
        `X` as input, and return a value indicating the distance between them.

    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    epsilon : float between 0. and 1., optional, default: ``0.1``
        Parameter controlling the approximation to the exact Vietoris–Rips
        filtration. If set to `0.`, :class:`SparseRipsPersistence` leads to the
        same results as :class:`VietorisRipsPersistence` but is slower.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Sparse Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge,
        and topological features at scales larger than this value will not be
        detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    reduced_homology : bool, optional, default: ``True``
       If ``True``, the earliest-born triple in homology dimension 0 which has
       infinite death is discarded from each diagram computed in
       :meth:`transform`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`. Set in :meth:`fit`.

    See also
    --------
    VietorisRipsPersistence, WeightedRipsPersistence, FlagserPersistence,
    WeakAlphaPersistence, EuclideanCechPersistence, ConsistentRescaling,
    ConsecutiveRescaling

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing sparse Vietoris–Rips persistent homology [1]_. Python
    bindings were modified for performance.

    References
    ----------
    .. [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference
           Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent__\
           cohomology.html>`_.

    """

    _hyperparameters = {
        "metric": {"type": (str, Callable)},
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
            },
        "coeff": {"type": int, "in": Interval(2, np.inf, closed="left")},
        "epsilon": {"type": Real, "in": Interval(0, 1, closed="both")},
        "max_edge_length": {"type": Real},
        "infinity_values": {"type": (Real, type(None))},
        "reduced_homology": {"type": bool}
        }

    def __init__(self, metric="euclidean", homology_dimensions=(0, 1),
                 coeff=2, epsilon=0.1, max_edge_length=np.inf,
                 infinity_values=None, reduced_homology=True, n_jobs=None):
        self.metric = metric
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.epsilon = epsilon
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        Xdgm = pairwise_distances(X, metric=self.metric)
        sparse_rips_complex = SparseRipsComplex(
            distance_matrix=Xdgm, max_edge_length=self.max_edge_length,
            sparse=self.epsilon
            )
        simplex_tree = sparse_rips_complex.create_simplex_tree(
            max_dimension=max(self._homology_dimensions) + 1
            )
        Xdgm = simplex_tree.persistence(
            homology_coeff_field=self.coeff, min_persistence=0
            )

        return Xdgm

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``"precomputed"``, and of distance matrices
            otherwise. Can be either a 3D ndarray whose zeroth dimension has
            size ``n_samples``, or a list containing ``n_samples`` 2D ndarrays.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``"precomputed"``, each entry of `X` should be
            compatible with a filtration, i.e. the value at index (i, j) should
            be no smaller than the values at diagonal indices (i, i) and
            (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])
        self._is_precomputed = self.metric == "precomputed"
        check_point_clouds(X, accept_sparse=True,
                           distance_matrices=self._is_precomputed)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]
        return self

    def transform(self, X, y=None):
        """For each point cloud or distance matrix in `X`, compute the
        relevant persistence diagram as an array of triples [b, d, q]. Each
        triple represents a persistent topological feature in dimension q
        (belonging to `homology_dimensions`) which is born at b and dies at d.
        Only triples in which b < d are meaningful. Triples in which b and d
        are equal ("diagonal elements") may be artificially introduced during
        the computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds if `metric`
            was not set to ``"precomputed"``, and of distance matrices
            otherwise. Can be either a 3D ndarray whose zeroth dimension has
            size ``n_samples``, or a list containing ``n_samples`` 2D ndarrays.
            Point cloud arrays have shape ``(n_points, n_dimensions)``, and if
            `X` is a list these shapes can vary between point clouds. If
            `metric` was set to ``"precomputed"``, each entry of `X` should be
            compatible with a filtration, i.e. the value at index (i, j) should
            be no smaller than the values at diagonal indices (i, i) and
            (j, j).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True,
                               distance_matrices=self._is_precomputed)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._gudhi_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(
            Xt, "gudhi", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
            )
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class WeakAlphaPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`weak alpha filtrations <TODO>`.

    Given a :ref:`point cloud <distance_matrices_and_point_clouds>` in
    Euclidean space, information about the appearance and disappearance of
    topological features (technically, :ref:`homology classes
    <homology_and_cohomology>`) of various dimensions and at different scales
    is summarised in the corresponding persistence diagram.

    The weak alpha filtration of a point cloud is defined to be the
    :ref:`Vietoris–Rips filtration
    <vietoris-rips_complex_and_vietoris-rips_persistence>` of the sparse matrix
    of Euclidean distances between neighbouring vertices in the Delaunay
    triangulation of the point cloud. In low dimensions, computing the
    persistent homology of this filtration can be much faster than computing
    Vietoris–Rips persistent homology via :class:`VietorisRipsPersistence`.

    **Important note**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Vietoris–Rips filtration parameter. Points whose
        distance is greater than this value will never be connected by an edge,
        and topological features at scales larger than this value will not be
        detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    reduced_homology : bool, optional, default: ``True``
       If ``True``, the earliest-born triple in homology dimension 0 which has
       infinite death is discarded from each diagram computed in
       :meth:`transform`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    VietorisRipsPersistence, WeightedRipsPersistence, SparseRipsPersistence,
    FlagserPersistence, EuclideanCechPersistence

    Notes
    -----
    Delaunay triangulation are computed by :class:`scipy.spatial.Delaunay`.
    `giotto-ph <https://github.com/giotto-ai/giotto-ph>`_ [1]_ is used as a C++
    backend for computing Vietoris–Rips persistent homology.

    References
    ----------
    .. [1] J. Burella Pérez et al, "giotto-ph: A Python Library for
           High-Performance Computation of Persistent Homology of Vietoris–Rips
           Filtrations", 2021; `arXiv:2107.05412
           <https://arxiv.org/abs/2107.05412>`_.

    """

    _hyperparameters = {
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
            },
        "coeff": {"type": int, "in": Interval(2, np.inf, closed="left")},
        "max_edge_length": {"type": Real},
        "infinity_values": {"type": (Real, type(None))},
        "reduced_homology": {"type": bool}
        }

    def __init__(self, homology_dimensions=(0, 1), coeff=2,
                 max_edge_length=np.inf, infinity_values=None,
                 reduced_homology=True, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.n_jobs = n_jobs

    def _weak_alpha_diagram(self, X):
        # `indices` will serve as the array of column indices
        indptr, indices = Delaunay(X).vertex_neighbor_vertices

        # Compute the array of row indices
        row = np.zeros_like(indices)
        row[indptr[1:-1]] = 1
        np.cumsum(row, out=row)

        # We only need the upper diagonal
        mask = indices > row
        row, col = row[mask], indices[mask]
        dists = np.linalg.norm(X[row] - X[col], axis=1)
        # Note: passing the shape explicitly should not be needed in more
        # recent versions of C++ ripser
        n_points = len(X)
        dm = coo_matrix((dists, (row, col)), shape=(n_points, n_points))

        Xdgms = ripser(dm, maxdim=self._max_homology_dimension,
                       thresh=self.max_edge_length, coeff=self.coeff,
                       metric="precomputed")["dgms"]

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])
        check_point_clouds(X)

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]

        return self

    def transform(self, X, y=None):
        """For each point cloud in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._weak_alpha_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(
            Xt, "ripser", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
            )
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class EuclideanCechPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    `Cech filtrations <cech_complex_and_cech_persistence>`_.

    Given a :ref:`point cloud <distance_matrices_and_point_clouds>` in
    Euclidean space, information about the appearance and disappearance of
    topological features (technically, :ref:`homology classes
    <homology_and_cohomology>`) of various dimensions and at different scales
    is summarised in the corresponding persistence diagram.

    **Important note**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_length : float, optional, default: ``numpy.inf``
        Maximum value of the Cech filtration parameter. Topological features at
        scales larger than this value will not be detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_length`. ``None`` means that this death
        value is declared to be equal to `max_edge_length`.

    reduced_homology : bool, optional, default: ``True``
       If ``True``, the earliest-born triple in homology dimension 0 which has
       infinite death is discarded in :meth:`transform`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_length`.

    See also
    --------
    VietorisRipsPersistence, FlagserPersistence, SparseRipsPersistence,
    WeakAlphaPersistence

    Notes
    -----
    `GUDHI <https://github.com/GUDHI/gudhi-devel>`_ is used as a C++ backend
    for computing Cech persistent homology [1]_. Python bindings were modified
    for performance.

    References
    ----------
    .. [1] C. Maria, "Persistent Cohomology", 2020; `GUDHI User and Reference
           Manual <http://gudhi.gforge.inria.fr/doc/3.1.0/group__persistent__\
           cohomology.html>`_.

    """

    _hyperparameters = {
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
            },
        "coeff": {"type": int, "in": Interval(2, np.inf, closed="left")},
        "max_edge_length": {"type": Real,
                            "in": Interval(0, np.inf, closed="right")},
        "infinity_values": {"type": (Real, type(None)),
                            "in": Interval(0, np.inf, closed="neither")},
        "reduced_homology": {"type": bool}
        }

    def __init__(self, homology_dimensions=(0, 1), coeff=2,
                 max_edge_length=np.inf, infinity_values=None,
                 reduced_homology=True, n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.coeff = coeff
        self.max_edge_length = max_edge_length
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.n_jobs = n_jobs

    def _gudhi_diagram(self, X):
        cech_complex = CechComplex(points=X, max_radius=self.max_edge_length)
        simplex_tree = cech_complex.create_simplex_tree(
            max_dimension=max(self._homology_dimensions) + 1
            )
        Xdgm = simplex_tree.persistence(homology_coeff_field=self.coeff,
                                        min_persistence=0)

        return Xdgm

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_point_clouds(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs"])

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_length
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._max_homology_dimension = self._homology_dimensions[-1]

        return self

    def transform(self, X, y=None):
        """For each point cloud in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input data representing a collection of point clouds. Can be either
            a 3D ndarray whose zeroth dimension has size ``n_samples``, or a
            list containing ``n_samples`` 2D ndarrays. Point cloud arrays have
            shape ``(n_points, n_dimensions)``, and if `X` is a list these
            shapes can vary between point clouds.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays in
            `X`. ``n_features`` equals :math:`\\sum_q n_q`, where :math:`n_q`
            is the maximum number of topological features in dimension
            :math:`q` across all samples in `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(self._gudhi_diagram)(x)
                                          for x in X)

        Xt = _postprocess_diagrams(
            Xt, "gudhi", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
            )
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )


@adapt_fit_transform_docs
class FlagserPersistence(BaseEstimator, TransformerMixin, PlotterMixin):
    """:ref:`Persistence diagrams <persistence_diagram>` resulting from
    :ref:`filtrations <filtered_complex>` of :ref:`directed or undirected flag
    complexes <clique_and_flag_complexes>` [1]_.

    Given a weighted directed or undirected graph, information about the
    appearance and disappearance of topological features (technically,
    :ref:`homology classes <homology_and_cohomology>`) of various dimension and
    at different scales is summarised in the corresponding persistence diagram.

    **Important note**:

        - Persistence diagrams produced by this class must be interpreted with
          care due to the presence of padding triples which carry no
          information. See :meth:`transform` for additional information.

    Parameters
    ----------
    homology_dimensions : list or tuple, optional, default: ``(0, 1)``
        Dimensions (non-negative integers) of the topological features to be
        detected.

    directed : bool, optional, default: ``True``
        If ``True``, :meth:`transform` computes the persistence diagrams of the
        filtered directed flag complexes arising from the input collection of
        weighted directed graphs. If ``False``, :meth:`transform` computes the
        persistence diagrams of the filtered undirected flag complexes obtained
        by regarding all input weighted graphs as undirected, and:

        - if `max_edge_weight` is ``numpy.inf``, it is sufficient to pass a
          collection of (dense or sparse) upper-triangular matrices;
        - if `max_edge_weight` is finite, it is recommended to pass either a
          collection of symmetric dense matrices, or a collection of sparse
          upper-triangular matrices.

    filtration : string, optional, default: ``"max"``
        Algorithm determining the filtration values of higher order simplices
        from the weights of the vertices and edges. Possible values are:
        ["dimension", "zero", "max", "max3", "max_plus_one", "product", "sum",
        "pmean", "pmoment", "remove_edges", "vertex_degree"]

    coeff : int prime, optional, default: ``2``
        Compute homology with coefficients in the prime field
        :math:`\\mathbb{F}_p = \\{ 0, \\ldots, p - 1 \\}` where :math:`p`
        equals `coeff`.

    max_edge_weight : float, optional, default: ``numpy.inf``
        Maximum edge weight to be considered in the filtration. All edge
        weights greater than this value will be considered as absent from the
        filtration and topological features at scales larger than this value
        will not be detected.

    infinity_values : float or None, default: ``None``
        Which death value to assign to features which are still alive at
        filtration value `max_edge_weight`. ``None`` means that this death
        value is declared to be equal to `max_edge_weight`.

    reduced_homology : bool, optional, default: ``True``
       If ``True``, the earliest-born triple in homology dimension 0 which has
       infinite death is discarded from each diagram computed in
       :meth:`transform`.

    max_entries : int, optional, default: ``-1``
        Number controlling the degree of precision in the matrix reductions
        performed by the the backend. Corresponds to the parameter
        ``approximation`` in :func:`pyflagser.flagser_weighted` and
        :func:`pyflagser.flagser_unweighted`. Increase for higher precision,
        decrease for faster computation. A good value is often ``100000`` in
        hard problems. A negative value computes highest possible precision.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    infinity_values_ : float
        Effective death value to assign to features which are still alive at
        filtration value `max_edge_weight`.

    See also
    --------
    VietorisRipsPersistence, WeightedRipsPersistence, SparseRipsPersistence,
    WeakAlphaPersistence, EuclideanCechPersistence, ConsistentRescaling,
    ConsecutiveRescaling

    Notes
    -----
    The `pyflagser <https://github.com/giotto-ai/pyflagser>`_ Python package
    is used for binding `Flagser <https://github.com/luetge/flagser>`_, a C++
    backend for computing the (persistent) homology of (filtered) directed
    flag complexes. For more details, please refer to the `flagser \
    documentation <https://github.com/luetge/flagser/blob/master/docs/\
    documentation_flagser.pdf>`_.

    References
    ----------
    .. [1] D. Luetgehetmann, D. Govc, J. P. Smith, and R. Levi, "Computing
           persistent homology of directed flag complexes", *Algorithms*,
           13(1), 2020.

    """

    _hyperparameters = {
        "homology_dimensions": {
            "type": (list, tuple),
            "of": {"type": int, "in": Interval(0, np.inf, closed="left")}
            },
        "directed": {"type": bool},
        "coeff": {"type": int, "in": Interval(2, np.inf, closed="left")},
        "max_edge_weight": {"type": Real},
        "infinity_values": {"type": (Real, type(None))},
        "reduced_homology": {"type": bool},
        "max_entries": {"type": int}
        }

    def __init__(self, homology_dimensions=(0, 1), directed=True,
                 filtration="max", coeff=2, max_edge_weight=np.inf,
                 infinity_values=None, reduced_homology=True, max_entries=-1,
                 n_jobs=None):
        self.homology_dimensions = homology_dimensions
        self.directed = directed
        self.filtration = filtration
        self.coeff = coeff
        self.max_edge_weight = max_edge_weight
        self.infinity_values = infinity_values
        self.reduced_homology = reduced_homology
        self.max_entries = max_entries
        self.n_jobs = n_jobs

    def _flagser_diagram(self, X):
        Xdgms = [np.empty((0, 2), dtype=float)] * self._min_homology_dimension
        Xdgms += flagser_weighted(X, max_edge_weight=self.max_edge_weight,
                                  min_dimension=self._min_homology_dimension,
                                  max_dimension=self._max_homology_dimension,
                                  directed=self.directed,
                                  filtration=self.filtration, coeff=self.coeff,
                                  approximation=self.max_entries)["dgms"]
        n_missing_dims = self._max_homology_dimension + 1 - len(Xdgms)
        if n_missing_dims:
            Xdgms += [np.empty((0, 2), dtype=float)] * n_missing_dims

        return Xdgms

    def fit(self, X, y=None):
        """Calculate :attr:`infinity_values_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input collection of adjacency matrices of weighted directed or
            undirected graphs. Can be either a 3D ndarray whose zeroth
            dimension has size ``n_samples``, or a list containing
            ``n_samples`` 2D ndarrays/sparse matrices. In each adjacency
            matrix, diagonal elements are vertex weights and off-diagonal
            elements are edge weights. It is assumed that a vertex weight
            cannot be larger than the weight of the edges it forms. The way
            zero values are handled depends on the format of the matrix. If the
            matrix is a dense ``numpy.ndarray``, zero values denote
            zero-weighted edges. If the matrix is a sparse ``scipy.sparse``
            matrix, explicitly stored off-diagonal zeros and all diagonal zeros
            denote zero-weighted edges. Off-diagonal values that have not been
            explicitly stored are treated by ``scipy.sparse`` as zeros but will
            be understood as infinitely-valued edges, i.e., edges absent from
            the filtration.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_point_clouds(X, accept_sparse=True, distance_matrices=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=["n_jobs",
                                                               "filtration"])

        if self.infinity_values is None:
            self.infinity_values_ = self.max_edge_weight
        else:
            self.infinity_values_ = self.infinity_values

        self._homology_dimensions = sorted(self.homology_dimensions)
        self._min_homology_dimension = self._homology_dimensions[0]
        self._max_homology_dimension = self._homology_dimensions[-1]

        return self

    def transform(self, X, y=None):
        """For each adjacency matrix in `X`, compute the relevant persistence
        diagram as an array of triples [b, d, q]. Each triple represents a
        persistent topological feature in dimension q (belonging to
        `homology_dimensions`) which is born at b and dies at d. Only triples
        in which b < d are meaningful. Triples in which b and d are equal
        ("diagonal elements") may be artificially introduced during the
        computation for padding purposes, since the number of non-trivial
        persistent topological features is typically not constant across
        samples. They carry no information and hence should be effectively
        ignored by any further computation.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input collection of adjacency matrices of weighted directed or
            undirected graphs. Can be either a 3D ndarray whose zeroth
            dimension has size ``n_samples``, or a list containing
            ``n_samples`` 2D ndarrays/sparse matrices. In each adjacency
            matrix, diagonal elements are vertex weights and off-diagonal
            elements are edges weights. It is assumed that a vertex weight
            cannot be larger than the weight of the edges it forms. The way
            zero values are handled depends on the format of the matrix. If
            the matrix is a dense ``numpy.ndarray``, zero values denote
            zero-weighted edges. If the matrix is a sparse ``scipy.sparse``
            matrix, explicitly stored off-diagonal zeros and all diagonal zeros
            denote zero-weighted edges. Off-diagonal values that have not been
            explicitly stored are treated by ``scipy.sparse`` as zeros but will
            be understood as infinitely-valued edges, i.e., edges absent from
            the filtration.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Array of persistence diagrams computed from the feature arrays or
            distance matrices in `X`. ``n_features`` equals
            :math:`\\sum_q n_q`, where :math:`n_q` is the maximum number of
            topological features in dimension :math:`q` across all samples in
            `X`.

        """
        check_is_fitted(self)
        X = check_point_clouds(X, accept_sparse=True, distance_matrices=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._flagser_diagram)(x) for x in X)

        Xt = _postprocess_diagrams(
            Xt, "flagser", self._homology_dimensions, self.infinity_values_,
            self.reduced_homology
            )
        return Xt

    @staticmethod
    def plot(Xt, sample=0, homology_dimensions=None, plotly_params=None):
        """Plot a sample from a collection of persistence diagrams, with
        homology in multiple dimensions.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_features, 3)
            Collection of persistence diagrams, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

        homology_dimensions : list, tuple or None, optional, default: ``None``
            Which homology dimensions to include in the plot. ``None`` means
            plotting all dimensions present in ``Xt[sample]``.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"traces"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_diagram(
            Xt[sample], homology_dimensions=homology_dimensions,
            plotly_params=plotly_params
            )
