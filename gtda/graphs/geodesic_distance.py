"""Graph geodesic distance calculations."""
# License: GNU AGPLv3

from functools import reduce
from operator import and_
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
from numpy.ma import masked_invalid
from numpy.ma.core import MaskedArray
from scipy.sparse import issparse, isspmatrix_csr
from scipy.sparse.csgraph import shortest_path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..base import PlotterMixin
from ..plotting import plot_heatmap
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import check_graph


@adapt_fit_transform_docs
class GraphGeodesicDistance(BaseEstimator, TransformerMixin, PlotterMixin):
    """Distance matrices arising from geodesic distances on graphs.

    For each (possibly weighted and/or directed) graph in a collection, this
    transformer calculates the length of the shortest (directed or undirected)
    path between any two of its vertices, setting it to ``numpy.inf`` when two
    vertices cannot be connected by a path.

    The graphs are represented by their adjacency matrices which can be dense
    arrays, sparse matrices or masked arrays. The following rules apply:

    - In dense arrays of Boolean type, entries which are ``False`` represent
      absent edges.
    - In dense arrays of integer or float type, zero entries represent edges
      of length 0. Absent edges must be indicated by ``numpy.inf``.
    - In sparse matrices, non-stored values represent absent edges. Explicitly
      stored zero or ``False`` edges represent edges of length 0.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    directed : bool, optional, default: ``True``
        If ``True`` (default), then find the shortest path on a directed graph.
        If ``False``, then find the shortest path on an undirected graph.

    unweighted : bool, optional, default: ``False``
        If ``True``, then find unweighted distances. That is, rather than
        finding the path between each point such that the sum of weights is
        minimized, find the path such that the number of edges is minimized.

    method : ``'auto'`` | ``'FW'`` | ``'D'`` | ``'BF'`` | ``'J'``, optional, \
        default: ``'auto'``
        Algorithm to use for shortest paths. See the `scipy documentation \
        <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.\
        csgraph.shortest_path.html>`_.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.graphs import TransitionGraph, GraphGeodesicDistance
    >>> X = np.arange(4).reshape(1, -1, 1)
    >>> X_tg = TransitionGraph(func=None).fit_transform(X)
    >>> print(X_tg[0].toarray())
    [[0 1 0 0]
     [0 0 1 0]
     [0 0 0 1]
     [0 0 0 0]]
    >>> X_ggd = GraphGeodesicDistance(directed=False).fit_transform(X_tg)
    >>> print(X_ggd[0])
    [[0. 1. 2. 3.]
     [1. 0. 1. 2.]
     [2. 1. 0. 1.]
     [3. 2. 1. 0.]]

    See also
    --------
    TransitionGraph, KNeighborsGraph

    """

    def __init__(self, n_jobs=None, directed=False, unweighted=False,
                 method='auto'):
        self.n_jobs = n_jobs
        self.directed = directed
        self.unweighted = unweighted
        self.method = method

    def _geodesic_distance(self, X, i=None):
        method_ = self.method
        if not issparse(X):
            diag = np.eye(X.shape[0], dtype=bool)
            if np.any(~np.logical_or(X, diag)):
                if self.method in ['auto', 'FW']:
                    if np.any(X < 0):
                        method_ = 'J'
                    else:
                        method_ = 'D'
                    warn(
                        f"Methods 'auto' and 'FW' are not supported when "
                        f"some edge weights are zero. Using '{method_}' "
                        f"instead for graph {i}."
                        )
            if not isinstance(X, MaskedArray):
                # Convert to a masked array with mask given by positions in
                # which infs or NaNs occur.
                if X.dtype != bool:
                    X = masked_invalid(X)
        elif X.shape[0] != X.shape[1]:
            n_vertices = max(X.shape)
            X = X.copy() if isspmatrix_csr(X) else X.tocsr()
            X.resize(n_vertices, n_vertices)

        return shortest_path(X, directed=self.directed,
                             unweighted=self.unweighted, method=method_)

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Input data: a collection of adjacency matrices of graphs. Each
            adjacency matrix may be a dense or a sparse array.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_graph(X)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Compute the lengths of graph shortest paths between any two
        vertices.

        Parameters
        ----------
        X : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Input data: a collection of ``n_samples`` adjacency matrices of
            graphs. Each adjacency matrix may be a dense array, a sparse
            matrix, or a masked array.

        y : None
            Ignored.

        Returns
        -------
        Xt : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Output collection of dense distance matrices. If the distance
            matrices all have the same shape, a single 3D ndarray is returned.

        """
        check_is_fitted(self, '_is_fitted')
        X = check_graph(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._geodesic_distance)(x, i=i) for i, x in enumerate(X))

        x0_shape = Xt[0].shape
        if reduce(and_, (x.shape == x0_shape for x in Xt), True):
            Xt = np.asarray(Xt)

        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='blues', plotly_params=None):
        """Plot a sample from a collection of distance matrices.

        Parameters
        ----------
        Xt : list of length n_samples, or ndarray of shape (n_samples, \
            n_vertices, n_vertices)
            Collection of distance matrices, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        colorscale : str, optional, default: ``'blues'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        plotly_params : dict or None, optional, default: ``None``
            Custom parameters to configure the plotly figure. Allowed keys are
            ``"trace"`` and ``"layout"``, and the corresponding values should
            be dictionaries containing keyword arguments as would be fed to the
            :meth:`update_traces` and :meth:`update_layout` methods of
            :class:`plotly.graph_objects.Figure`.

        Returns
        -------
        fig : :class:`plotly.graph_objects.Figure` object
            Plotly figure.

        """
        return plot_heatmap(
            Xt[sample], colorscale=colorscale,
            title=f"{sample}-th geodesic distance matrix",
            plotly_params=plotly_params
            )
