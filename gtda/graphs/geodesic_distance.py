"""Graph geodesic distance calculations."""
# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.graph_shortest_path import graph_shortest_path
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

    The graphs are encoded as sparse adjacency matrices, while the outputs
    are dense distance matrices of variable size.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.graphs import TransitionGraph, GraphGeodesicDistance
    >>> X = np.arange(4).reshape(1, -1, 1)
    >>> tg = TransitionGraph(func=None).fit_transform(X)
    >>> print(tg[0].toarray())
    [[False  True False False]
     [ True False  True False]
     [False  True False  True]
     [False False  True False]]
    >>> ggd = GraphGeodesicDistance().fit_transform(tg)
    >>> print(ggd[0])
    [[0. 1. 2. 3.]
     [1. 0. 1. 2.]
     [2. 1. 0. 1.]
     [3. 2. 1. 0.]]

    See also
    --------
    TransitionGraph, KNeighborsGraph, gtda.homology.VietorisRipsPersistence

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _geodesic_distance(self, X):
        X_distance = graph_shortest_path(X)
        X_distance[X_distance == 0] = np.inf  # graph_shortest_path returns a
        # float64 array, so inserting np.inf does not change the type.
        np.fill_diagonal(X_distance, 0)
        return X_distance

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or \
            (n_samples, n_vertices, n_vertices)
            Input data, i.e. a collection of adjacency matrices of graphs.
            Each adjacency matrix may be a dense or a sparse array.

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
        """Use :meth:`sklearn.utils.graph_shortest_path.graph_shortest_path`
        to compute the lengths of graph shortest paths between any two
        vertices.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or \
            (n_samples, n_vertices, n_vertices)
            Input data, i.e. a collection of adjacency matrices of graphs.
            Each adjacency matrix may be a dense or sparse array.

        y : None
            Ignored.

        Returns
        -------
        Xt : ndarray of shape (n_samples,) or \
             (n_samples, n_vertices, n_vertices)
            Array of distance matrices. If the distance matrices have variable
            size across samples, `Xt` is a one-dimensional array of dense
            arrays.

        """
        check_is_fitted(self, '_is_fitted')
        X = check_graph(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._geodesic_distance)(x) for x in X)
        Xt = np.array(Xt)
        return Xt

    @staticmethod
    def plot(Xt, sample=0, colorscale='blues'):
        """Plot a sample from a collection of distance matrices.

        Parameters
        ----------
        Xt : ndarray of shape (n_samples, n_points, n_points)
            Collection of distance matrices, such as returned by
            :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample to be plotted.

        colorscale : str, optional, default: ``'blues'``
            Color scale to be used in the heat map. Can be anything allowed by
            :class:`plotly.graph_objects.Heatmap`.

        """
        return plot_heatmap(Xt[sample], colorscale=colorscale)
