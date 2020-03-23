"""Construct transition graphs from dynamical systems."""
# License: GNU AGPLv3

from types import FunctionType
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy import sparse as sp
from scipy.sparse import SparseEfficiencyWarning
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params


def identity(x):
    """The identity function."""
    return x


@adapt_fit_transform_docs
class TransitionGraph(BaseEstimator, TransformerMixin):
    """Undirected transition graphs from arrays of time-evolving states.

    Let A be a two-dimensional array viewed as a time series (along the row
    axis) of one-dimensional arrays encoding the "state" of a system. The
    corresponding *undirected transition graph* (or *network*) has as vertex
    set the set of all unique states (rows) in A, and there is an edge
    between two vertices if and only if one of the corresponding states
    immediately follows the other in A.

    Given a collection of two-dimensional arrays, this transformer performs
    two tasks:

        1. Optionally, it preprocesses the arrays by applying a function row
           by row to them. This can be used e.g. as a "compression" step
           to reduce the size of the state space.
        2. It computes the undirected transition graph of each array as a
           sparse matrix.

    Parameters
    ----------
    func : None or callable, optional, default: ``numpy.argsort``
        If a callable, it is the function to be applied to each row of each
        array as a preprocessing steps. Allowed callables are functions
        mapping 1-D arrays to 1-D arrays of constant length, and must be
        compatible with :mod:`numpy.apply_along_axis`. If ``None``, this
        function is the identity (no preprocessing). The default is
        ``numpy.argsort``, which makes the final transition graphs
        *ordinal partition networks* [1]_ [2]_ [3]_.

    func_params : None or dict, optional, default: ``None``
        Additional keyword arguments for `func`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    effective_func_params_ : dict
        A copy of `func_params` if this was not set to ``None``, otherwise an
        empty dictionary.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.graphs import TransitionGraph
    >>> X = np.array([[[1, 0], [2, 3], [5, 4]],
    ...               [[5, 4], [5, 4], [5, 4]])
    >>> tg = TransitionGraph()
    >>> tg = tg.fit(X)
    >>> print(tg.transform(X)[0].toarray())
    [[0 1]
     [1 0]]
    >>> print(tg.transform(X)[1].toarray())
    [[1 0]
     [0 0]]

    See also
    --------
    GraphGeodesicDistance, gtda.time_series.TakensEmbedding

    Notes
    -----
    In general, the shapes of the sparse matrices output by :meth:`transform`
    will be different across samples, and the same row or column index will
    refer to different states in different samples.

    References
    ----------
    .. [1] M. Small, "Complex networks from time series: Capturing dynamics",
           *2013 IEEE International Symposium on Circuits and Systems
           (IS-CAS2013)*, 2013; doi: `10.1109/iscas.2013.6572389
           <http://dx.doi.org/10.1109/iscas.2013.6572389>`_.

    .. [2] M. McCullough, M. Small, T. Stemler, and H. Ho-Ching Iu, "Time
           lagged ordinal partition networks for capturing dynamics of
           continuous dynamical systems"; *Chaos: An Interdisciplinary
           Journal of Nonlinear Science* **25** (5), p. 053101, 2015;
           doi: `10.1063/1.4919075 <http://dx.doi.org/10.1063/1.4919075>`_.

    .. [3] A. Myers, E. Munch, and F. A. Khasawneh, "Persistent homology of
           complex networks for dynamic state detection"; *Phys. Rev. E*
           **100**, 022314, 2019; doi: `10.1103/PhysRevE.100.022314
           <http://dx.doi.org/10.1109/CVPR.2015.7299106>`_.

    """

    _hyperparameters = {
        'func': {'type': (FunctionType, type(None))},
        'func_params': {'type': (dict, type(None))}
    }

    def __init__(self, func=np.argsort, func_params=None, n_jobs=None):
        self.func = func
        self.func_params = func_params
        self.n_jobs = n_jobs

    def _make_adjacency_matrix(self, X):
        Xm = np.apply_along_axis(self._func, 1, X)
        Xm = np.unique(Xm, axis=0, return_inverse=True)[1]
        n_indices = 2 * (len(Xm) - 1)
        first = Xm[:-1]
        second = Xm[1:]
        Xm = sp.csr_matrix((np.full(n_indices, True),
                           (np.concatenate([first, second]),
                            np.concatenate([second, first]))))
        # Silence sparse warnings TODO: Benchmark
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', SparseEfficiencyWarning)
            sp.csr_matrix.setdiag(Xm, 0)
        return Xm

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_time_steps, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, allow_nd=True)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.func is None:
            self._func = identity
        else:
            self._func = self.func

        if self.func_params is None:
            self.effective_func_params_ = {}
        else:
            self.effective_func_params_ = self.func_params.copy()

        return self

    def transform(self, X, y=None):
        """Create transition graphs from the input data and return their
        adjacency matrices. The graphs are simple, undirected and
        unweighted, and the adjacency matrices are sparse matrices of type
        bool.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_time_steps, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : array of sparse boolean matrices, shape (n_samples,)
            The collection of ``n_samples`` transition graphs. Each transition
            graph is encoded by a sparse matrix of boolean type.

        """
        check_is_fitted(self)
        Xt = check_array(X, allow_nd=True)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._make_adjacency_matrix)(x) for x in Xt)
        Xt = np.asarray(Xt)
        return Xt
