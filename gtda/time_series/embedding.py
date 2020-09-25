"""Time series embedding."""
# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array, column_or_1d

from ._utils import _time_delay_embedding, _mutual_information, \
    _false_nearest_neighbors
from ..base import TransformerResamplerMixin, PlotterMixin
from ..plotting import plot_point_cloud
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params, check_collection

_TAKENS_EMBEDDING_HYPERPARAMETERS = {
        'time_delay': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'dimension': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'stride': {'type': int, 'in': Interval(1, np.inf, closed='left')}
        }


def takens_embedding_optimal_parameters(X, max_time_delay, max_dimension,
                                        stride=1, n_jobs=None, validate=True):
    """Compute the "optimal" parameters for a Takens (time-delay) embedding
    [1]_ of a univariate time series.

    First, an optimal time delay is found by minimising the time-delayed mutual
    information among values no greater than `max_time_delay`. Then, a
    heuristic based on an algorithm in [2]_ is used to select an embedding
    dimension which, when increased, does not reveal a large proportion of
    "false nearest neighbors".

    Parameters
    ----------
    X : ndarray of shape (n_samples,) or (n_samples, 1)
        Input data representing a single univariate time series.

    max_time_delay : int, required
        Maximum time delay between two consecutive values for constructing one
        embedded point.

    max_dimension : int, required
        Maximum embedding dimension that will be considered in the
        optimization.

    stride : int, optional, default: ``1``
        Stride duration between two consecutive embedded points. It defaults to
        1 as this is the usual value in the statement of Takens's embedding
        theorem.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    validate : bool, optional, default: ``True``
        Whether the input and hyperparameters should be validated.

    Returns
    -------
    time_delay : int
        The "optimal" time delay less than or equal to `max_dimension`, as
        determined by minimizing the time-delayed mutual information.

    dimension : int
        The "optimal" embedding dimension less than or equal to
        `max_dimension`, as determined by a false nearest neighbors heuristic
        once `time_delay` is computed.

    See also
    --------
    SingleTakensEmbedding, TakensEmbedding, SlidingWindow

    References
    ----------
    .. [1] F. Takens, "Detecting strange attractors in turbulence". In: Rand
           D., Young LS. (eds) *Dynamical Systems and Turbulence, Warwick
           1980*. Lecture Notes in Mathematics, vol. 898. Springer, 1981;
           `DOI: 10.1007/BFb0091924 <https://doi.org/10.1007/BFb0091924>`_.

    .. [2] M. B. Kennel, R. Brown, and H. D. I. Abarbanel, "Determining
           embedding dimension for phase-space reconstruction using a
           geometrical construction"; *Phys. Rev. A* **45**, pp. 3403--3411,
           1992; `DOI: 10.1103/PhysRevA.45.3403
           <https://doi.org/10.1103/PhysRevA.45.3403>`_.

    """
    if validate:
        _hyperparameters = _TAKENS_EMBEDDING_HYPERPARAMETERS.copy()
        validate_params({'validate': validate}, {'validate': {'type': bool}})
        validate_params({'time_delay': max_time_delay,
                         'dimension': max_dimension, 'stride': stride},
                        _hyperparameters)
        X = column_or_1d(X)

    mutual_information_list = Parallel(n_jobs=n_jobs)(
        delayed(_mutual_information)(X, time_delay, n_bins=100)
        for time_delay in range(1, max_time_delay + 1))
    time_delay = \
        mutual_information_list.index(min(mutual_information_list)) + 1

    n_false_nbhrs_list = Parallel(n_jobs=n_jobs)(
        delayed(_false_nearest_neighbors)(X, time_delay, dim, stride=stride)
        for dim in range(1, max_dimension + 3))
    variation_list = [np.abs(n_false_nbhrs_list[dim - 1]
                             - 2 * n_false_nbhrs_list[dim] +
                             n_false_nbhrs_list[dim + 1])
                      / (n_false_nbhrs_list[dim] + 1) / dim
                      for dim in range(2, max_dimension + 1)]
    dimension = variation_list.index(min(variation_list)) + 2

    return time_delay, dimension


@adapt_fit_transform_docs
class SlidingWindow(BaseEstimator, TransformerResamplerMixin):
    """Sliding windows onto the data.

    Useful in time series analysis to convert a sequence of objects (scalar or
    array-like) into a sequence of windows on the original sequence. Each
    window stacks together consecutive objects, and consecutive windows are
    separated by a constant stride.

    Parameters
    ----------
    size : int, optional, default: ``10``
        Size of each sliding window.

    stride : int, optional, default: ``1``
        Stride between consecutive windows.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.time_series import SlidingWindow
    >>> # Create a time series of two-dimensional vectors, and a corresponding
    >>> # time series of scalars
    >>> X = np.arange(20).reshape(-1, 2)
    >>> y = np.arange(10)
    >>> windows = SlidingWindow(size=3, stride=3)
    >>> # Fit and transform X
    >>> X_windows = windows.fit_transform(X)
    >>> print(X_windows)
    [[[ 2  3]
      [ 4  5]
      [ 6  7]]
     [[ 8  9]
      [10 11]
      [12 13]]
     [[14 15]
      [16 17]
      [18 19]]]
    >>> # Resample y
    >>> yr = windows.resample(y)
    >>> print(yr)
    [3 6 9]

    See also
    --------
    SingleTakensEmbedding, TakensEmbedding

    Notes
    -----
    The current implementation favours the last entry over the first one, in
    the sense that the last entry of the last window always equals the last
    entry in the original time series. Hence, a number of initial entries
    (depending on the remainder of the division between ``n_samples - size``
    and ``stride``) may be lost.

    """

    _hyperparameters = {
        'size': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'stride': {'type': int, 'in': Interval(1, np.inf, closed='left')}
        }

    def __init__(self, size=10, stride=1):
        self.size = size
        self.stride = stride

    def _window_indices(self, X):
        n_samples = X.shape[0]
        n_windows, offset = divmod(n_samples - self.size, self.stride)
        n_windows += 1
        if n_windows <= 0:
            raise ValueError(
                f"Number of samples ({n_samples}) cannot be less than window "
                f"size ({self.size})."
                )
        indices = np.tile(np.arange(self.size), (n_windows, 1))
        indices += np.arange(n_windows)[:, None] * self.stride + offset
        return indices

    def slice_windows(self, X):
        indices = self._window_indices(X)
        return indices[:, [0, -1]] + np.array([0, 1])

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        self

        """
        check_array(X, ensure_2d=False, allow_nd=True)
        validate_params(self.get_params(), self._hyperparameters)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Slide windows over X.

        Parameters
        ----------
        X : ndarray of shape (n_samples, ...)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        Xt : ndarray of shape (n_windows, size, ...)
            Windows of consecutive entries of the original time series.
            ``n_windows = (n_samples - size) // stride  + 1``.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_array(X, ensure_2d=False, allow_nd=True)

        window_indices = self._window_indices(Xt)

        Xt = Xt[window_indices]
        return Xt

    def resample(self, y, X=None):
        """Resample `y` so that, for any i > 0, the minus i-th entry of the
        resampled vector corresponds in time to the last entry of the minus
        i-th window produced by :meth:`transform`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target.

        X : None
            There is no need for input data, yet the pipeline API requires this
            parameter.

        Returns
        -------
        yr : ndarray of shape (n_samples_new,)
            The resampled target. ``n_samples_new = (n_samples - size)
            // stride + 1``.

        """
        check_is_fitted(self, '_is_fitted')
        yr = column_or_1d(y)

        yr = yr[:self.size - 2:-self.stride][::-1]
        return yr


@adapt_fit_transform_docs
class SingleTakensEmbedding(BaseEstimator, TransformerResamplerMixin):
    """Representation of a single univariate time series as a point cloud.

    Based on a time-delay embedding technique named after F. Takens [1]_ [2]_.
    Given a discrete time series :math:`(X_0, X_1, \\ldots)` and a sequence of
    evenly sampled times :math:`t_0, t_1, \\ldots`, one extracts a set of
    :math:`d`-dimensional vectors of the form :math:`(X_{t_i}, X_{t_i + \\tau},
    \\ldots , X_{t_i + (d-1)\\tau})` for :math:`i = 0, 1, \\ldots`. This set is
    called the :ref:`Takens embedding <takens_embedding>` of the time series
    and can be interpreted as a point cloud.

    The difference between :math:`t_{i+1}` and :math:`t_i` is called the
    stride, :math:`\\tau` is called the time delay, and :math:`d` is called the
    (embedding) dimension.

    If :math:`d` and :math:`\\tau` are not explicitly set, suitable values are
    searched for during :meth:`fit` [3]_ [4]_.

    To compute time-delay embeddings of several time series simultaneously, use
    :class:`TakensEmbedding` instead.

    Parameters
    ----------
    parameters_type : ``'search'`` | ``'fixed'``, optional, default: \
        ``'search'``
        If set to ``'fixed'``, the values of `time_delay` and `dimension` are
        used directly in :meth:`transform`. If set to ``'search'``,
        :func:`takens_embedding_optimal_parameter` is run in :meth:`fit` to
        estimate optimal values for these quantities and store them as
        :attr:`time_delay_` and :attr:`dimension_`.

    time_delay : int, optional, default: ``1``
        Time delay between two consecutive values for constructing one embedded
        point. If `parameters_type` is ``'search'``, it corresponds to the
        maximum time delay that will be considered.

    dimension : int, optional, default: ``5``
        Dimension of the embedding space. If `parameters_type` is ``'search'``,
        it corresponds to the maximum embedding dimension that will be
        considered.

    stride : int, optional, default: ``1``
        Stride duration between two consecutive embedded points. It defaults to
        1 as this is the usual value in the statement of Takens's embedding
        theorem.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    time_delay_ : int
        Actual time delay used to embed. If
        `parameters_type` is ``'search'``, it is the calculated optimal time
        delay and is less than or equal to `time_delay`. Otherwise it is equal
        to `time_delay`.

    dimension_ : int
        Actual embedding dimension used to embed. If `parameters_type` is
        ``'search'``, it is the calculated optimal embedding dimension and is
        less than or equal to `dimension`. Otherwise it is equal to
        `dimension`.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.time_series import SingleTakensEmbedding
    >>> # Create a noisy signal
    >>> rng = np.random.default_rng()
    >>> n_samples = 10000
    >>> signal = np.asarray([np.sin(x / 50) + 0.5 * rng.random()
    ...                      for x in range(n_samples)])
    >>> # Set up the transformer
    >>> STE = SingleTakensEmbedding(parameters_type='search', dimension=5,
    ...                             time_delay=5, n_jobs=-1)
    >>> # Fit and transform
    >>> signal_embedded = STE.fit_transform(signal)
    >>> print('Optimal time delay based on mutual information:',
    ...       STE.time_delay_)
    Optimal time delay based on mutual information: 5
    >>> print('Optimal embedding dimension based on false nearest neighbors:',
    ...       STE.dimension_)
    Optimal embedding dimension based on false nearest neighbors: 2
    >>> print(signal_embedded.shape)
    (9995, 2)

    See also
    --------
    TakensEmbedding, SlidingWindow, takens_embedding_optimal_parameters

    Notes
    -----
    The current implementation favours the last value over the first one, in
    the sense that the last coordinate of the last vector in a Takens embedded
    time series always equals the last value in the original time series.
    Hence, a number of initial values (depending on the remainder of the
    division between ``n_samples - dimension * (time_delay - 1) - 1`` and the
    stride) may be lost.

    References
    ----------
    .. [1] F. Takens, "Detecting strange attractors in turbulence". In: Rand
           D., Young LS. (eds) *Dynamical Systems and Turbulence, Warwick
           1980*. Lecture Notes in Mathematics, vol. 898. Springer, 1981;
           `DOI: 10.1007/BFb0091924 <https://doi.org/10.1007/BFb0091924>`_.

    .. [2] J. A. Perea and J. Harer, "Sliding Windows and Persistence: An \
           Application of Topological Methods to Signal Analysis"; \
           *Foundations of Computational Mathematics*, **15**, \
            pp. 799--838; `DOI: 10.1007/s10208-014-9206-z
           <https://doi.org/10.1007/s10208-014-9206-z>`_.

    .. [3] M. B. Kennel, R. Brown, and H. D. I. Abarbanel, "Determining
           embedding dimension for phase-space reconstruction using a
           geometrical construction"; *Phys. Rev. A* **45**, pp. 3403--3411,
           1992; `DOI: 10.1103/PhysRevA.45.3403
           <https://doi.org/10.1103/PhysRevA.45.3403>`_.

    .. [4] N. Sanderson, "Topological Data Analysis of Time Series using
           Witness Complexes"; PhD thesis, University of Colorado at
           Boulder, 2018; `https://scholar.colorado.edu/math_gradetds/67
           <https://scholar.colorado.edu/math_gradetds/67>`_.

    """

    _hyperparameters = _TAKENS_EMBEDDING_HYPERPARAMETERS.copy()
    _hyperparameters['parameters_type'] = \
        {'type': str, 'in': ['fixed', 'search']}

    def __init__(self, parameters_type='search', time_delay=1, dimension=5,
                 stride=1, n_jobs=None):
        self.parameters_type = parameters_type
        self.time_delay = time_delay
        self.dimension = dimension
        self.stride = stride
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """If necessary, compute the optimal time delay and embedding
        dimension. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Input data.

        y : None
            There is no need for a target, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object

        """
        X = column_or_1d(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.parameters_type == 'search':
            self.time_delay_, self.dimension_ = \
                takens_embedding_optimal_parameters(
                    X, self.time_delay, self.dimension, stride=self.stride,
                    n_jobs=self.n_jobs, validate=False
                    )
        else:
            self.time_delay_ = self.time_delay
            self.dimension_ = self.dimension

        return self

    def transform(self, X, y=None):
        """Compute the Takens embedding of `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Input data.

        y : None
            Ignored.

        Returns
        -------
        Xt : ndarray of shape (n_points, n_dimensions)
            Output point cloud in Euclidean space of dimension given by
            :attr:`dimension_`. ``n_points = (n_samples - time_delay *
            (dimension - 1) - 1) // stride + 1``.

        """
        check_is_fitted(self)
        Xt = column_or_1d(X).copy()

        Xt = _time_delay_embedding(
            Xt, time_delay=self.time_delay_, dimension=self.dimension_,
            stride=self.stride
            )

        return Xt

    def resample(self, y, X=None):
        """Resample `y` so that, for any i > 0, the minus i-th entry of the
        resampled vector corresponds in time to the last coordinate of the
        minus i-th embedding vector produced by :meth:`transform`.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target.

        X : None
            There is no need for input data, yet the pipeline API requires this
            parameter.

        Returns
        -------
        yr : ndarray of shape (n_samples_new,)
            The resampled target. ``n_samples_new = (n_samples - time_delay *
            (dimension - 1) - 1) // stride + 1``.

        """
        check_is_fitted(self)
        yr = column_or_1d(y)

        final_index = self.time_delay_ * (self.dimension_ - 1)
        yr = yr[:final_index - 1:-self.stride][::-1]
        return yr


@adapt_fit_transform_docs
class TakensEmbedding(BaseEstimator, TransformerMixin, PlotterMixin):
    """Point clouds from collections of time series via independent Takens
    embeddings.

    This transformer takes collections of (possibly multivariate) time series
    as input, applies the Takens embedding algorithm described in
    :class:`SingleTakensEmbedding` to each independently, and returns a
    corresponding collection of point clouds in Euclidean space (or possibly
    higher-dimensional structures, see `flatten`).

    Parameters
    ----------
    time_delay : int, optional, default: ``1``
        Time delay between two consecutive values for constructing one embedded
        point.

    dimension : int, optional, default: ``2``
        Dimension of the embedding space (per variable, in the multivariate
        case).

    stride : int, optional, default: ``1``
        Stride duration between two consecutive embedded points.

    flatten : bool, optional, default: ``True``
        Only relevant when the input of :meth:`transform` represents a
        collection of multivariate or tensor-valued time series. If ``True``,
        ensures that the output is a 3D ndarray or list of 2D arrays. If
        ``False``, each entry of the input collection leads to an array of
        dimension one higher than the entry's dimension. See Examples.

    ensure_last_value : bool, optional, default: ``True``
        Whether the value(s) representing the last measurement(s) must be
        be present in the output as the last coordinate(s) of the last
        embedding vector(s). If ``False``, the first measurement(s) is (are)
        present as the 0-th coordinate(s) of the 0-th vector(s) instead.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.time_series import TakensEmbedding

    Two univariate time series of duration 4:

    >>> X = np.arange(8).reshape(2, 4)
    >>> print(X)
    [[0 1 2 3]
     [4 5 6 7]]
    >>> TE = TakensEmbedding(time_delay=1, dimension=2)
    >>> print(TE.fit_transform(X))
    [[[0 1]
      [1 2]
      [2 3]]
     [[5 6]
      [6 7]
      [7 8]]]

    Two multivariate time series of duration 4, with 2 variables:

    >>> x = np.arange(8).reshape(2, 1, 4)
    >>> X = np.concatenate([x, -x], axis=1)
    >>> print(X)
    [[[ 0  1  2  3]
      [ 0 -1 -2 -3]]
     [[ 4  5  6  7]
      [-4 -5 -6 -7]]]

    Pass `flatten` as ``True`` (default):

    >>> TE = TakensEmbedding(time_delay=1, dimension=2, flatten=True)
    >>> print(TE.fit_transform(X))
    [[[ 0  1  0 -1]
      [ 1  2 -1 -2]
      [ 2  3 -2 -3]]
     [[ 4  5 -4 -5]
      [ 5  6 -5 -6]
      [ 6  7 -6 -7]]]

    Pass `flatten` as ``False``:

    >>> TE = TakensEmbedding(time_delay=1, dimension=2, flatten=False)
    >>> print(TE.fit_transform(X))
    [[[[ 0  1]
       [ 1  2]
       [ 2  3]]
      [[ 0 -1]
       [-1 -2]
       [-2 -3]]]
     [[[ 4  5]
       [ 5  6]
       [ 6  7]]
      [[-4 -5]
       [-5 -6]
       [-6 -7]]]]

    See also
    --------
    SingleTakensEmbedding, SlidingWindow, takens_embedding_optimal_parameters

    Notes
    -----
    To compute the Takens embedding of a single univariate time series in the
    form of a 1D array or column vector, use :class:`SingleTakensEmbedding`
    instead.

    Unlike :class:`SingleTakensEmbedding`, this transformer does not include
    heuristics to optimize the choice of time delay and embedding dimension.
    The function :func:`takens_embedding_optimal_parameters` is specifically
    dedicated to this task, but only on a single univariate time series.

    If dealing with a forecasting problem on a single time series, this
    transformer can be used after an instance of :class:`SlidingWindow` and
    before an instance of a homology transformer, to produce topological
    features from sliding windows over the time series.

    """

    _hyperparameters = _TAKENS_EMBEDDING_HYPERPARAMETERS.copy()
    _hyperparameters.update({'flatten': {'type': bool},
                             'ensure_last_value': {'type': bool}})

    def __init__(self, time_delay=1, dimension=2, stride=1, flatten=True,
                 ensure_last_value=True):
        self.time_delay = time_delay
        self.dimension = dimension
        self.stride = stride
        self.flatten = flatten
        self.ensure_last_value = ensure_last_value

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input collection of time series. A 2D array or list of 1D arrays is
            interpreted as a collection of univariate time series. A 3D array
            or list of 2D arrays is interpreted as a collection of multivariate
            time series, each with shape ``(n_variables, n_timestamps)``. More
            generally, :math`N`-dimensional arrays or lists of
            (:math`N-1`)-dimensional arrays (:math:`N \\geq 3`) are interpreted
            as collections of tensor-valued time series, each with time indexed
            by the last axis.

        y : None
            There is no need for a target, yet the pipeline API requires this
            parameter.

        Returns
        -------
        self : object

        """
        check_collection(X, copy=False)
        validate_params(self.get_params(), self._hyperparameters)
        self._is_fitted = True

        return self

    def transform(self, X, y=None):
        """Compute the Takens embedding of each entry in `X`.

        Parameters
        ----------
        X : ndarray or list of length n_samples
            Input collection of time series. A 2D array or list of 1D arrays is
            interpreted as a collection of univariate time series. A 3D array
            or list of 2D arrays is interpreted as a collection of multivariate
            time series, each with shape ``(n_variables, n_timestamps)``. More
            generally, :math`N`-dimensional arrays or lists of
            (:math`N-1`)-dimensional arrays (:math:`N \\geq 3`) are interpreted
            as collections of tensor-valued time series, each with time indexed
            by the last axis.

        y : None
            Ignored.

        Returns
        -------
        Xt : ndarray or list of length n_samples
            The result of performing a Takens embedding of each entry in `X`
            with the given parameters. If `X` is a 2D array or a list of 1D
            arrays, `Xt` is a 3D array or a list of 2D arrays (respectively),
            each entry of which has shape ``(n_points, dimension)`` where
            ``n_points = (n_timestamps - time_delay * (dimension - 1) - 1) // \
            stride + 1``. If `X` is an :math`N`-dimensional array or a list of
            (:math`N-1`)-dimensional arrays (:math:`N \\geq 3`), the output
            shapes depend on the `flatten` parameter:

                - if `flatten` is ``True``, `Xt` is still a 3D array or a
                  list of 2D arrays (respectively), each entry of which has
                  shape ``(n_points, dimension * n_variables)`` where
                  ``n_points`` is as above and ``n_variables`` is the product
                  of the sizes of all axes in said entry except the last.
                - if `flatten` is ``False``, `Xt` is an
                  (:math`N+1`)-dimensional array or list of
                  :math`N`-dimensional arrays.

        """
        check_is_fitted(self, '_is_fitted')
        Xt = check_collection(X, copy=True)

        Xt = _time_delay_embedding(
            Xt, time_delay=self.time_delay, dimension=self.dimension,
            stride=self.stride, flatten=self.flatten,
            ensure_last_value=self.ensure_last_value
            )

        return Xt

    @staticmethod
    def plot(Xt, sample=0, plotly_params=None):
        """Plot a sample from a collection of Takens embeddings of time series,
        as a point cloud in 2D or 3D. If points in the window have more than
        three dimensions, only the first three are plotted.

        Parameters
        ----------
        Xt : ndarray or list of length n_samples
            Collection of point clouds, such as returned by :meth:`transform`.

        sample : int, optional, default: ``0``
            Index of the sample in `Xt` to be plotted.

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
        return plot_point_cloud(Xt[sample], plotly_params=plotly_params)
