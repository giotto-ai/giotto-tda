"""Covering schemes for one or several dimensions."""
# License: GNU AGPLv3

import warnings
from functools import partial
from itertools import product

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.exceptions import DataDimensionalityWarning, NotFittedError
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from .utils._cover import _check_has_one_column, \
    _remove_empty_and_duplicate_intervals
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class OneDimensionalCover(BaseEstimator, TransformerMixin):
    """Cover of one-dimensional data coming from open overlapping intervals.

    In :meth:`fit`, given a training array `X` representing a collection of
    real numbers, a cover of the real line by open intervals
    :math:`I_k = (a_k, b_k)` (:math:`k = 1, \\ldots, n`,
    :math:`a_k < a_{k+1}`, :math:`b_k < b_{k+1}`) is constructed
    based on the distribution of values in `X`. In :meth:`transform`,
    the cover is applied to a new array `X'` to yield a cover of `X'`.

    All covers constructed in :meth:`fit` have :math:`a_1 = -\\infty`
    and :math:`b_n = + \\infty``. Two kinds of cover are currently available:
    "uniform" and "balanced". A uniform cover is such that
    :math:`b_1 - m = b_2 - a_2 = \\cdots = M - a_n` where :math:`m` and
    :math:`M` are the minimum and maximum values in `X` respectively. A
    balanced cover is such that approximately the same number of unique
    values from `X` is contained in each cover interval.

    Parameters
    ----------
    kind : ``'uniform'`` | ``'balanced'``, optional, default: ``'uniform'``
        The kind of cover to use.

    n_intervals : int, optional, default: ``10``
        The number of intervals in the cover calculated in :meth:`fit`.

    overlap_frac : float, optional, default: ``0.1``
        If the cover is uniform, this is the ratio between the length of the
        intersection between consecutive intervals and the length of each
        interval. If the cover is balanced, this is the analogous fractional
        overlap for a uniform cover of the closed interval
        :math:`(0.5, N + 0.5)` where :math:`N` is the number of unique
        values in the training array (see the Notes).

    Attributes
    ----------
    left_limits_ : ndarray of shape (n_intervals,)
        Left limits of the cover intervals computed in :meth:`fit`. See the
        Notes.

    right_limits_ : ndarray of shape (n_intervals,)
        Right limits of the cover intervals computed in :meth:`fit`. See the
        Notes.

    Notes
    -----
    In the case of a balanced cover, :meth:`left_limits_` and
    :meth:`right_limits_` are computed as follows given a training array `X`:
    first, entries in `X` are ranked in ascending order, starting at 1 and
    with the same rank repeated in the case of equal values; then, the closed
    interval :math:`(0.5, N + 0.5)`, where :math:`N` is the maximum
    rank observed, is covered uniformly with parameters `n_intervals` and
    `overlap_frac`, yielding intervals :math:`(\\alpha_k, \\beta_k)`;
    the final cover is made of intervals :math:`(a_k, b_k)` where, for
    :math:`k > 1` (resp. :math:`k < ` `n_intervals`), :math:`a_k` (resp.
    :math:`b_k`) is the value of any entry in `X` ranked as the floor (
    resp. ceiling) of :math:`\\alpha_k` (resp. :math:`\\beta_k`).

    See also
    --------
    CubicalCover

    """

    _hyperparameters = {
        'kind': {'type': str, 'in': ['uniform', 'balanced']},
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')}
        }

    def __init__(self, kind='uniform', n_intervals=10, overlap_frac=0.1):
        self.kind = kind
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def _fit_uniform(self, X):
        self.left_limits_, self.right_limits_ = self._find_interval_limits(
            X, self.n_intervals, self.overlap_frac, is_uniform=True)
        return self

    def _fit_balanced(self, X):
        X_rank = rankdata(X, method='dense') - 1
        left_limits, right_limits = self._find_interval_limits(
            X_rank, self.n_intervals, self.overlap_frac, is_uniform=False)
        left_limits_int = left_limits.astype(int)
        left_ranks = np.where(left_limits >= 0, left_limits_int, -1)
        right_limits_int = right_limits.astype(int)
        right_ranks = np.where(right_limits_int == right_limits,
                               right_limits_int,
                               right_limits_int + 1)
        self.left_limits_, self.right_limits_ = self._limits_from_ranks(
            X_rank, X.flatten(), left_ranks, right_ranks)
        return self

    def fit(self, X, y=None):
        """Compute all cover interval limits according to `X` and store them
        in :attr:`left_limits_` and :attr:`right_limits_`. Then, return the
        estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)
        if self.overlap_frac <= 1e-8:
            warnings.warn("`overlap_frac` is close to zero, "
                          "which might cause numerical issues and errors.",
                          RuntimeWarning)

        if X.ndim == 2:
            _check_has_one_column(X)

        is_uniform = self.kind == 'uniform'
        fitter = self._fit_uniform if is_uniform else self._fit_balanced
        return fitter(X)

    def _transform(self, X):
        return np.logical_and(X > self.left_limits_, X < self.right_limits_)

    def transform(self, X, y=None):
        """Compute a cover of `X` according to the cover of the real line
        computed in :meth:`fit`, and return it as a two-dimensional boolean
        array. Each column indicates the location of entries in `X`
        belonging to a common cover interval.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_cover_sets)
            Encoding of the cover of `X` as a boolean array. In general,
            ``n_cover_sets`` is less than or equal to `n_intervals` as empty
            or duplicated cover sets are removed.

        """
        check_is_fitted(self)
        Xt = check_array(X, ensure_2d=False)

        if Xt.ndim == 2:
            _check_has_one_column(Xt)
        else:
            Xt = Xt[:, None]

        if self.kind == 'balanced':
            # Test whether self.left_limits_ and self.right_limits_ have
            # been created -- to catch cases in which transform is run after
            # fit_transform but not after fit.
            self._check_limit_attrs()

        Xt = self._transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt

    def _fit_transform_balanced(self, X):
        """Shortcut in the case of a balanced cover, avoiding overhead
        from calculation of self.left_limits_ and self.right_limits_.

        Stores hidden attributes _left_limits and _right_limits which refer
        to a cover of the interval (-0.5, n_unique - 0.5) where n_unique is
        the number of unique points in X.

        """
        X_rank = rankdata(X, method='dense') - 1
        self._left_limits, self._right_limits = self._find_interval_limits(
            X_rank, self.n_intervals, self.overlap_frac, is_uniform=False)
        X_rank = np.broadcast_to(X_rank[:, None],
                                 (X.shape[0], self.n_intervals))
        Xt = np.logical_and(X_rank > self._left_limits,
                            X_rank < self._right_limits)
        return Xt

    def _fit_transform(self, X):
        if self.kind == 'uniform':
            Xt = self._fit_uniform(X)._transform(X)
        else:
            Xt = self._fit_transform_balanced(X)
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to the data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples,) or (n_samples, 1)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_cover_sets)
            Encoding of the cover of `X` as a boolean array. In general,
            ``n_cover_sets`` is less than or equal to `n_intervals` as empty
            or duplicated cover sets are removed.

        """
        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        if Xt.ndim == 2:
            _check_has_one_column(Xt)
        else:
            Xt = Xt[:, None]

        Xt = self._fit_transform(Xt)
        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt

    def get_fitted_intervals(self):
        """Returns the open intervals computed in :meth:`fit`, as a list of
        tuples (a, b) where a < b.

        """
        check_is_fitted(self)
        if self.kind == 'balanced':
            # Test whether self.left_limits_ and self.right_limits_ have
            # been created
            self._check_limit_attrs()
        return list(zip(self.left_limits_, self.right_limits_))

    def _check_limit_attrs(self):
        limit_attrs = ['left_limits_', 'right_limits_']
        has_limits = all([hasattr(self, attr) for attr in limit_attrs])
        if not has_limits:
            raise NotFittedError(
                "When the cover is balanced and n_intervals > 1, the left "
                "and right limits of the cover intervals are not "
                "explicitly calculated during 'fit_transform'. Please "
                "call 'fit' explicitly on the same data before using this "
                "method.")

    def _find_interval_limits(self, X, n_intervals, overlap_frac,
                              is_uniform=True):
        if is_uniform:
            min_val, max_val = np.min(X), np.max(X)
            only_one_pt = (min_val == max_val)
        else:
            # Assume X is the result of a call to scipy.stats.rankdata
            min_val, max_val = -0.5, np.max(X) + 0.5
            only_one_pt = (min_val == max_val - 1)

        # Allow X to have one unique sample only if one interval is required,
        # in which case the fitted interval will be (-np.inf, np.inf).
        if only_one_pt and n_intervals > 1:
            raise ValueError(
                f"Only one unique filter value found, cannot fit "
                f"{n_intervals} > 1 intervals.")

        left_limits, right_limits = \
            self._cover_limits(min_val, max_val, n_intervals, overlap_frac)
        if is_uniform:
            left_limits[0], right_limits[-1] = -np.inf, np.inf
        return left_limits, right_limits

    def _limits_from_ranks(self, X_rank, X, left_ranks, right_ranks):
        n_intervals = self.n_intervals
        X_rank = np.broadcast_to(X_rank[:, None],
                                 (X_rank.shape[0], n_intervals))
        left_mask = (X_rank == left_ranks)
        right_mask = (X_rank == right_ranks)
        left_indices = (np.flatnonzero(left_mask[:, i])
                        for i in range(n_intervals))
        right_indices = (np.flatnonzero(right_mask[:, i])
                         for i in range(n_intervals))
        left_limits = np.array([
            X[nonzero_indices[0]] if nonzero_indices.size else -np.inf
            for nonzero_indices in left_indices
            ])
        right_limits = np.array([
            X[nonzero_indices[0]] if nonzero_indices.size else np.inf
            for nonzero_indices in right_indices
            ])
        left_limits[0] = -np.inf
        right_limits[-1] = np.inf
        return left_limits, right_limits

    @staticmethod
    def _cover_limits(min_val, max_val, n_intervals, overlap_frac):
        # Construct a uniform cover of the interval [min_val, max_val].
        # Let the length of each interval be l. The equation to solve for l is
        # (n_intervals - 1) * l * (1 - overlap_frac) + l = max_val - min_val.
        # The maximum left endpoint is at min_val + (n_intervals - 1) * (1 -
        # overlap_frac) * l
        total_len = max_val - min_val
        interval_len = total_len / \
            (n_intervals - (n_intervals - 1) * overlap_frac)

        last = min_val + (n_intervals - 1) * (1 - overlap_frac) * interval_len
        left_limits = np.linspace(min_val, last, num=n_intervals,
                                  endpoint=True)
        right_limits = left_limits + interval_len
        return left_limits, right_limits


@adapt_fit_transform_docs
class CubicalCover(BaseEstimator, TransformerMixin):
    """Cover of multi-dimensional data coming from overlapping hypercubes
    (technically, parallelopipeds) given by taking products of one-dimensional
    intervals.

    In :meth:`fit`, :class:`OneDimensionalCover` objects are fitted
    independently on each column of the input array, according to the same
    parameters passed to the constructor. For example, if the
    :class:`CubicalCover` object is instantiated with ``kind='uniform'``,
    ``n_intervals=10`` and ``overlap_frac=0.1``, then each column of the
    input array is used to construct a cover of the real line by 10
    equal-length intervals with fractional overlap of 0.1. Each element of the
    resulting multi-dimensional cover of Euclidean space is of the form
    :math:`I_{i, \\ldots, k} = I^{(0)}_i \\times \\cdots \\times
    I^{(d-1)}_k` where :math:`d` is the number of columns in the input
    array, and :math:`I^{(l)}_j` is the :math:`j`th cover interval
    constructed for feature dimension :math:`l`. In :meth:`transform`,
    the cover is applied to a new array `X'` to yield a cover of `X'`.

    Parameters
    ----------
    kind : ``'uniform'`` | ``'balanced'``, optional, default: ``'uniform'``
        The kind of cover to use.

    n_intervals : int, optional, default: ``10``
        The number of intervals in the covers of each feature dimension
        calculated in :meth:`fit`.

    overlap_frac : float, optional, default: ``0.1``
        The fractional overlap between consecutive intervals in the covers of
        each feature dimension calculated in :meth:`fit`.

    See also
    --------
    OneDimensionalCover

    """

    _hyperparameters = {
        'kind': {'type': str, 'in': ['uniform', 'balanced']},
        'n_intervals': {'type': int, 'in': Interval(1, np.inf, closed='left')},
        'overlap_frac': {'type': float, 'in': Interval(0, 1, closed='neither')}
        }

    def __init__(self, kind='uniform', n_intervals=10, overlap_frac=0.1):
        self.kind = kind
        self.n_intervals = n_intervals
        self.overlap_frac = overlap_frac

    def _clone_and_apply_to_column(self, X, coverer, method_name, i):
        # method is either a fit-type or a fit_transform-type method
        try:
            return getattr(clone(coverer), method_name)(X[:, [i]])
        except ValueError as ve:
            if ve.args[0] == f"Only one unique filter value found, cannot " \
                             f"fit {self.n_intervals} > 1 intervals.":
                raise ValueError(
                    f"Only one unique filter value found along feature "
                    f"dimension {i}, cannot fit {self.n_intervals} > 1 "
                    f"intervals there.")
            else:
                raise ve

    def _fit(self, X):
        coverer = OneDimensionalCover(kind=self.kind,
                                      n_intervals=self.n_intervals,
                                      overlap_frac=self.overlap_frac)
        is_uniform = self.kind == 'uniform'
        fitter = '_fit_uniform' if is_uniform else '_fit_balanced'
        self._coverers = [
            partial(self._clone_and_apply_to_column, X, coverer, fitter)(i)
            for i in range(X.shape[1])
            ]
        self._n_features_fit = X.shape[1]
        return self

    def fit(self, X, y=None):
        """Compute all open cover parallelopipeds according to `X`,
        as products of one-dimensional intervals covering each feature
        dimension separately. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        # Reshape filter function values derived from FunctionTransformer
        if X.ndim == 1:
            X = X[:, None]

        return self._fit(X)

    def _transform(self, X):
        # Calculate 1D cover for each column
        covers = [coverer._transform(X[:, [i]])
                  for i, coverer in enumerate(self._coverers)]

        Xt = self._combine_one_dim_covers(covers)
        return Xt

    def transform(self, X, y=None):
        """Compute a cover of `X` according to the cover of Euclidean space
        computed in :meth:`fit`, and return it as a two-dimensional boolean
        array whose each column indicates the location of entries in `X`
        belonging to a common cover interval.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_cover_sets)
            Encoding of the cover of `X` as a boolean array. In general,
            ``n_cover_sets`` is less than or equal to n_intervals *
            n_features` as empty or duplicated cover sets are removed.

        """
        check_is_fitted(self, '_coverers')
        Xt = check_array(X, ensure_2d=False)

        # Reshape filter function values derived from FunctionTransformer
        if Xt.ndim == 1:
            Xt = Xt[:, None]

        n_features_fit = self._n_features_fit
        n_features = Xt.shape[1]
        if n_features != n_features_fit:
            raise DataDimensionalityWarning(
                f"Different number of columns between `fit` ({n_features_fit})"
                f" and `transform` ({n_features}).")

        if self.kind == 'balanced':
            # Test on the first coverer whether the left_limits_ and
            # right_limits_ attributes are present
            self._coverers[0]._check_limit_attrs()

        Xt = self._transform(Xt)
        return Xt

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to the data, then transform it.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_cover_sets)
            Encoding of the cover of `X` as a boolean array. In general,
            ``n_cover_sets`` is less than or equal to `n_intervals *
            n_features` as empty or duplicated cover sets are removed.

        """
        Xt = check_array(X, ensure_2d=False)
        validate_params(self.get_params(), self._hyperparameters)

        # Reshape filter function values derived from FunctionTransformer
        if Xt.ndim == 1:
            Xt = Xt[:, None]

        if self.kind == 'uniform':
            Xt = self._fit(Xt)._transform(Xt)
            return Xt

        # Calculate 1D cover for each column
        coverer = OneDimensionalCover(kind=self.kind,
                                      n_intervals=self.n_intervals,
                                      overlap_frac=self.overlap_frac)
        coverers = [clone(coverer) for _ in range(Xt.shape[1])]
        fit_transformer = '_fit_transform_balanced'
        covers = [
            partial(self._clone_and_apply_to_column,
                    Xt, coverer, fit_transformer)(i)
            for i, coverer in enumerate(coverers)
            ]
        # Only store attributes if above succeeds
        self._coverers = coverers
        self._n_features_fit = Xt.shape[1]
        Xt = self._combine_one_dim_covers(covers)
        return Xt

    @staticmethod
    def _combine_one_dim_covers(covers):
        # Stack intervals for each cover
        intervals = (
            [cover[:, i] for i in range(cover.shape[1])] for cover in covers
            )

        # Calculate masks for pullback cover
        Xt = np.array([np.logical_and.reduce(t)
                       for t in product(*intervals)]).T

        Xt = _remove_empty_and_duplicate_intervals(Xt)
        return Xt
