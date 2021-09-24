"""Feature extraction from persistence diagrams."""
# License: GNU AGPLv3

from numbers import Real
from types import FunctionType

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from scipy.stats import entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import DistanceMetric
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted

from ._metrics import _AVAILABLE_AMPLITUDE_METRICS, _parallel_amplitude
from ._features import _AVAILABLE_POLYNOMIALS, _implemented_polynomial_recipes
from ._utils import _subdiagrams, _bin, _homology_dimensions_to_sorted_ints
from ..utils._docs import adapt_fit_transform_docs
from ..utils.intervals import Interval
from ..utils.validation import validate_params, check_diagrams


@adapt_fit_transform_docs
class PersistenceEntropy(BaseEstimator, TransformerMixin):
    """:ref:`Persistence entropies <persistence_entropy>` of persistence
    diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and their respective persistence entropies are
    calculated as the (base 2) Shannon entropies of the collections of
    differences d - b ("lifetimes"), normalized by the sum of all such
    differences. Optionally, these entropies can be normalized according to a
    simple heuristic, see `normalize`.

    **Important notes**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.
        - By default, persistence subdiagrams containing only triples with zero
          lifetime will have corresponding (normalized) entropies computed as
          ``numpy.nan``. To avoid this, set a value of `nan_fill_value`
          different from ``None``.

    Parameters
    ----------
    normalize : bool, optional, default: ``False``
        When ``True``, the persistence entropy of each diagram is normalized by
        the logarithm of the sum of lifetimes of all points in the diagram.
        Can aid comparison between diagrams in an input collection when these
        have different numbers of (non-trivial) points. See [1]_.

    nan_fill_value : float or None, optional, default: ``-1.``
        If a float, (normalized) persistence entropies initially computed as
        ``numpy.nan`` are replaced with this value. If ``None``, these values
        are left as ``numpy.nan``.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    NumberOfPoints, Amplitude, TopologicalVector, ComplexPolynomial

    References
    ----------
    .. [1] A. Myers, E. Munch, and F. A. Khasawneh, "Persistent Homology of
           Complex Networks for Dynamic State Detection"; *Phys. Rev. E*
           **100**, 022314, 2019; `DOI: 10.1103/PhysRevE.100.022314
           <https://doi.org/10.1103/PhysRevE.100.022314>`_.

    """

    _hyperparameters = {
        'normalize': {'type': bool},
        'nan_fill_value': {'type': (Real, type(None))}
        }

    def __init__(self, normalize=False, nan_fill_value=-1., n_jobs=None):
        self.normalize = normalize
        self.nan_fill_value = nan_fill_value
        self.n_jobs = n_jobs

    @staticmethod
    def _persistence_entropy(X, normalize=False, nan_fill_value=None):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_entropy = entropy(X_lifespan, base=2, axis=1)
        if normalize:
            lifespan_sums = np.sum(X_lifespan, axis=1)
            X_entropy /= np.log2(lifespan_sums)
        if nan_fill_value is not None:
            np.nan_to_num(X_entropy, nan=nan_fill_value, copy=False)
        X_entropy = X_entropy[:, None]
        return X_entropy

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        return self

    def transform(self, X, y=None):
        """Compute the persistence entropies of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions)
            Persistence entropies: one value per sample and per homology
            dimension seen in :meth:`fit`. Index i along axis 1 corresponds to
            the i-th homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        with np.errstate(divide='ignore', invalid='ignore'):
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(self._persistence_entropy)(
                    _subdiagrams(X[s], [dim]),
                    normalize=self.normalize,
                    nan_fill_value=self.nan_fill_value
                    )
                for dim in self.homology_dimensions_
                for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs))
                )
        Xt = np.concatenate(Xt).reshape(self._n_dimensions, len(X)).T

        return Xt


@adapt_fit_transform_docs
class Amplitude(BaseEstimator, TransformerMixin):
    """:ref:`Amplitudes <vectorization_amplitude_and_kernel>` of persistence
    diagrams.

    For each persistence diagram in a collection, a vector of amplitudes or a
    single scalar amplitude is calculated according to the following steps:

        1. The diagram is partitioned into subdiagrams according to homology
           dimension.
        2. The amplitude of each subdiagram is calculated according to the
           parameters `metric` and `metric_params`. This gives a vector of
           amplitudes, :math:`\\mathbf{a} = (a_{q_1}, \\ldots, a_{q_n})` where
           the :math:`q_i` range over the available homology dimensions.
        3. The final result is either :math:`\\mathbf{a}` itself or a norm of
           :math:`\\mathbf{a}`, specified by the parameter `order`.

    **Important notes**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.
        - The shape of outputs of :meth:`transform` depends on the value of the
          `order` parameter.

    Parameters
    ----------
    metric : ``'bottleneck'`` | ``'wasserstein'`` | ``'betti'`` | \
        ``'landscape'`` | ``'silhouette'`` | ``'heat'`` | \
        ``'persistence_image'``, optional, default: ``'landscape'``
        Distance or dissimilarity function used to define the amplitude of a
        subdiagram as its distance from the (trivial) diagonal diagram:

        - ``'bottleneck'`` and ``'wasserstein'`` refer to the identically named
          perfect-matching--based notions of distance.
        - ``'betti'`` refers to the :math:`L^p` distance between Betti curves.
        - ``'landscape'`` refers to the :math:`L^p` distance between
          persistence landscapes.
        - ``'silhouette'`` refers to the :math:`L^p` distance between
          silhouettes.
        - ``'heat'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams.
        - ``'persistence_image'`` refers to the :math:`L^p` distance between
          Gaussian-smoothed diagrams represented on birth-persistence axes.

    metric_params : dict or None, optional, default: ``None``
        Additional keyword arguments for the metric function (passing ``None``
        is equivalent to passing the defaults described below):

        - If ``metric == 'bottleneck'`` there are no available arguments.
        - If ``metric == 'wasserstein'`` the only argument is `p` (float,
          default: ``2.``).
        - If ``metric == 'betti'`` the available arguments are `p` (float,
          default: ``2.``) and `n_bins` (int, default: ``100``).
        - If ``metric == 'landscape'`` the available arguments are `p` (float,
          default: ``2.``), `n_bins` (int, default: ``100``) and `n_layers`
          (int, default: ``1``).
        - If ``metric == 'silhouette'`` the available arguments are `p` (float,
          default: ``2.``), `power` (float, default: ``1.``) and `n_bins` (int,
          default: ``100``).
        - If ``metric == 'heat'`` the available arguments are `p` (float,
          default: ``2.``), `sigma` (float, default: ``0.1``) and `n_bins`
          (int, default: ``100``).
        - If ``metric == 'persistence_image'`` the available arguments are `p`
          (float, default: ``2.``), `sigma` (float, default: ``0.1``), `n_bins`
          (int, default: ``100``) and `weight_function` (callable or None,
          default: ``None``).

    order : float or None, optional, default: ``None``
        If ``None``, :meth:`transform` returns for each diagram a vector of
        amplitudes corresponding to the dimensions in
        :attr:`homology_dimensions_`. Otherwise, the :math:`p`-norm of these
        vectors with :math:`p` equal to `order` is taken.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    effective_metric_params_ : dict
        Dictionary containing all information present in `metric_params` as
        well as relevant quantities computed in :meth:`fit`.

    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    NumberOfPoints, PersistenceEntropy, TopologicalVector, ComplexPolynomial, \
    PairwiseDistance, Scaler, Filtering

    Notes
    -----
    To compute amplitudes without first splitting the computation between
    different homology dimensions, data should be first transformed by an
    instance of :class:`ForgetDimension`.

    """

    _hyperparameters = {
        'metric': {'type': str, 'in': _AVAILABLE_AMPLITUDE_METRICS.keys()},
        'order': {'type': (Real, type(None)),
                  'in': Interval(0, np.inf, closed='right')},
        'metric_params': {'type': (dict, type(None))}
        }

    def __init__(self, metric='landscape', metric_params=None, order=None,
                 n_jobs=None):
        self.metric = metric
        self.metric_params = metric_params
        self.order = order
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and compute
        :attr:`effective_metric_params`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        if self.metric_params is None:
            self.effective_metric_params_ = {}
        else:
            self.effective_metric_params_ = self.metric_params.copy()
        validate_params(self.effective_metric_params_,
                        _AVAILABLE_AMPLITUDE_METRICS[self.metric])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)

        self.effective_metric_params_['samplings'], \
            self.effective_metric_params_['step_sizes'] = \
            _bin(X, self.metric, **self.effective_metric_params_)

        if self.metric == 'persistence_image':
            weight_function = self.effective_metric_params_.get(
                'weight_function', None
                )
            weight_function = \
                np.ones_like if weight_function is None else weight_function
            self.effective_metric_params_['weight_function'] = weight_function

        return self

    def transform(self, X, y=None):
        """Compute the amplitudes or amplitude vectors of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of X.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions) if `order` \
            is ``None``, else (n_samples, 1)
            Amplitudes or amplitude vectors of the diagrams in `X`. In the
            second case, index i along axis 1 corresponds to the i-th homology
            dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        Xt = check_diagrams(X, copy=True)

        Xt = _parallel_amplitude(Xt, self.metric,
                                 self.effective_metric_params_,
                                 self.homology_dimensions_,
                                 self.n_jobs)
        if self.order is not None:
            Xt = np.linalg.norm(Xt, axis=1, ord=self.order).reshape(-1, 1)

        return Xt


@adapt_fit_transform_docs
class NumberOfPoints(BaseEstimator, TransformerMixin):
    """Number of off-diagonal points in persistence diagrams, per homology
    dimension.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and their respective numbers of off-diagonal points
    are calculated.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    Amplitude, PersistenceEntropy, TopologicalVector, ComplexPolynomial

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    @staticmethod
    def _number_points(X):
        return np.count_nonzero(X[:, :, 1] - X[:, :, 0], axis=1)

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_`. Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit = np.unique(X[0, :, 2])
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)
        self._n_dimensions = len(self.homology_dimensions_)

        return self

    def transform(self, X, y=None):
        """Compute a vector of numbers of off-diagonal points for each diagram
        in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions)
            Number of points: one value per sample and per homology dimension
            seen in :meth:`fit`. Index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagrams(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._number_points)(_subdiagrams(X, [dim])[s])
            for dim in self.homology_dimensions_
            for s in gen_even_slices(len(X), effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt).reshape(self._n_dimensions, len(X)).T

        return Xt


@adapt_fit_transform_docs
class ComplexPolynomial(BaseEstimator, TransformerMixin):
    """Coefficients of complex polynomials whose roots are obtained from points
    in persistence diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    first considered separately. For each subdiagram, the polynomial whose
    roots are complex numbers obtained from its birth-death pairs is
    computed, and its :attr:`n_coefficients_` highest-degree complex
    coefficients excluding the top one are stored into a single real vector
    by concatenating the vector of all real parts with the vector of all
    imaginary parts [1]_ (if not enough coefficients are available to form a
    vector of the required length, padding with zeros is performed). Finally,
    all such vectors coming from different subdiagrams are concatenated to
    yield a single vector for the diagram.

    There are three possibilities for mapping birth-death pairs :math:`(b, d)`
    to complex polynomial roots. They are:

    .. math::
       :nowrap:

       \\begin{gather*}
       R(b, d) = b + \\mathrm{i} d, \\\\
       S(b, d) = \\frac{d - b}{\\sqrt{2} r} (b + \\mathrm{i} d), \\\\
       T(b, d) = \\frac{d - b}{2} [\\cos{r} - \\sin{r} + \
       \\mathrm{i}(\\cos{r} + \\sin{r})],
       \\end{gather*}

    where :math:`r = \\sqrt{b^2 + d^2}`.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    polynomial_type : ``'R'`` | ``'S'`` | ``'T'``, optional, default: ``'R'``
        Type of complex polynomial to compute.

    n_coefficients : list, tuple, int or None, optional, default: ``10``
        Number of complex coefficients per homology dimension. If an int then
        the number of coefficients will be equal to that value for each
        homology dimension. If ``None`` then, for each homology dimension in
        the collection of persistence diagrams seen in :meth:`fit`, the number
        of complex coefficients is defined to be the (constant, see
        **Important note**) number of points in each subdiagram in that
        homology dimension.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    n_coefficients_ : tuple
        Effective number of complex coefficients per homology dimension. Set in
        :meth:`fit`.

    See also
    --------
    NumberOfPoints, Amplitude, PersistenceEntropy, TopologicalVector

    References
    ----------
    .. [1] B. Di Fabio and M. Ferri, "Comparing Persistence Diagrams Through
           Complex Vectors"; in *Image Analysis and Processing — ICIAP 2015*,
           2015; `DOI: 10.1007/978-3-319-23231-7_27
           <https://doi.org/10.1007/978-3-319-23231-7_27>_.

    """
    _hyperparameters = {
        'n_coefficients': {'type': (int, type(None), list, tuple),
                           'in': Interval(1, np.inf, closed='left'),
                           'of': {'type': int,
                                  'in': Interval(1, np.inf, closed='left')}},
        'polynomial_type': {'type': str,
                            'in': _AVAILABLE_POLYNOMIALS.keys()}
        }

    def __init__(self, n_coefficients=10, polynomial_type='R', n_jobs=None):
        self.n_coefficients = n_coefficients
        self.polynomial_type = polynomial_type
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and compute :attr:`n_coefficients_`. Then,
        return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit, counts = np.unique(X[0, :, 2],
                                                    return_counts=True)
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)

        _n_homology_dimensions = len(self.homology_dimensions_)
        _homology_dimensions_counts = dict(zip(homology_dimensions_fit,
                                               counts))
        if self.n_coefficients is None:
            self.n_coefficients_ = \
                tuple([_homology_dimensions_counts[dim]
                       for dim in self.homology_dimensions_])
        elif type(self.n_coefficients) in (list, tuple):
            if len(self.n_coefficients) != _n_homology_dimensions:
                raise ValueError(
                    f'`n_coefficients` has length {len(self.n_coefficients)} '
                    f'while diagrams in `X` have {_n_homology_dimensions} '
                    f'homology dimensions.'
                    )
            self.n_coefficients_ = self.n_coefficients
        else:
            self.n_coefficients_ = \
                tuple([self.n_coefficients] * _n_homology_dimensions)

        self._polynomial_function = \
            _implemented_polynomial_recipes[self.polynomial_type]

        return self

    def _complex_polynomial(self, X, n_coefficients):
        Xt = np.zeros(2 * n_coefficients)
        X = X[X[:, 0] != X[:, 1]]

        roots = self._polynomial_function(X)
        coefficients = np.poly(roots)

        coefficients = np.array(coefficients[1:])
        dimension = min(n_coefficients, coefficients.shape[0])
        Xt[:dimension] = coefficients[:dimension].real
        Xt[n_coefficients:n_coefficients + dimension] = \
            coefficients[:dimension].imag

        return Xt

    def transform(self, X, y=None):
        """Compute vectors of real and imaginary parts of coefficients of
        complex polynomials obtained from each diagram in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features_new)
            Polynomial coefficients: real and imaginary parts of the complex
            polynomials obtained in each homology dimension from each diagram
            in `X`. ``n_features_new`` is equal to twice the sum of all entries
            in :attr:`n_coefficients_`.

        """
        check_is_fitted(self)
        Xt = check_diagrams(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._complex_polynomial)(
                _subdiagrams(Xt[s:s + 1], [dim], remove_dim=True)[0],
                self.n_coefficients_[d]
                )
            for s in range(len(X))
            for d, dim in enumerate(self.homology_dimensions_)
            )
        Xt = np.concatenate(Xt).reshape(len(X), -1)

        return Xt


@adapt_fit_transform_docs
class TopologicalVector(BaseEstimator, TransformerMixin):
    """Topological vectors from persistence diagrams.

    The topological vector associated to a persistence diagram is the sorted
    vector of a slight modification of the pairwise distances between the
    points in the persistence diagram. See [1]_.

    **Important note**:

        - Input collections of persistence diagrams for this transformer must
          satisfy certain requirements, see e.g. :meth:`fit`.

    Parameters
    ----------
    n_distances : list, tuple, int or None, optional, default: ``10``
        Number of distances to keep per homology dimension. If an int then the
        number of distances will be equal to that value for each homology
        dimension. If ``None`` then, for each homology dimension in the
        collection of persistence diagrams seen in :meth:`fit`, the number of
        distances is defined to be the (constant, see **Important note**)
        number of points in each subdiagram in that homology dimension.

    metric : string or callable, optional, default: ``'chebyshev'``
        The distance metric to use. See the documentation of
        :class:`sklearn.neighbors.DistanceMetric` for a list of available
        metrics. The default ``'chebyshev'`` is as proposed in [1]_.

    metric_params : dict or None, optional, default: ``{}``
        Additional keyword arguments for the metric function.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    homology_dimensions_ : tuple
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    n_distances_ : tuple
        Effective number of distances per homology dimension. Set in
        :meth:`fit`.

    See also
    --------
    NumberOfPoints, Amplitude, PersistenceEntropy, ComplexPolynomial, \
    PairwiseDistance

    References
    ----------
    .. [1] M. Carrière, S. Y. Oudot, and M. Ovsjanikov, "Stable Topological
           Signatures for Points on 3D Shapes"; *Computer Graphics Forum*
           **34** (5), 2015; `DOI: 10.1111:cgf.12692`2015
           <https://doi.org/10.1111:cgf.12692`2015>`_.

    """

    _hyperparameters = {
        'n_distances': {'type': (int, type(None), list, tuple),
                        'in': Interval(1, np.inf, closed='left'),
                        'of': {'type': int,
                               'in': Interval(1, np.inf, closed='left')}},
        'metric': {'type': (str, FunctionType)},
        'metric_params': {'type': dict}
        }

    def __init__(self, n_distances=10, metric='chebyshev', metric_params={},
                 n_jobs=None):
        self.n_distances = n_distances
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and compute :attr:`n_distances_`. Then,
        return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagrams(X)
        validate_params(
            self.get_params(), self._hyperparameters, exclude=['n_jobs'])

        # Find the unique homology dimensions in the 3D array X passed to `fit`
        # assuming that they can all be found in its zero-th entry
        homology_dimensions_fit, counts = np.unique(X[0, :, 2],
                                                    return_counts=True)
        self.homology_dimensions_ = \
            _homology_dimensions_to_sorted_ints(homology_dimensions_fit)

        _n_homology_dimensions = len(self.homology_dimensions_)
        _homology_dimensions_counts = dict(zip(homology_dimensions_fit,
                                               counts))

        if self.n_distances is None:
            self.n_distances_ = \
                tuple([_homology_dimensions_counts[dim]
                       for dim in self.homology_dimensions_])
        elif type(self.n_distances) in [list, tuple]:
            if len(self.n_distances) != _n_homology_dimensions:
                raise ValueError(
                    f'`n_distances` has length {len(self.n_distances)} '
                    f'while diagrams in `X` have {_n_homology_dimensions} '
                    f'homology dimensions.'
                    )
            self.n_distances_ = self.n_distances
        else:
            self.n_distances_ = \
                tuple([self.n_distances] * self.homology_dimensions_)

        self._distance_function = \
            DistanceMetric.get_metric(self.metric, **self.metric_params)

        return self

    def _topological_vector(self, Xd, n_distances):
        Xv = np.zeros(n_distances)
        distances = self._distance_function.pairwise(Xd)

        min_persistence = 0.5 * (Xd[:, 1] - Xd[:, 0])
        vector = np.flip(np.sort(
            np.triu(np.minimum(distances, min_persistence)), axis=None), 0
            )

        dimension = min(len(vector), n_distances)
        Xv[:dimension] = vector[:dimension]

        return Xv

    def transform(self, X, y=None):
        """Compute the topological vectors of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).
            It is important that, for each possible homology dimension, the
            number of triples for which q equals that homology dimension is
            constants across the entries of `X`.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------

        Xt : ndarray of shape (n_samples, n_features_new)
            Output data. Topological vectors of the diagrams in `X`.
            ``n_features_new`` is equal to the sum of all entries in
            :attr:`n_distances_`.

        """
        check_is_fitted(self)
        Xt = check_diagrams(X)

        Xt = Parallel(n_jobs=self.n_jobs)(
            delayed(self._topological_vector)(
                _subdiagrams(Xt[s:s + 1], [dim], remove_dim=True)[0],
                self.n_distances_[d]
            )
            for s in range(len(X))
            for d, dim in enumerate(self.homology_dimensions_)
            )
        Xt = np.concatenate(Xt).reshape(len(X), -1)

        return Xt
