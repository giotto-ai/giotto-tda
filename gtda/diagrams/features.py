"""Feature extraction from persistence diagrams."""
# License: GNU AGPLv3

import numbers

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import gen_even_slices
from sklearn.utils.validation import check_is_fitted

from ._metrics import betti_curves, landscapes, heats
from ._utils import _subdiagrams, _discretize
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params, check_diagram


@adapt_fit_transform_docs
class PersistenceEntropy(BaseEstimator, TransformerMixin):
    """`Persistence entropies <https://giotto.ai/theory>`_ of persistence
    diagrams.

    Given a persistence diagrams consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and their respective persistence entropies are
    calculated as the (base e) entropies of the collections of differences
    d - b, normalized by the sum of all such differences.

    Parameters
    ----------
    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    BettiCurve, PersistenceLandscape, HeatKernel, Amplitude, \
    PairwiseDistance, gtda.homology.VietorisRipsPersistence

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _persistence_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape(-1, 1)
        return - np.sum(np.nan_to_num(
            X_normalized * np.log(X_normalized)), axis=1).reshape(-1, 1)

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

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)

        self.homology_dimensions_ = sorted(set(X[0, :, 2]))
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

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions)
            Persistence entropies: one value per sample and per homology
            dimension seen in :meth:`fit`. Index i along axis 1 corresponds
            to the i-th homology dimension in :attr:`homology_dimensions_`.

        """
        # Check if fit had been called
        check_is_fitted(self)
        X = check_diagram(X)

        with np.errstate(divide='ignore', invalid='ignore'):
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(self._persistence_entropy)(_subdiagrams(X, [dim])[s])
                for dim in self.homology_dimensions_
                for s in gen_even_slices(
                    X.shape[0], effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt).reshape(self._n_dimensions, X.shape[0]).T
        return Xt


@adapt_fit_transform_docs
class BettiCurve(BaseEstimator, TransformerMixin):
    """`Betti curves <https://giotto.ai/theory>`_ of persistence diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and their respective Betti curves are obtained by
    evenly sampling the `filtration parameter <https://giotto.ai/theory>`_.

    Parameters
    ----------
    n_values : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1
        unless in a :obj:`joblib.parallel_backend` context. ``-1`` means
        using all processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    PersistenceLandscape, PersistenceEntropy, HeatKernel, Amplitude, \
    PairwiseDistance, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the j-th entry of a Betti
    curve in homology dimension q typically arises from a different parameter
    values to the j-th entry of a curve in dimension q'.

    """

    _hyperparameters = {'n_values': [int, (1, np.inf)]}

    def __init__(self, n_values=100, n_jobs=None):
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)
        validate_params(self.get_params(), self._hyperparameters)

        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, _ = _discretize(X, n_values=self.n_values)
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}
        return self

    def transform(self, X, y=None):
        """Compute the Betti curves of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_values)
            Betti curves: one curve (represented as a one-dimensional array
            of integer values) per sample and per homology dimension seen
            in :meth:`fit`. Index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        # Check if fit had been called
        check_is_fitted(self)
        X = check_diagram(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(betti_curves)(
                _subdiagrams(X, [dim], remove_dim=True)[s],
                self._samplings[dim])
            for dim in self.homology_dimensions_
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).\
            reshape(self._n_dimensions, X.shape[0], -1).\
            transpose((1, 0, 2))
        return Xt


@adapt_fit_transform_docs
class PersistenceLandscape(BaseEstimator, TransformerMixin):
    """`Persistence landscapes <https://giotto.ai/theory>`_ of persistence
    diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], subdiagrams corresponding to distinct homology dimensions are
    considered separately, and layers of their respective persistence
    landscapes are obtained by evenly sampling the `filtration parameter
    <https://giotto.ai/theory>`_.

    Parameters
    ----------
    n_layers : int, optional, default: ``1``
        How many layers to consider in the persistence landscape.

    n_values : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    BettiCurve, PersistenceEntropy, HeatKernel, Amplitude, \
    PairwiseDistance, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the j-th entry of the
    k-layer of a persistence landscape in homology dimension q typically
    arises from a different parameter value to the j-th entry of a k-layer in
    dimension q'.

    """

    _hyperparameters = {'n_layers': [int, (1, np.inf)],
                        'n_values': [int, (1, np.inf)]}

    def __init__(self, n_layers=1, n_values=100, n_jobs=None):
        self.n_layers = n_layers
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)
        validate_params(self.get_params(), self._hyperparameters)

        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, _ = _discretize(X, n_values=self.n_values)
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}

        return self

    def transform(self, X, y=None):
        """Compute the persistence landscapes of diagrams in `X`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, \
            n_layers, n_values)
            Persistence lanscapes: one landscape (represented as a
            two-dimensional array) per sample and per homology dimension seen
            in :meth:`fit`. Each landscape contains a number `n_layers` of
            layers. Index i along axis 1 corresponds to the i-th homology
            dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagram(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(landscapes)(
                _subdiagrams(X, [dim], remove_dim=True)[s],
                self._samplings[dim],
                self.n_layers)
            for dim in self.homology_dimensions_
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).reshape(self._n_dimensions, X.shape[0],
                                        self.n_layers, self.n_values).\
            transpose((1, 0, 2, 3))
        return Xt


@adapt_fit_transform_docs
class HeatKernel(BaseEstimator, TransformerMixin):
    """Convolution of persistence diagrams with a Gaussian kernel.

    Based on ideas in [1]_. Given a persistence diagram consisting of
    birth-death-dimension triples [b, d, q], subdiagrams corresponding to
    distinct homology dimensions are considered separately and regarded as sums
    of Dirac deltas. Then, the convolution with a Gaussian kernel is computed
    over a rectangular grid of locations evenly sampled from appropriate
    ranges of the `filtration parameter <https://giotto.ai/theory>`_. The
    same is done with the reflected images of the subdiagrams about the
    diagonal, and the difference between the results of the two convolutions is
    computed. The result can be thought of as a raster image.

    Parameters
    ----------
    sigma : float
        Standard deviation for Gaussian kernel.

    n_values : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each number in `homology_dimensions_`, a discrete sampling of
        filtration parameters, calculated during :meth:`fit` according to the
        minimum birth and maximum death values observed across all samples.

    See also
    --------
    BettiCurve, PersistenceLandscape, PersistenceEntropy, Amplitude, \
    PairwiseDistance, gtda.homology.VietorisRipsPersistence

    Notes
    -----
    The samplings in :attr:`samplings_` are in general different between
    different homology dimensions. This means that the (i, j)-th pixel of a
    persistence image in homology dimension q typically arises from a different
    pair of parameter values to the (i, j)-th pixel of a persistence image in
    dimension q'.

    References
    ----------
    .. [1] J. Reininghaus, S. Huber, U. Bauer, and R. Kwitt, "A Stable
           Multi-Scale Kernel for Topological Machine Learning"; *2015 IEEE
           Conference on Computer Vision and Pattern Recognition (CVPR)*,
           pp. 4741--4748, 2015; doi: `10.1109/CVPR.2015.7299106
           <http://dx.doi.org/10.1109/CVPR.2015.7299106>`_.

    """

    _hyperparameters = {'sigma': [numbers.Number, (1e-16, np.inf)],
                        'n_values': [int, (1, np.inf)]}

    def __init__(self, sigma, n_values=100, n_jobs=None):
        self.sigma = sigma
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Store all observed homology dimensions in
        :attr:`homology_dimensions_` and, for each dimension separately,
        store evenly sample filtration parameter values in :attr:`samplings_`.
        Then, return the estimator.

        This method is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)
        validate_params(self.get_params(), self._hyperparameters)

        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self._samplings, self._step_size = _discretize(
            X, n_values=self.n_values)
        self.samplings_ = {dim: s.flatten()
                           for dim, s in self._samplings.items()}
        return self

    def transform(self, X, y=None):
        """Compute raster images obtained from diagrams in `X` by convolution
        with a Gaussian kernel.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_homology_dimensions, n_values, \
            n_values)
            Raster images: one image per sample and per homology dimension seen
            in :meth:`fit`. Index i along axis 1 corresponds to the i-th
            homology dimension in :attr:`homology_dimensions_`.

        """
        check_is_fitted(self)
        X = check_diagram(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            heats)(_subdiagrams(X, [dim], remove_dim=True)[s],
                   self._samplings[dim], self._step_size[dim], self.sigma)
            for dim in self.homology_dimensions_
            for s in gen_even_slices(X.shape[0],
                                     effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).reshape(self._n_dimensions, X.shape[0],
                                        self.n_values, self.n_values).\
            transpose((1, 0, 2, 3))
        return Xt
