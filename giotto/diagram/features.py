# License: Apache 2.0

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import gen_even_slices
from ..utils.validation import check_diagram
from ._utils import _subdiagrams, _discretize
from giotto.diagram._metrics import betti_curves, landscapes, heats


class PersistenceEntropy(BaseEstimator, TransformerMixin):
    """`Persistence entropies <LINK TO GLOSSARY>`_ of persistence diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], its q-persistence entropy is the (base e) entropy of the
    collection of differences d - b for points of homology dimension q,
    normalized by the sum of all such differences.

    Parameters
    ----------
    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`, sorted in ascending order.

    See also
    --------
    VietorisRipsPersistence

    """

    def __init__(self, n_jobs=None):
        self.n_jobs = n_jobs

    def _persistence_entropy(self, X):
        X_lifespan = X[:, :, 1] - X[:, :, 0]
        X_normalized = X_lifespan / np.sum(X_lifespan, axis=1).reshape((-1, 1))
        return - np.sum(np.nan_to_num(
            X_normalized * np.log(X_normalized)), axis=1).reshape((-1, 1))

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)
        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each persistence diagram, compute the vector of persistence
        entropies corresponding to each homology dimension seen in :meth:`fit`.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_homology_dimensions)
            Array of persistence entropies (one value per sample and per
            homology dimension).

        """
        # Check if fit had been called
        check_is_fitted(self, ['_is_fitted'])
        X = check_diagram(X)

        n_samples = len(X)

        with np.errstate(divide='ignore', invalid='ignore'):
            Xt = Parallel(n_jobs=self.n_jobs)(
                delayed(self._persistence_entropy)(_subdiagrams(X, [dim])[s])
                for dim in self.homology_dimensions_
                for s in gen_even_slices(
                    n_samples, effective_n_jobs(self.n_jobs))
            )
        Xt = np.concatenate(Xt).reshape((self._n_dimensions, n_samples)).T
        return Xt


class BettiCurve(BaseEstimator, TransformerMixin):
    """`Betti curves <LINK TO GLOSSARY>`_ of persistence diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], the value of its q-Betti curve at parameter r is simply the
    number of persistent features in homology dimension q which are alive at r.
    Approximate Betti curves are constructed by sampling the `filtration
    parameter <LINK TO GLOSSARY>`_ at evenly spaced values, once per
    available homology dimension.

    Parameters
    ----------
    n_values : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: None
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
    VietorisRipsPersistence

    """
    def __init__(self, n_values=100, n_jobs=None):
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)
        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self.samplings_, _ = _discretize(X, n_values=self.n_values)
        return self

    def transform(self, X, y=None):
        """Compute the Betti curves (one per homology dimension available in
        :meth:`fit`) of each diagram.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_homology_dimensions, n_values)
            Betti curves.

        """
        # Check if fit had been called
        check_is_fitted(self, ['homology_dimensions_', 'samplings_'])
        X = check_diagram(X)

        n_samples = len(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(betti_curves)(
                _subdiagrams(X, [dim], remove_dim=True)[s],
                self.samplings_[dim])
            for dim in self.homology_dimensions_
            for s in gen_even_slices(n_samples, effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).\
            reshape((self._n_dimensions, n_samples, -1)).\
            transpose((1, 0, 2))
        return Xt


class PersistenceLandscape(BaseEstimator, TransformerMixin):
    """`Persistence landscapes <LINK TO GLOSSARY>`_ of persistence diagrams.

    Given a persistence diagram consisting of birth-death-dimension triples
    [b, d, q], for each n its q-persistence
    landscape up to layer k is computed.

    Parameters
    ----------
    n_values : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each number in `homology_dimensions_`, store a discrete sampling of
        filtration parameters calculated during :meth:`fit`.

    See also
    --------
    VietorisRipsPersistence

    """
    def __init__(self, n_layers=1, n_values=100, n_jobs=None):
        self.n_layers = n_layers
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)

        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self.samplings_, _ = _discretize(X, n_values=self.n_values)

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """For each persistence subdiagram corresponding to an homology
        dimension k, compute that subdiagram's landscapes.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_homology_dimensions, \
            n_layers, n_values)
            Array of the persistence landscapes of the diagrams in `X`.

        """
        check_is_fitted(self, ['homology_dimensions_', 'samplings_'])
        X = check_diagram(X)

        n_samples = len(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(landscapes)(
                _subdiagrams(X, [dim], remove_dim=True)[s],
                self.samplings_[dim],
                self.n_layers)
            for dim in self.homology_dimensions_
            for s in gen_even_slices(n_samples, effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).reshape((self._n_dimensions, n_samples,
                                         self.n_layers, self.n_values)).\
            transpose((1, 0, 2, 3))
        return Xt


class HeatKernel(BaseEstimator, TransformerMixin):
    """Transformer for the calculation of the persistence landscapes from a
    collection of persistence diagrams. Given a generic persistence diagram
    consisting of birth-death-dimension triples [b, d, q], their q-persistence
    landscapes are TO DO.

    Parameters
    ----------
    sigma : float
        Standard deviation for Gaussian kernel.

    n_values : int, optional, default: ``100``
        The number of filtration parameter values, per available homology
        dimension, to sample during :meth:`fit`.

    n_jobs : int or None, optional, default: None
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors.

    Attributes
    ----------
    homology_dimensions_ : list
        Homology dimensions seen in :meth:`fit`.

    samplings_ : dict
        For each number in `homology_dimensions_`, store a discrete sampling of
        filtration parameters calculated during :meth:`fit`.

    See also
    --------
    VietorisRipsPersistence

    """
    def __init__(self, sigma, n_values=100, n_jobs=None):
        self.sigma = sigma
        self.n_values = n_values
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        self : object

        """
        X = check_diagram(X)
        self.homology_dimensions_ = sorted(list(set(X[0, :, 2])))
        self._n_dimensions = len(self.homology_dimensions_)

        self.samplings_, self._step_size = _discretize(
            X, n_values=self.n_values)
        return self

    def transform(self, X, y=None):
        """For each persistence subdiagram corresponding to an homology
        dimension k, compute that subdiagram's landscapes.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features, 3)
            Input data. Array of persistence diagrams, each a collection of
            triples [b, d, q] representing persistent topological features
            through their birth (b), death (d) and homology dimension (q).

        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.

        Returns
        -------
        Xt : ndarray, shape (n_samples, n_homology_dimensions, n_values, \
            n_values)
            Array of the persistence landscapes of the diagrams in `X`.

        """
        check_is_fitted(self, ['homology_dimensions_', 'samplings_'])
        X = check_diagram(X)

        n_samples = len(X)

        Xt = Parallel(n_jobs=self.n_jobs)(delayed(
            heats)(_subdiagrams(X, [dim], remove_dim=True)[s],
                   self.samplings_[dim], self._step_size[dim], self.sigma)
            for dim in self.homology_dimensions_
            for s in gen_even_slices(n_samples, effective_n_jobs(self.n_jobs)))
        Xt = np.concatenate(Xt).reshape((self._n_dimensions, n_samples,
                                         self.n_values, self.n_values)).\
            transpose((1, 0, 2, 3))
        return Xt
