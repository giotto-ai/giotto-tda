# License: Apache 2.0

import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import math

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class HeatDiffusion(BaseEstimator, TransformerMixin):
    """Compute heat diffusion.

    Module that computes heat diffusion evolution at different
    points in time. The initial condition and sampling times are
    taken as input.

    """
    def fit(self, X, taus, initial_condition=None, order=50, proc="exact"):
        """Compute heat diffusion throughout the complex.

        Parameters
        ----------
        X : ndarray
            Laplacian matrix to be used to compute the diffusion process,
            shape (n_simplices, n_simplices).

        taus : ndarray
            Array contaning the points in time at which sampling
            the heat diffusion process.

        initial_condition : ndarray, optional, default "None"
            Initial condition for the diffusion process. If "None",
            the diffusion will be computed by using the identity matrix with
            shape (n_simplices, n_simplices). If the initial conditions are
            not deltas placed on one simplex, the absolute values of diffusion
            vectors depend on the orientation given to the simplices (K-order
            diffusion with K>0 ).

        order : int, optional, default "30"
            Order of the polynomial approximation.

        proc : string, optional, default "exact"
            Procedure to compute the signatures (approximate == by
            using Chebychev approx -- or exact). At the moment,
            The 'approximate' option holds only for 0-order diffusion.

        Returns
        -------
        self : object

        """

        if initial_condition is not None:
            check_array(initial_condition)
            self.initial_condition = initial_condition
        else:
            self.initial_condition = np.identity(X.shape[0])

        self.taus_ = taus
        self.proc_ = proc
        self.order_ = order

        return self

    def transform(self, X, y=None):
        """Compute heat diffusion.

        Compute heat diffusion processes both for a set of s-simplexes or for
        all simplexes of order s.

        Parameters
        ----------
        X : csr_matrix
            Laplacian matrix to be used to compute the diffusion process,
            shape (n_simplices, n_simplices).

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------

        heat : ndarray
            3D-array containing the diffusion vectors. The shape is
            (n_simplices, n_initial_condition, n_times) where the initial
            condition placed as input have been transformed by solving the heat
            equation in time.

        """

        check_is_fitted(self, ['taus_', 'proc_', 'order_'])
        check_array(X, accept_sparse=True)
        heat = np.squeeze(np.asarray(self._compute_heat_diffusion(X)))
        heat = np.swapaxes(np.swapaxes(heat, 0, 2), 0, 1)

        return heat

    def _compute_heat_diffusion(self, lap):

        eps = 1e-9
        n_simplices = csr_matrix.get_shape(lap)[0]

        norm = np.vectorize(lambda x: 0 if np.abs(x) < eps else x)
        n_filters = len(self.taus_)

        if self.proc_ == 'exact':
            eigenvals, U = self._get_eigens(lap)

            heat = list()
            for i in range(n_filters):
                temp = U.dot(np.diagflat(
                    np.exp(- self.taus_[i] * eigenvals).flatten())).dot(U.T).\
                    dot(self.initial_condition)
                heat.append((norm(temp)))
        else:
            heat = [sp.sparse.csc_matrix((n_simplices, n_simplices)) for i in
                    range(n_filters)]
            monome = {0: sp.sparse.eye(n_simplices),
                      1: lap - sp.sparse.eye(n_simplices)}
            for k in range(2, self.order_ + 1):
                monome[k] = 2 * (lap - sp.sparse.eye(n_simplices)).dot(
                    monome[k - 1]) - monome[k - 2]
            for i in range(n_filters):
                coeffs = self._compute_cheb_coeff_basis(
                    self.taus_[i], self.order_)
                temp = sp.sum([coeffs[k] * monome[k] for k in
                               range(0, self.order_ + 1)])
                heat[i] = norm(temp.A)  # cleans up the small coefficients
        return heat

    def _compute_cheb_coeff_basis(self, scale, order):
        xx = np.array([np.cos((2 * i - 1) * 1.0 / (2 * order) * math.pi)
                       for i in range(1, order + 1)])
        basis = [np.ones((1, order)), np.array(xx)]
        for k in range(order + 1 - 2):
            basis.append(2 * np.multiply(xx, basis[-1]) - basis[-2])
        basis = np.vstack(basis)
        f = np.exp(-scale * (xx + 1))
        products = np.einsum("j,ij->ij", f, basis)
        coeffs = 2.0 / order * products.sum(1)
        coeffs[0] = coeffs[0] / 2
        return list(coeffs)

    def _get_eigens(self, lap):
        vals, vecs = np.linalg.eigh(lap.todense())
        return np.real(vals), vecs
