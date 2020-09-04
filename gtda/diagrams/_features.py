from numbers import Real
from types import FunctionType

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans, MiniBatchKMeans
from gtda.utils.intervals import Interval


_AVAILABLE_POLYNOMIALS = {
    'R': {},
    'S': {},
    'T': {},
}


def R_polynomial(Xd):
    roots = Xd[:, 0] + 1j * Xd[:, 1]

    return roots


def S_polynomial(Xd):
    alpha = np.linalg.norm(Xd, axis=1)
    alpha = np.where(alpha==0, np.ones(Xd.shape[0]), alpha)
    roots = np.multiply(
        np.multiply(
            (Xd[:, 0]+1j*Xd[:, 1]), (Xd[:, 1]-Xd[:, 0])
        ),
        1./(np.sqrt(2)*alpha)
    )

    return roots


def T_polynomial(Xd):
    alpha = np.linalg.norm(Xd, axis=1)
    roots = np.multiply(
        (Xd[:, 1]-Xd[:, 0])/2, np.cos(alpha) - np.sin(alpha)
        + 1j * (np.cos(alpha) + np.sin(alpha))
    )

    return roots


_implemented_polynomial_recipes = {
    'R': R_polynomial,
    'S': S_polynomial,
    'T': T_polynomial,
}
