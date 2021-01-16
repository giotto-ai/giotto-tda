# License: GNU AGPLv3

import warnings
from itertools import product

import numpy as np
from joblib import Parallel, delayed

_AVAILABLE_FUNCTIONS = {
    "identity": {},
    "argmax": {},
    "argmin": {},
    "min": {},
    "max": {},
    "mean": {},
    "std": {},
    "median": {},
    "average": {"weights": {"type": np.ndarray}}
    }

_implemented_function_recipes = {
    "identity": lambda X, axis: X.reshape(len(X), -1),
    "argmax": np.argmax,
    "argmin": np.argmin,
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "std": np.std,
    "median": np.median,
    "average": np.average
    }


def _parallel_featurization(Xt, function, function_params, n_jobs):
    if callable(function):
        return function(Xt, axis=-1, **function_params)
    else:  # Assume function is a list or tuple of functions or None
        channel_idx = [j for j, f in enumerate(function) if f is not None]
        n_samples = len(Xt)
        index_pairs = product(range(n_samples), channel_idx)
        Xt = Parallel(n_jobs=n_jobs)(
            delayed(function[j])(Xt[i, j], **function_params[j])
            for i, j in index_pairs
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore",
                                  category=np.VisibleDeprecationWarning)
            Xt = np.array(Xt)
        if Xt.dtype == np.dtype('object'):
            Xt = np.concatenate(list(map(np.ravel, Xt)))

        return Xt.reshape(n_samples, -1)
