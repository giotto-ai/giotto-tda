# License: GNU AGPLv3

import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices


_AVAILABLE_FUNCTIONS = {
    "flatten": {},
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
    "flatten": lambda X, axis: X.reshape(len(X), -1),
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
    Xt = Parallel(n_jobs=n_jobs)(
        delayed(function)(Xt[s], axis=-1, **function_params)
        for s in gen_even_slices(len(Xt), effective_n_jobs(n_jobs))
        )
    Xt = np.concatenate(Xt)

    return Xt
