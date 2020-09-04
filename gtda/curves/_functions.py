import numpy as np
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.utils import gen_even_slices


_AVAILABLE_FUNCTIONS = {
    'argmax': {},
    'argmin': {},
    'min': {},
    'max': {},
    'mean': {},
    'std': {},
    'median': {},
    'average': {'weights': {'type': np.ndarray}},
}

implemented_function_recipes = {
    'argmax': np.argmax,
    'argmin': np.argmin,
    'min': np.min,
    'max': np.max,
    'mean': np.mean,
    'std': np.std,
    'median': np.median,
    'average': np.average,
}


def _parallel_featurization(Xt, function, function_params, n_jobs):
    function_func = implemented_function_recipes[function]

    Xt = Parallel(n_jobs=n_jobs)(
        delayed(function_func)(Xt[s], axis=-1, **function_params)
        for s in gen_even_slices(
                Xt.shape[0], effective_n_jobs(n_jobs))
    )
    Xt = np.concatenate(Xt)

    return Xt
