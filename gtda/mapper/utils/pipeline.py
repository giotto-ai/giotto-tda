"""Utility functions for scikit-learn pipelines."""
# License: GNU AGPLv3

from functools import partial
from inspect import signature

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def _make_func_apply_along_axis_1(func):
    return partial(np.apply_along_axis, func, 1)


def _reshape_after_apply(func, arr):
    res = func(arr)
    if res.ndim == 1:
        res = res[:, None]
    return res


def transformer_from_callable_on_rows(func, validate=True):
    """Construct a transformer from a callable acting on 1D arrays.

    Given a callable which can act on 1D arrays, this function returns a
    fit-transformer which applies the callable to slices of 2D arrays along
    axis 1. When possible, the array output by the transformer's
    :meth:`fit_transform` is two-dimensional.

    Parameters
    ----------
    func : callable or None
        A callable object, or ``None`` which returns the identity transformer.

    validate : bool, optional, default: ``True``
        Whether the output transformer should implement input validation.

    Returns
    -------
    function_transformer : :class:`sklearn.preprocessing.FunctionTransformer` \
        object
        Output fit-transformer.

    Examples
    --------
    >>> import numpy as np
    >>> from gtda.mapper import transformer_from_callable_on_rows
    >>> function_transformer = transformer_from_callable_on_rows(np.sum)
    >>> X = np.array([[0, 1], [2, 3]])
    >>> print(function_transformer.fit_transform(X))
    [[1],
     [5]]

    """
    if func is not None:
        func_params = signature(func).parameters
        if 'axis' in func_params:  # Use native (faster) numpy implementation
            func_along_axis = partial(func, axis=1, keepdims=True)
        else:
            func_along_axis = partial(_reshape_after_apply,
                                      _make_func_apply_along_axis_1(func))
    else:
        func_along_axis = None

    return FunctionTransformer(func=func_along_axis, validate=validate)


def identity(validate=False):
    return FunctionTransformer(validate=validate)
