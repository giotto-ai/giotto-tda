from functools import partial
from inspect import signature

import numpy as np
from sklearn.preprocessing import FunctionTransformer


def make_func_apply_along_axis_1(func):
    return partial(np.apply_along_axis, func, 1)


def func_from_callable_on_rows(func):
    if func is None:
        return None
    func_params = signature(func).parameters
    if 'axis' in func_params:
        return partial(func, axis=1)
    return make_func_apply_along_axis_1(func)


def identity():
    return FunctionTransformer(validate=True)
