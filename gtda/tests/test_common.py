# License: GNU AGPLv3
from functools import partial

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from gtda.images.preprocessing import Binarizer, Inverter


# mark checks to skip
SKIP_TESTS = {
  'Binarizer':  [],
  'Inverter':  [],
}

# mark tests as a known failure
# TODO: these should be addressed later.
# Note with scikit-learn 0.23 these can be moved to estimator tags
XFAIL_TESTS = {
  'Binarizer':  ["check_transformer_data_not_an_array",
                 "check_transformer_general",
                 "check_transformer_general(readonly_memmap=True)",
                 ],
  'Inverter':  ["check_transformer_data_not_an_array",
                "check_transformer_general",
                "check_transformer_general(readonly_memmap=True)",
                ],
}

LISTFAIL_TESTS = ['check_estimators_dtypes', 'check_fit_score_takes_y',
                  'check_estimators_fit_returns_self',
                  'check_estimators_fit_returns_self(readonly_memmap=True)',
                  'check_complex_data', 'check_dtype_object',
                  'check_estimators_empty_data_messages',
                  'check_pipeline_consistency', 'check_estimators_nan_inf',
                  'check_estimators_overwrite_params',
                  'check_estimator_sparse_data', 'check_estimators_pickle',
                  'check_fit2d_predict1d', 'check_methods_subset_invariance',
                  'check_fit2d_1sample', 'check_fit2d_1feature',
                  'check_dict_unchanged', 'check_dont_overwrite_parameters',
                  'check_fit_idempotent']


# adapted from sklearn.utils.estimator_check v0.22
def _get_callable_name(obj):
    """Get string representation of a function or a partial function name

    Examples
    --------
    >>> def f(x=2): pass
    >>> _get_callable_name(f)
    'f'
    >>> _get_callable_name(partial(f, x=1))
    'f(x=1)'
    """
    if not isinstance(obj, partial):
        return obj.__name__

    if not obj.keywords:
        return obj.func.__name__

    kwstring = ",".join([f"{k}={v}"
                         for k, v in obj.keywords.items()])
    return f"{obj.func.__name__}({kwstring})"


def _get_estimator_name(estimator):
    """Get string representation for classes and class instances

    Examples
    --------
    >>> from sklearn.preprocessing import StandardScaler
    >>> _get_estimator_name(StandardScaler)
    'StandardScaler'
    >>> _get_estimator_name(StandardScaler())
    'StandardScaler'
    """
    if isinstance(estimator, type):
        # this is class
        return estimator.__name__
    else:
        # this an instance
        return estimator.__class__.__name__


@parametrize_with_checks(
    [Binarizer, Inverter]
)
def test_sklearn_api(check, estimator, request):
    estimator_name = _get_estimator_name(estimator)
    check_name = _get_callable_name(check)

    if check_name in SKIP_TESTS[estimator_name]:
        # skip this test
        pytest.skip()

    if check_name in XFAIL_TESTS[estimator_name]:
        # mark tests as a known failure
        request.applymarker(pytest.mark.xfail(
            run=True, reason='known failure'))

    if check_name in LISTFAIL_TESTS:
        request.applymarker(pytest.mark.xfail(
            run=True, reason='Known failure: 2d input.'))

    check(estimator)
