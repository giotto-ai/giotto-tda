"""Utilities for docstring building."""
# License: GNU AGPLv3

import re
from functools import wraps
from inspect import getdoc

from sklearn.base import TransformerMixin

inputs_start = 'Parameters\n----------\n'
inputs_end = 'Returns\n-------\n'
outputs_end = '\n\n'


def get_preamble_docs(docs):
    re_search = re.search(
        f'^(.*){inputs_start}', docs, flags=re.DOTALL)
    return re_search.group(1)


def get_inputs_docs(docs):
    re_search = re.search(
        f'{inputs_start}(.*){inputs_end}', docs, flags=re.DOTALL)
    return re_search.group(1)


def get_outputs_docs(docs):
    re_search = re.search(
        f'{inputs_end}(.*){outputs_end}', docs, flags=re.DOTALL)
    if re_search is None:
        re_search = re.search(
            f'{inputs_end}(.*)$', docs, flags=re.DOTALL)
    return re_search.group(1)


standard_fit_transform_docs = getdoc(TransformerMixin.fit_transform)
standard_intro_docs = get_preamble_docs(standard_fit_transform_docs)
intro_docs = re.sub(r'\bX\b', '`X`', standard_intro_docs)
intro_docs = re.sub(r'\by\b', '`y`', intro_docs)
intro_docs = re.sub(r'\bfit_params\b', '`fit_params`', intro_docs)
standard_inputs_docs = get_inputs_docs(standard_fit_transform_docs)
standard_outputs_docs = get_outputs_docs(standard_fit_transform_docs)


def make_fit_transform_docs(fit_docs, transform_docs):
    """Create docstring for a :meth:`fit_transform` method.

    Uses the standard documentation for
    :class:`sklearn.base.TransformerMixin` as a template, but replaces the
    "Parameters" section with the corresponding section from `fit_docs`,
    and the "Returns" section with the corresponding section from
    `transform_docs`. Also performs other cosmetic changes.

    Parameters
    ----------
    fit_docs : str
        Docstring for :meth:`fit`. Should contain a "Parameters" section.

    transform_docs : str
        Docstring for :meth:`transform`. Should contain a "Returns" section.

    """

    inputs_docs = get_inputs_docs(fit_docs)
    outputs_docs = get_outputs_docs(transform_docs)
    new_docstring = standard_fit_transform_docs.\
        replace(standard_intro_docs, intro_docs).\
        replace(standard_inputs_docs, inputs_docs).\
        replace(standard_outputs_docs, outputs_docs)
    return new_docstring


def adapt_fit_transform_docs(transformermixin_cls):
    """Class decorator changing the docstring for :meth:`fit_transform`.

    Fetches the :meth:`fit` and :meth:`transform` docstrings of a class
    implementing :meth:`fit_transform`, creates adapted docstring using
    :func:`gtda.utils._docs.make_fit_transform_docs`, and
    wraps the original :meth:`fit_transform` implementation with one with
    the new docstring.

    This is particularly useful for classes inheriting from
    :class:`sklearn.base.TransformerMixin`, when the standard docstring is
    inadequate because of exotic input shapes or types.

    Parameters
    ----------
    transformermixin_cls : type
        A class containing a method :meth:`fit_transform`.

    Returns
    -------
    transformermixin_cls : type
        Input class in which the :meth:`fit_transform` method has been
        replaced by a version with a modified docstring but otherwise identical
        behaviour.

    """

    fit_docs = getdoc(getattr(transformermixin_cls, 'fit'))
    transform_docs = getdoc(getattr(transformermixin_cls, 'transform'))

    def make_new_fit_transform(original_fit_transform):
        @wraps(original_fit_transform)
        def fit_transform_wrapper(*args, **kwargs):
            return original_fit_transform(*args, **kwargs)
        fit_transform_wrapper.__doc__ = \
            make_fit_transform_docs(fit_docs, transform_docs)
        return fit_transform_wrapper

    new_fit_transform = make_new_fit_transform(
        getattr(transformermixin_cls, 'fit_transform'))
    setattr(transformermixin_cls, 'fit_transform', new_fit_transform)
    return transformermixin_cls
