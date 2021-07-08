"""Utilities for input validation."""
# License: GNU AGPLv3

from functools import reduce
from operator import and_
from warnings import warn

import numpy as np
from scipy.sparse import issparse
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.utils.validation import check_array


def _check_array_mod(X, **kwargs):
    """Modified version of :func:`sklearn.utils.validation.check_array. When
    keyword parameter `force_all_finite` is set to False, NaNs are not
    accepted but infinity is."""
    if not kwargs.get('force_all_finite', True):
        Xnew = check_array(X, **kwargs)
        if np.isnan(Xnew if not issparse(Xnew) else Xnew.data).any():
            raise ValueError("Input contains NaNs. Only finite values and "
                             "infinity are allowed when parameter "
                             "`force_all_finite` is False.")
        return Xnew
    return check_array(X, **kwargs)


def check_diagrams(X, copy=False):
    """Input validation for collections of persistence diagrams.

    Basic type and sanity checks are run on the input collection and the
    array is converted to float type before returning. In particular,
    the input is checked to be an ndarray of shape ``(n_samples, n_points,
    3)``.

    Parameters
    ----------
    X : object
        Input object to check/convert.

    copy : bool, optional, default: ``False``
        Whether a forced copy should be triggered.

    Returns
    -------
    X_validated : ndarray of shape (n_samples, n_points, 3)
        The converted and validated array of persistence diagrams.

    """
    X_array = _check_array_mod(X, ensure_2d=False, allow_nd=True,
                               force_all_finite=False, copy=copy)
    if X_array.ndim != 3:
        raise ValueError(
            f"Input should be a 3D ndarray, the shape is {X_array.shape}."
            )
    if X_array.shape[2] != 3:
        raise ValueError(
            f"Input should be a 3D ndarray with a 3rd dimension of 3 "
            f"components, but there are {X_array.shape[2]} components."
            )

    homology_dimensions = sorted(np.unique(X_array[0, :, 2]))
    for dim in homology_dimensions:
        if dim == np.inf:
            if len(homology_dimensions) != 1:
                raise ValueError(
                    f"numpy.inf is a valid homology dimension for a stacked "
                    f"diagram but it should be the only one: "
                    f"homology_dimensions = {homology_dimensions}."
                    )
        else:
            if (dim != int(dim)) or (dim < 0):
                raise ValueError(
                    f"Homology dimensions should be positive integers or "
                    f"numpy.inf: {dim} can't be cast to an int of the same "
                    f"value."
                    )

    n_points_below_diag = np.sum(X_array[:, :, 1] < X_array[:, :, 0])
    if n_points_below_diag:
        raise ValueError(
            f"All points of all persistence diagrams should be above the "
            f"diagonal, i.e. X[:, :, 1] >= X[:, :, 0]. {n_points_below_diag} "
            f"points are below the diagonal."
            )

    return X_array


def check_graph(X):
    # TODO
    return X


def _validate_params(parameters, references, rec_name=None):
    types_tuple = (list, tuple, np.ndarray, dict)

    def _validate_params_single(_parameter, _reference, _name):
        if _reference is None:
            return

        _ref_type = _reference.get('type', None)

        # Check that _parameter has the correct type
        if not ((_ref_type is None) or isinstance(_parameter, _ref_type)):
            raise TypeError(f"Parameter `{_name}` is of type "
                            f"{type(_parameter)} while it should be of type "
                            f"{_ref_type}.")

        # If neither the reference type is list, tuple, np.ndarray or dict,
        # nor _parameter is an instance of one of these types, the checks are
        # performed on _parameter directly.
        elif not ((_ref_type in types_tuple)
                  or isinstance(_parameter, types_tuple)):
            ref_in = _reference.get('in', None)
            ref_other = _reference.get('other', None)
            if _parameter is not None:
                if not ((ref_in is None) or _parameter in ref_in):
                    raise ValueError(f"Parameter `{_name}` is {_parameter}, "
                                     f"which is not in {ref_in}.")
            # Perform any other checks via the callable ref_others
            if ref_other is not None:
                return ref_other(_parameter)

        # Explicitly return the type of _reference if one of list, tuple,
        # np.ndarray or dict.
        else:
            return _ref_type

    for name, parameter in parameters.items():
        if name not in references.keys():
            name_extras = "" if rec_name is None else f" in `{rec_name}`"
            raise KeyError(f"`{name}`{name_extras} is not an available "
                           f"parameter. Available parameters are in "
                           f"{tuple(references.keys())}.")

        reference = references[name]
        ref_type = _validate_params_single(parameter, reference, name)
        if ref_type:
            ref_of = reference.get('of', None)
            if ref_of is None:
                # if ref_of is None, the elements are not to be validated
                continue
            elif ref_type == dict:
                _validate_params(parameter, ref_of, rec_name=name)
            else:  # List, tuple or ndarray type
                for i, parameter_elem in enumerate(parameter):
                    _validate_params_single(parameter_elem, ref_of,
                                            f"{name}[{i}]")


def validate_params(parameters, references, exclude=None):
    """Function to automate the validation of (hyper)parameters.

    Parameters
    ----------
    parameters : dict, required
        Dictionary in which the keys parameter names (as strings) and the
        corresponding values are parameter values. Unless `exclude` (see
        below) contains some of the keys in this dictionary, all parameters
        are checked against `references`.

    references : dict, required
        Dictionary in which the keys are parameter names (as strings). Let
        ``name`` and ``parameter`` denote a key-value pair in `parameters`.
        Since ``name`` should also be a key in `references`, let ``reference``
        be the corresponding value there. Then, ``reference`` must be a
        dictionary containing any of the following keys:

        - ``'type'``, mapping to a class or tuple of classes. ``parameter``
          is checked to be an instance of this class or tuple of classes.

        - ``'in'``, mapping to an object, when the value of ``'type'`` is
          not one of ``list``, ``tuple``, ``numpy.ndarray`` or ``dict``.
          Letting ``ref_in`` denote that object, the following check is
          performed: ``parameter in ref_in``.

        - ``'of'``, mapping to a dictionary, when the value of ``'type'``
          is one of ``list``, ``tuple``, ``numpy.ndarray`` or ``dict``.
          Let ``ref_of`` denote that dictionary. Then:

          a) If ``reference['type'] == dict`` – meaning that ``parameter``
             should be a dictionary – ``ref_of`` should have a similar
             structure as `references`, and :func:`validate_params` is called
             recursively on ``(parameter, ref_of)``.
          b) Otherwise, ``ref_of`` should have a similar structure as
             ``reference`` and each entry in ``parameter`` is checked to
             satisfy the constraints in ``ref_of``.

        - ``'other'``, which should map to a callable defining custom checks on
          ``parameter``.

    exclude : list or None, optional, default: ``None``
        List of parameter names which are among the keys in `parameters` but
        should be excluded from validation. ``None`` is equivalent to
        passing the empty list.

    """
    exclude_ = [] if exclude is None else exclude
    parameters_ = {key: value for key, value in parameters.items()
                   if key not in exclude_}
    return _validate_params(parameters_, references)


def check_point_clouds(X, distance_matrices=False, **kwargs):
    """Input validation on arrays or lists representing collections of point
    clouds or of distance/adjacency matrices.

    The input is checked to be either a single 3D array using a single call
    to :func:`sklearn.utils.validation.check_array`, or a list of 2D arrays by
    calling :func:`sklearn.utils.validation.check_array` on each entry.

    Parameters
    ----------
    X : object
        Input object to check / convert.

    distance_matrices : bool, optional, default: ``False``
        Whether the input represents a collection of distance matrices or of
        concrete point clouds in Euclidean space. In the first case, entries
        are allowed to be infinite unless otherwise specified in `kwargs`.

    **kwargs
        Keyword arguments accepted by
        :func:`sklearn.utils.validation.check_array`, with the following
        caveats: 1) `ensure_2d` and `allow_nd` are ignored; 2) if not passed
        explicitly, `force_all_finite` is set to be the boolean negation of
        `distance_matrices`; 3) when `force_all_finite` is set to ``False``,
        NaN inputs are not allowed; 4) `accept_sparse` and
        `accept_large_sparse` are only meaningful in the case of lists of 2D
        arrays, in which case they are passed to individual instances of
        :func:`sklearn.utils.validation.check_array` validating each entry
        in the list.

    Returns
    -------
    Xnew : ndarray or list
        The converted and validated object.

    """
    kwargs_ = {'force_all_finite': not distance_matrices}
    kwargs_.update(kwargs)
    kwargs_.pop('allow_nd', None)
    kwargs_.pop('ensure_2d', None)
    if hasattr(X, 'shape') and hasattr(X, 'ndim'):
        if X.ndim != 3:
            if X.ndim == 2:
                extra_2D = \
                    "\nReshape your input X using X.reshape(1, *X.shape) or " \
                    "X[None, :, :] if X is a single point cloud/distance " \
                    "matrix/adjacency matrix of a weighted graph."
            else:
                extra_2D = ""
            raise ValueError(
                f"Input must be a single 3D array or a list of 2D arrays or "
                f"sparse matrices. Structure of dimension {X.ndim} passed."
                + extra_2D
                )
        if (X.shape[1] != X.shape[2]) and distance_matrices:
            raise ValueError(
                f"Input array X must have X.shape[1] == X.shape[2]: "
                f"{X.shape[1]} != {X.shape[2]} passed.")
        elif (X.shape[1] == X.shape[2]) and not distance_matrices:
            warn(
                "Input array X has X.shape[1] == X.shape[2]. This is "
                "consistent with a collection of distance/adjacency "
                "matrices, but the input is being treated as a collection "
                "of vectors in Euclidean space.",
                DataDimensionalityWarning, stacklevel=2
                )
        Xnew = _check_array_mod(X, allow_nd=True, **kwargs_)
    else:
        has_check_failed = False
        messages = []
        Xnew = []
        for i, x in enumerate(X):
            try:
                xnew = _check_array_mod(x, ensure_2d=True, **kwargs_)
                if distance_matrices and not issparse(xnew):
                    if not x.shape[0] == x.shape[1]:
                        raise ValueError(
                            f"All arrays must be square: {x.shape[0]} rows "
                            f"and {x.shape[1]} columns found in this array."
                            )
                Xnew.append(xnew)
            except ValueError as e:
                has_check_failed = True
                messages.append(f"Entry {i}:\n{e}")
        if has_check_failed:
            raise ValueError(
                "The following errors were raised by the inputs:\n\n" +
                "\n\n".join(messages)
                )

        if not distance_matrices:
            if reduce(and_, (x.shape[0] == x.shape[1] for x in X), True):
                warn(
                    "All arrays/matrices are square. This is consistent with "
                    "a collection of distance/adjacency matrices, but the "
                    "entries will be treated as collections of vectors in "
                    "Euclidean space.", DataDimensionalityWarning,
                    stacklevel=2
                    )

        ref_dim = X[0].shape  # Shape of first sample
        if reduce(and_, (x.shape == ref_dim for x in X[1:]), True):
            Xnew = np.asarray(Xnew)

    return Xnew


def check_collection(X, **kwargs):
    """Generic input validation on arrays or lists of arrays.

    Parameters
    ----------
    X : object
        Input object to check / convert.

    **kwargs
        Keyword arguments accepted by
        :func:`sklearn.utils.validation.check_array`, with the following
        caveats: 1) `ensure_2d` and `allow_nd` are ignored; 2) when
        `force_all_finite` is set to ``False``, NaN inputs are not allowed.

    Returns
    -------
    Xnew : ndarray or list
        The converted and validated object.

    """
    kwargs_ = kwargs.copy()
    kwargs_.pop('allow_nd', None)
    kwargs_.pop('ensure_2d', None)
    if hasattr(X, 'shape') and hasattr(X, 'ndim'):
        Xnew = _check_array_mod(X, ensure_2d=True, allow_nd=True, **kwargs_)
    else:
        has_check_failed = False
        messages = []
        Xnew = []
        for i, x in enumerate(X):
            try:
                xnew = _check_array_mod(x, ensure_2d=False, allow_nd=True,
                                        **kwargs_)
                Xnew.append(xnew)
            except ValueError as e:
                has_check_failed = True
                messages.append(f"Entry {i}:\n{e}")
        if has_check_failed:
            raise ValueError(
                "The following errors were raised by the inputs:\n\n" +
                "\n\n".join(messages)
                )

    return Xnew
