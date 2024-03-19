"""Feature extraction from curves."""
# License: GNU AGPLv3

from copy import deepcopy
from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array

from ._functions import _AVAILABLE_FUNCTIONS, _implemented_function_recipes, \
    _parallel_featurization
from ..utils._docs import adapt_fit_transform_docs
from ..utils.validation import validate_params


@adapt_fit_transform_docs
class StandardFeatures(BaseEstimator, TransformerMixin):
    """Standard features from multi-channel curves.

    A multi-channel (integer sampled) curve is a 2D array of shape
    ``(n_channels, n_bins)``, where each row represents the y-values in one of
    the channels. This transformer applies scalar or vector-valued functions
    channel-wise to extract features from each multi-channel curve in a
    collection. The output is always a 2D array such that row ``i`` is the
    concatenation of the outputs of the chosen functions on the channels in the
    ``i``-th (multi-)curve in the collection.

    Parameters
    ----------
    function : string, callable, list or tuple, optional, default: ``"max"``
        Function or list/tuple of functions to apply to each channel of each
        multi-channel curve. Functions can map to scalars or to 1D arrays. If a
        string (see below) or a callable, then the same function is applied to
        all channels. Otherwise, `function` is a list/tuple of the same length
        as the number of entries along axis 1 in the collection passed to
        :meth:`fit`. Lists/tuples may contain allowed strings (see below),
        callables, and ``None`` in some positions to indicate that no feature
        should be extracted from the corresponding channel. Available strings
        are ``"identity"``, ``"argmin"``, ``"argmax"``, ``"min"``, ``"max"``,
        ``"mean"``, ``"std"``, ``"median"`` and ``"average"``.

    function_params : dict, None, list or tuple, optional, default: ``None``
        Additional keyword arguments for the function or functions in
        `function`. Passing ``None`` is equivalent to passing no arguments.
        Otherwise, if `function` is a single string or callable then
        `function_params` must be a dictionary. For functions encoded by
        allowed strings, the dictionary keys are as follows:

        - If ``function == "average"``, the only key is ``"weights"``
          (np.ndarray or None, default: ``None``).
        - Otherwise, there are no allowed keys.

        If `function` is a list or tuple, `function_params` must be a list or
        tuple of dictionaries (or ``None``) as above, of the same length as
        `function`.

    n_jobs : int or None, optional, default: ``None``
        The number of jobs to use for the computation. ``None`` means 1 unless
        in a :obj:`joblib.parallel_backend` context. ``-1`` means using all
        processors. Ignored if `function` is one of the allowed string options.

    Attributes
    ----------
    n_channels_ : int
        Number of channels present in the 3D array passed to :meth:`fit`. Must
        match the number of channels in the 3D array passed to
        :meth:`transform`.

    effective_function_ : callable or tuple
        Callable, or tuple of callables or ``None``, describing the function(s)
        used to compute features in each available channel. It is a single
        callable only when `function` was passed as a string.

    effective_function_params_ : dict or tuple
        Dictionary or tuple of dictionaries containing all information present
        in `function_params` as well as relevant quantities computed in
        :meth:`fit`. It is a single dict only when `function` was passed as a
        string. ``None``s are converted to empty dictionaries.

    """
    _hyperparameters = {
        "function": {"type": (str, Callable, list, tuple),
                     "in": tuple(_AVAILABLE_FUNCTIONS.keys()),
                     "of": {"type": (str, Callable, type(None)),
                            "in": tuple(_AVAILABLE_FUNCTIONS.keys())}},
        "function_params": {"type": (dict, type(None), list, tuple)},
        }

    def __init__(self, function="max", function_params=None, n_jobs=None):
        self.function = function
        self.function_params = function_params
        self.n_jobs = n_jobs

    def _validate_params(self):
        params = self.get_params().copy()
        _hyperparameters = deepcopy(self._hyperparameters)
        if not isinstance(self.function, str):
            _hyperparameters["function"].pop("in")
        try:
            validate_params(params, _hyperparameters, exclude=["n_jobs"])
        # Another go if we fail because function is a list/tuple containing
        # callables and the "in" key checks fail
        except ValueError as ve:
            end_string = f"which is not in " \
                         f"{tuple(_AVAILABLE_FUNCTIONS.keys())}."
            function = params["function"]
            if ve.args[0].endswith(end_string) \
                    and isinstance(function, (list, tuple)):
                params["function"] = [f for f in function
                                      if isinstance(f, str)]
                validate_params(params, _hyperparameters, exclude=["n_jobs"])
            else:
                raise ve

        if isinstance(self.function, (list, tuple)) \
                and isinstance(self.function_params, dict):
            raise TypeError("If `function` is a list/tuple then "
                            "`function_params` must be a list/tuple of dict, "
                            "or None.")
        elif isinstance(self.function, (str, Callable)) \
                and isinstance(self.function_params, (list, tuple)):
            raise TypeError("If `function` is a string or a callable "
                            "function then `function_params` must be a dict "
                            "or None.")

    def fit(self, X, y=None):
        """Compute :attr:`n_channels_` and :attr:`effective_function_params_`.
        Then, return the estimator.

        This function is here to implement the usual scikit-learn API and hence
        work in pipelines.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input data. Collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        self : object

        """
        check_array(X, ensure_2d=False, allow_nd=True)
        if X.ndim != 3:
            raise ValueError("Input must be 3-dimensional.")
        self._validate_params()

        self.n_channels_ = X.shape[1]

        if isinstance(self.function, str):
            self.effective_function_ = \
                _implemented_function_recipes[self.function]

            if self.function_params is None:
                self.effective_function_params_ = {}
            else:
                validate_params(self.function_params,
                                _AVAILABLE_FUNCTIONS[self.function])
                self.effective_function_params_ = self.function_params.copy()

        elif isinstance(self.function, Callable):
            self.effective_function_ = \
                tuple([self.function] * self.n_channels_)

            if self.function_params is None:
                self.effective_function_params_ = \
                    tuple([{}] * self.n_channels_)
            else:
                self.effective_function_params_ = \
                    tuple([self.function_params.copy()] * self.n_channels_)
        else:
            n_functions = len(self.function)
            if len(self.function) != self.n_channels_:
                raise ValueError(
                    f"`function` has length {n_functions} while curves in `X` "
                    f"have {self.n_channels_} channels."
                    )

            if self.function_params is None:
                self._effective_function_params = [{}] * self.n_channels_
            else:
                self._effective_function_params = self.function_params
                n_function_params = len(self._effective_function_params)
                if n_function_params != self.n_channels_:
                    raise ValueError(f"`function_params` has length "
                                     f"{n_function_params} while curves in "
                                     f"`X` have {self.n_channels_} channels.")

            self.effective_function_ = []
            self.effective_function_params_ = []
            for f, p in zip(self.function, self._effective_function_params):
                if isinstance(f, str):
                    validate_params(p, _AVAILABLE_FUNCTIONS[f])
                    self.effective_function_.\
                        append(_implemented_function_recipes[f])
                else:
                    self.effective_function_.append(f)
                self.effective_function_params_.append({} if p is None
                                                       else p.copy())
            self.effective_function_ = tuple(self.effective_function_)
            self.effective_function_params_ = \
                tuple(self.effective_function_params_)

        return self

    def transform(self, X, y=None):
        """Compute features of multi-channel curves.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_channels, n_bins)
            Input collection of multi-channel curves.

        y : None
            There is no need for a target in a transformer, yet the pipeline
            API requires this parameter.

        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Output collection of features of multi-channel curves.
            ``n_features`` is the sum of the number of features output by the
            (non-``None``) functions on their respective channels.

        """
        check_is_fitted(self)
        Xt = check_array(X, ensure_2d=False, allow_nd=True)
        if Xt.ndim != 3:
            raise ValueError("Input must be 3-dimensional.")
        if Xt.shape[1] != self.n_channels_:
            raise ValueError(f"Number of channels must be the same as in "
                             f"`fit`. Passed {Xt.shape[1]}, expected "
                             f"{self.n_channels_}.")

        Xt = _parallel_featurization(Xt, self.effective_function_,
                                     self.effective_function_params_,
                                     self.n_jobs)

        return Xt
