"""Convenience class decorators for use in a Mapper context."""
# License: GNU AGPLv3

from sklearn.base import TransformerMixin


def method_to_transform(cls, method_name):
    """Wrap a class to add a :meth:`transform` method as an alias to an
    existing method.

    An example of use is for classes possessing a :meth:`score` method such as
    kernel density estimators and anomaly/novelty detection estimators,
    allow for these estimators are to be used as steps in a pipeline.

    Note that 1D array outputs are reshaped into 2D column vectors before
    being returned by the new :meth:`transform`.

    Parameters
    ----------
    cls : object
        Class to be wrapped. If `method_name` is not one of its methods,
        :meth:`transform` always returns ``None``.

    method_name : str
        Name of the method in `cls` to which :meth:`transform` will be
        an alias. The fist argument of this method (after ``self``) becomes
        the ``X`` input for :meth:`transform`.

    Returns
    -------
    wrapped_cls : object
        New class inheriting from :class:`sklearn.base.TransformerMixin`, so
        that both :meth:`transform` and :meth:`fit_transform` are available.
        Its name is the name of `cls` prepended with ``'Extended'``.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.neighbors import KernelDensity
    >>> from gtda.mapper import method_to_transform
    >>> X = np.random.random((100, 2))
    >>> kde = KernelDensity()

    Extend ``KernelDensity`` to give it a ``transform`` method as an alias
    of ``score_samples`` (up to output shape). The new class is instantiated
    with the same parameters as the original one.

    >>> ExtendedKDE = method_to_transform(KernelDensity, 'score_samples')
    >>> extended_kde = ExtendedKDE()
    >>> Xt = kde.fit(X).score_samples(X)
    >>> print(Xt.shape)
    (100,)
    >>> Xt_extended = extended_kde.fit_transform(X)
    >>> print(Xt_extended.shape)
    (100, 1)
    >>> np.array_equal(Xt, Xt_extended.flatten())
    True

    """
    class ExtendedEstimator(cls, TransformerMixin):
        def transform(self, X, y=None):
            has_method = hasattr(self, method_name)
            if has_method:
                Xt = getattr(self, method_name)(X)
                # reshape 1D estimators to have shape (n_samples, 1)
                if Xt.ndim == 1:
                    Xt = Xt[:, None]
                return Xt

    ExtendedEstimator.__name__ = 'Extended' + cls.__name__

    return ExtendedEstimator
