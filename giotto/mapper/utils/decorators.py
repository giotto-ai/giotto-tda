from sklearn.base import TransformerMixin


def method_to_transform(wrapped, method_name):
    """TODO: add documentation to this decorator
    """
    def wrapper(wrapped):
        class ExtendedEstimator(wrapped, TransformerMixin):
            def transform(self, X, y=None):
                has_transform = hasattr(wrapped, 'transform')
                has_method = hasattr(self, method_name)
                if (not has_transform) and has_method:
                    return getattr(self, method_name)(X)
        ExtendedEstimator.__name__ = 'Extended' + wrapped.__name__
        return ExtendedEstimator
    return wrapper(wrapped)
