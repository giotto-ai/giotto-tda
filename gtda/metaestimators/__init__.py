"""The module :mod:`gtda.metaestimators` implements meta-estimators, i.e.
estimators which take other estimators as parameters."""

from .collection_transformer import CollectionTransformer

__all__ = [
    'CollectionTransformer'
    ]
