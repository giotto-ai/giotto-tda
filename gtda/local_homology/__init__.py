"""The module :mod:`gtda.local_homology` implements transformers
to generate local persistence diagrams."""


from .simplicial import KNeighborsLocalVietorisRips, \
    RadiusLocalVietorisRips

__all__ = [
    'KNeighborsLocalVietorisRips',
    'RadiusLocalVietorisRips',
    ]
