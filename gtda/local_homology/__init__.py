"""The module :mod:`gtda.local` implements transformers to generate
local persistence diagrams."""

# The LocalVietorisBase was ignored, since it is not to be used.
from .simplicial import KNeighborsLocalVietorisRips, \
    RadiusLocalVietorisRips 

__all__ = [
    'KNeighborsLocalVietorisRips',
    'RadiusLocalVietorisRips',
    ]