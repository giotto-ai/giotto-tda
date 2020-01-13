"""The module :mod:`giotto.mapper` implements the Mapper algorithm for
topological clustering and visualisation."""

from .cluster import FirstHistogramGap, FirstSimpleGap
from .cover import CubicalCover, OneDimensionalCover
from .filter import Eccentricity, Entropy, Projection
from .pipeline import make_mapper_pipeline
from .visualization import (
    plot_static_mapper_graph, plot_interactive_mapper_graph)

__all__ = [
    'Projection',
    'Eccentricity',
    'Entropy',
    'OneDimensionalCover',
    'CubicalCover',
    'FirstSimpleGap',
    'FirstHistogramGap',
    'make_mapper_pipeline',
    'plot_static_mapper_graph',
    'plot_interactive_mapper_graph'
]
