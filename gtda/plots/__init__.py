"""The module :mod:`gtda.plots` implements function to plot
giotto-tda transformers' input/output."""

from .point_clouds import plot_point_cloud
from .homology import plot_diagram
from .diagrams import plot_betti_curves, plot_betti_surfaces, plot_landscapes,\
    plot_persistence_image
from .image import plot_image

__all__ = [
    'plot_point_cloud',
    'plot_diagram',
    'plot_betti_curves',
    'plot_betti_surfaces',
    'plot_landscapes',
    'plot_image',
    'plot_persistence_image',
]
