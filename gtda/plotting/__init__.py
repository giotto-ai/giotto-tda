"""The module :mod:`gtda.plotting` implements function to plot the
outputs of giotto-tda transformers."""

from .point_clouds import plot_point_cloud
from .persistence_diagrams import plot_diagram
from .diagram_representations import plot_betti_curves, plot_betti_surfaces, \
    plot_landscapes, plot_persistence_image, plot_silhouettes, plot_heat_kernel
from .images import ImagePlotter

__all__ = [
    'plot_point_cloud',
    'plot_diagram',
    'plot_betti_curves',
    'plot_betti_surfaces',
    'plot_landscapes',
    'plot_persistence_image',
    'plot_silhouettes',
    'plot_heat_kernel',
    'ImagePlotter'
]
