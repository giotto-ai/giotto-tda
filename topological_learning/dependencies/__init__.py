import pyximport
pyximport.install()

try:
    #from .gudhi_bottleneck import bottleneck_distance
    raise ImportError()
except ImportError:
    from gudhi import bottleneck_distance
    print("Using original gudhi bottleneck_distance.")

from .hera_wasserstein import wasserstein
