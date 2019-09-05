import pyximport
pyximport.install()

try:
    from .hera_wasserstein import wasserstein
except ImportError:
    def wasserstein(diagram_1, diagram_2, p=1, delta=0.01):
        pass
    print("Function wasserstein_distance not available.")
