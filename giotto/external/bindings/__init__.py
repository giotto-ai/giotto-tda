import pyximport
pyximport.install(language_level=3)

try:
    from .hera_wasserstein import wasserstein
except ImportError:
    def wasserstein(diagram_1, diagram_2, p=1, delta=0.01):
        pass
    print("Function wasserstein_distance not available.")
