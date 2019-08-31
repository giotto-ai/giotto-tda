import pyximport
pyximport.install()
from distutils.errors import CompileError

try:
    from .hera_wasserstein import wasserstein
except CompileError:
    def wasserstein(diagram_1, diagram_2, p=1, delta=0.01):
        pass
    print("Function wasserstein_distance not available.")
