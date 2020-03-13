from ..modules.gtda_cech_complex import Cech_complex_interface
from . import SimplexTree


# CechComplex python interface
class CechComplex:
    """ The data structure is a proximity graph, containing edges when the edge
    length is less or equal * to a given max_radius. The set of all simplices
    is filtered by the radius of their minimal enclosing ball.
    """
    def __init__(self, points, max_radius=0):
        """CechComplex constructor.
        :param points: A list of points in d-Dimension.
        :type points: list of coordinates of double
        :param max_radius: A distance matrix (full square or lower
            triangular).
        """
        self.thisref = Cech_complex_interface(points, max_radius)

    def __del__(self):
        if self.thisref is not None:
            del self.thisref

    def create_simplex_tree(self, max_dimension=1):
        """
        :param max_dimension: graph expansion for rips until this given maximal
            dimension.
        :type max_dimension: int
        :returns: A simplex tree created from the Delaunay Triangulation.
        :rtype: SimplexTree
        """
        simplex_tree = SimplexTree()
        self.thisref.create_simplex_tree(simplex_tree.thisptr, max_dimension)
        return simplex_tree
