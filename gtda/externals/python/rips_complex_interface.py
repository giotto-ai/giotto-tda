from ..modules.gtda_sparse_rips_complex \
    import Rips_complex_interface
from . import SimplexTree


# RipsComplex python interface
class RipsComplex:
    """The data structure is a one skeleton graph, or Rips graph, containing
    edges when the edge length is less or equal to a given threshold. Edge
    length is computed from a user given point cloud with a given distance
    function, or a distance matrix.
    """
    def __init__(self, points=None, distance_matrix=None,
                 max_edge_length=float('inf')):
        """RipsComplex constructor.
        :param max_edge_length: Rips value.
        :type max_edge_length: float
        :param points: A list of points in d-Dimension.
        :type points: list of list of double
        Or
        :param distance_matrix: A distance matrix (full square or lower
            triangular).
        :type points: list of list of double
        """
        self.thisref = Rips_complex_interface()

        if distance_matrix is not None:
            self.thisref.init_matrix(distance_matrix, max_edge_length)
        else:
            if points is None:
                # Empty Rips construction
                points = []
            self.thisref.init_points(points, max_edge_length)

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


# SparseRipsComplex python interface
class SparseRipsComplex:
    """The data structure is a one skeleton graph, or Rips graph, containing
    edges when the edge length is less or equal to a given threshold. Edge
    length is computed from a user given point cloud with a given distance
    function, or a distance matrix.
    Even truncated in filtration value and dimension, the Rips complex remains
    quite large. However, it is possible to approximate it by a much smaller
    filtered simplicial complex
    (linear size, with constants that depend on ε and the doubling dimension of
    the space) that is (1+O(ϵ))−interleaved with it (in particular, their
    persistence diagrams are at log-bottleneck distance at most O(ϵ)).
    """
    def __init__(self, points=None, distance_matrix=None,
                 max_edge_length=float('inf'), sparse=0.0):
        """SparseRipsComplex constructor.
        :param max_edge_length: Rips value.
        :type max_edge_length: float
        :param points: A list of points in d-Dimension.
        :type points: list of list of double
        Or
        :param distance_matrix: A distance matrix (full square or lower
            triangular).
        :type points: list of list of double
        And in both cases
        :param sparse: If this is not None, it switches to building a sparse
            Rips and represents the approximation parameter epsilon.
        :type sparse: float
        """
        self.thisref = Rips_complex_interface()

        if distance_matrix is not None:
            self.thisref.init_matrix_sparse(distance_matrix,
                                            max_edge_length,
                                            sparse)
        else:
            if points is None:
                # Empty Rips construction
                points = []
            self.thisref.init_points_sparse(points, max_edge_length,
                                            sparse)

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
