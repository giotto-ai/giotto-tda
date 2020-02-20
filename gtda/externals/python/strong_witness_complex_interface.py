from ..modules.gtda_strong_witness_complex \
    import Strong_witness_complex_interface
from . import SimplexTree


# StrongWitnessComplex python interface
class StrongWitnessComplex:
    """Constructs (strong) witness complex for a given table of nearest
    landmarks with respect to witnesses.
    """

    def __init__(self, nearest_landmark_table=None):
        """StrongWitnessComplex constructor.
        :param nearest_landmark_table: A list of lists of nearest landmarks and
        their distances.  `nearest_landmark_table[w][k]==(l,d)` means that l is
        the k-th nearest landmark to
            witness w, and d is the (squared) distance between l and w.
        :type nearest_landmark_table: list of list of pair of int and float
        """
        self.thisptr = None

        if nearest_landmark_table is not None:
            self.thisptr = \
                Strong_witness_complex_interface(nearest_landmark_table)

    def __del__(self):
        if self.thisptr is not None:
            del self.thisptr

    def __is_defined(self):
        """Returns true if WitnessComplex pointer is not NULL.
         """
        if self.thisptr is not None:
            return True
        return False

    def create_simplex_tree(self, max_alpha_square=float('inf'),
                            limit_dimension=-1):
        """
        :param max_alpha_square: The maximum relaxation parameter.
            Default is set to infinity.
        :type max_alpha_square: float
        :returns: A simplex tree created from the Delaunay Triangulation.
        :rtype: SimplexTree
        """
        stree = SimplexTree()
        stree_int_ptr = stree.thisptr
        if limit_dimension != -1:
            self.thisptr.create_simplex_tree(stree_int_ptr, max_alpha_square,
                                             limit_dimension)
        else:
            self.thisptr.create_simplex_tree(stree_int_ptr, max_alpha_square)
        return stree
