import numpy as np
from ..modules.gtda_simplex_tree import *


class SimplexTree:
    """The simplex tree is an efficient and flexible data structure for
    representing general (filtered) simplicial complexes. The data structure
    is described in Jean-Daniel Boissonnat and Clément Maria. The Simplex
    Tree: An Efficient Data Structure for General Simplicial Complexes.
    Algorithmica, pages 1–22, 2014.
    This class is a filtered, with keys, and non contiguous vertices version
    of the simplex tree.
    """
    # cdef Simplex_tree_interface_full_featured * thisptr
    # cdef Simplex_tree_persistence_interface * pcohptr

    # Fake constructor that does nothing but documenting the constructor
    def __init__(self):
        """SimplexTree constructor.
        """
        self.thisptr = Simplex_tree_interface_full_featured()
        self.pcohptr = None

    def __del__(self):
        if self.thisptr is not None:
            del self.thisptr
        if self.pcohptr is not None:
            del self.pcohptr

    def __is_defined(self):
        """Returns true if SimplexTree pointer is not NULL.
         """
        if self.thisptr is not None:
            return True
        return False

    def __is_persistence_defined(self):
        """Returns true if Persistence pointer is not NULL.
         """
        if self.pcohptr is not None:
            return True
        return False

    def filtration(self, simplex):
        """This function returns the filtration value for a given N-simplex in
        this simplicial complex, or +infinity if it is not in the complex.
        :param simplex: The N-simplex, represented by a list of vertex.
        :type simplex: list of int.
        :returns:  The simplicial complex filtration value.
        :rtype:  float
        """
        return self.thisptr.simplex_filtration(simplex)

    def assign_filtration(self, simplex, filtration):
        """This function assigns the simplicial complex filtration value for a
        given N-simplex.
        :param simplex: The N-simplex, represented by a list of vertex.
        :type simplex: list of int.
        :param filtration:  The simplicial complex filtration value.
        :type filtration:  float
        """
        self.thisptr.assign_simplex_filtration(simplex, filtration)

    def initialize_filtration(self):
        """This function initializes and sorts the simplicial complex
        filtration vector.
        .. note::
            This function must be launched before
            :func:`persistence()<gudhi.SimplexTree.persistence>`,
            :func:`betti_numbers()<gudhi.SimplexTree.betti_numbers>`,
            :func:`persistent_betti_numbers()<gudhi.SimplexTree.persistent_betti_numbers>`,
            or :func:`get_filtration()<gudhi.SimplexTree.get_filtration>`
            after :func:`inserting<gudhi.SimplexTree.insert>` or
            :func:`removing<gudhi.SimplexTree.remove_maximal_simplex>`
            simplices.
        """
        self.thisptr.initialize_filtration()

    def num_vertices(self):
        """This function returns the number of vertices of the simplicial
        complex.
        :returns:  The simplicial complex number of vertices.
        :rtype:  int
        """
        return self.thisptr.num_vertices()

    def num_simplices(self):
        """This function returns the number of simplices of the simplicial
        complex.
        :returns:  the simplicial complex number of simplices.
        :rtype:  int
        """
        return self.thisptr.num_simplices()

    def dimension(self):
        """This function returns the dimension of the simplicial complex.
        :returns:  the simplicial complex dimension.
        :rtype:  int
        .. note::
            This function is not constant time because it can recompute
            dimension if required (can be triggered by
            :func:`remove_maximal_simplex()<gudhi.SimplexTree.remove_maximal_simplex>`
            or
            :func:`prune_above_filtration()<gudhi.SimplexTree.prune_above_filtration>`
            methods).
        """
        return self.thisptr.dimension()

    def upper_bound_dimension(self):
        """This function returns a valid dimension upper bound of the
        simplicial complex.
        :returns:  an upper bound on the dimension of the simplicial complex.
        :rtype:  int
        """
        return self.thisptr.upper_bound_dimension()

    def set_dimension(self, dimension):
        """This function sets the dimension of the simplicial complex.
        :param dimension: The new dimension value.
        :type dimension: int.
        .. note::
            This function must be used with caution because it disables
            dimension recomputation when required
            (this recomputation can be triggered by
            :func:`remove_maximal_simplex()<gudhi.SimplexTree.remove_maximal_simplex>`
            or
            :func:`prune_above_filtration()<gudhi.SimplexTree.prune_above_filtration>`
            ).
        """
        self.thisptr.set_dimension(dimension)

    def find(self, simplex):
        """This function returns if the N-simplex was found in the simplicial
        complex or not.
        :param simplex: The N-simplex to find, represented by a list of vertex.
        :type simplex: list of int.
        :returns:  true if the simplex was found, false otherwise.
        :rtype:  bool
        """
        csimplex = [i for i in simplex]
        return self.thisptr.find_simplex(csimplex)

    def insert(self, simplex, filtration=0.0):
        """This function inserts the given N-simplex and its subfaces with the
        given filtration value (default value is '0.0'). If some of those
        simplices are already present with a higher filtration value, their
        filtration value is lowered.
        :param simplex: The N-simplex to insert, represented by a list of
            vertex.
        :type simplex: list of int.
        :param filtration: The filtration value of the simplex.
        :type filtration: float.
        :returns:  true if the simplex was not yet in the complex, false
            otherwise (whatever its original filtration value).
        :rtype:  bool
        """
        csimplex = [i for i in simplex]
        return self.thisptr.insert_simplex_and_subfaces(csimplex,
                                                        filtration)

    def get_filtration(self):
        """This function returns a list of all simplices with their given
        filtration values.
        :returns:  The simplices sorted by increasing filtration values.
        :rtype:  list of tuples(simplex, filtration)
        """
        filtration = self.thisptr.get_filtration()
        ct = []
        for filtered_complex in filtration:
            v = [vertex for vertex in filtered_complex[0]]
            ct.append((v, filtered_complex[1]))
        return ct

    def get_skeleton(self, dimension):
        """This function returns the (simplices of the) skeleton of a maximum
        given dimension.
        :param dimension: The skeleton dimension value.
        :type dimension: int.
        :returns:  The (simplices of the) skeleton of a maximum dimension.
        :rtype:  list of tuples(simplex, filtration)
        """
        skeleton = self.thisptr.get_skeleton(dimension)
        ct = []
        for filtered_simplex in skeleton:
            v = [vertex for vertex in filtered_simplex[0]]
            ct.append((v, filtered_simplex[1]))
        return ct

    def get_star(self, simplex):
        """This function returns the star of a given N-simplex.
        :param simplex: The N-simplex, represented by a list of vertex.
        :type simplex: list of int.
        :returns:  The (simplices of the) star of a simplex.
        :rtype:  list of tuples(simplex, filtration)
        """
        csimplex = [i for i in simplex]
        star = self.thisptr.get_star(csimplex)
        ct = []
        for filtered_simplex in star:
            v = [vertex for vertex in filtered_simplex[0]]
            ct.append((v, filtered_simplex[1]))
        return ct

    def get_cofaces(self, simplex, codimension):
        """This function returns the cofaces of a given N-simplex with a
        given codimension.
        :param simplex: The N-simplex, represented by a list of vertex.
        :type simplex: list of int.
        :param codimension: The codimension. If codimension = 0, all cofaces
            are returned (equivalent of get_star function)
        :type codimension: int.
        :returns:  The (simplices of the) cofaces of a simplex
        :rtype:  list of tuples(simplex, filtration)
        """
        csimplex = [i for i in simplex]
        cofaces = self.thisptr.get_cofaces(csimplex, codimension)
        ct = []
        for filtered_simplex in cofaces:
            v = [vertex for vertex in filtered_simplex[0]]
            ct.append((v, filtered_simplex[1]))
        return ct

    def remove_maximal_simplex(self, simplex):
        """This function removes a given maximal N-simplex from the simplicial
        complex.
        :param simplex: The N-simplex, represented by a list of vertex.
        :type simplex: list of int.
        .. note::
            Be aware that removing is shifting data in a flat_map
            (:func:`initialize_filtration()<gudhi.SimplexTree.initialize_filtration>` to be done).
        .. note::
            The dimension of the simplicial complex may be lower after calling
            remove_maximal_simplex than it was before. However,
            :func:`upper_bound_dimension()<gudhi.SimplexTree.upper_bound_dimension>`
            method will return the old value, which
            remains a valid upper bound. If you care, you can call
            :func:`dimension()<gudhi.SimplexTree.dimension>`
            to recompute the exact dimension.
        """
        self.thisptr.remove_maximal_simplex(simplex)

    def prune_above_filtration(self, filtration):
        """Prune above filtration value given as parameter.
        :param filtration: Maximum threshold value.
        :type filtration: float.
        :returns: The filtration modification information.
        :rtype: bool
        .. note::
            Some simplex tree functions require the filtration to be valid.
            prune_above_filtration function is not launching
            :func:`initialize_filtration()<gudhi.SimplexTree.initialize_filtration>`
            but returns the filtration modification
            information. If the complex has changed , please call
            :func:`initialize_filtration()<gudhi.SimplexTree.initialize_filtration>`
            to recompute it.
        .. note::
            Note that the dimension of the simplicial complex may be lower
            after calling
            :func:`prune_above_filtration()<gudhi.SimplexTree.prune_above_filtration>`
            than it was before. However,
            :func:`upper_bound_dimension()<gudhi.SimplexTree.upper_bound_dimension>`
            will return the old value, which remains a
            valid upper bound. If you care, you can call
            :func:`dimension()<gudhi.SimplexTree.dimension>`
            method to recompute the exact dimension.
        """
        return self.thisptr.prune_above_filtration(filtration)

    def expansion(self, max_dim):
        """Expands the Simplex_tree containing only its one skeleton
        until dimension max_dim.
        The expanded simplicial complex until dimension :math:`d`
        attached to a graph :math:`G` is the maximal simplicial complex of
        dimension at most :math:`d` admitting the graph :math:`G` as
        :math:`1`-skeleton.
        The filtration value assigned to a simplex is the maximal filtration
        value of one of its edges.
        The Simplex_tree must contain no simplex of dimension bigger than
        1 when calling the method.
        :param max_dim: The maximal dimension.
        :type max_dim: int.
        """
        self.thisptr.expansion(max_dim)

    def make_filtration_non_decreasing(self):
        """This function ensures that each simplex has a higher filtration
        value than its faces by increasing the filtration values.
        :returns: True if any filtration value was modified,
        False if the filtration was already non-decreasing.
        :rtype: bool
        .. note::
            Some simplex tree functions require the filtration to be valid.
            make_filtration_non_decreasing function is not launching
            :func:`initialize_filtration()<gudhi.SimplexTree.initialize_filtration>`
            but returns the filtration modification
            information. If the complex has changed , please call
            :func:`initialize_filtration()<gudhi.SimplexTree.initialize_filtration>`
            to recompute it.
        """
        return self.thisptr.make_filtration_non_decreasing()

    def persistence(self, homology_coeff_field=11, min_persistence=0,
                    persistence_dim_max = False):
        """This function returns the persistence of the simplicial complex.
        :param homology_coeff_field: The homology coefficient field. Must be a
            prime number. Default value is 11.
        :type homology_coeff_field: int.
        :param min_persistence: The minimum persistence value to take into
            account (strictly greater than min_persistence). Default value is
            0.0.
            Sets min_persistence to -1.0 to see all values.
        :type min_persistence: float.
        :param persistence_dim_max: If true, the persistent homology for the
            maximal dimension in the complex is computed. If false, it is
            ignored. Default is false.
        :type persistence_dim_max: bool
        :returns: The persistence of the simplicial complex.
        :rtype:  list of pairs(dimension, pair(birth, death))
        """
        if self.pcohptr is not None:
            del self.pcohptr
        self.pcohptr = Simplex_tree_persistence_interface(self.thisptr,
                                                          persistence_dim_max)
        persistence_result = []
        if self.pcohptr is not None:
            persistence_result = \
                self.pcohptr.get_persistence(homology_coeff_field,
                                             min_persistence)
        return persistence_result

    def betti_numbers(self):
        """This function returns the Betti numbers of the simplicial complex.
        :returns: The Betti numbers ([B0, B1, ..., Bn]).
        :rtype:  list of int
        :note: betti_numbers function requires
            :func:`persistence()<gudhi.SimplexTree.persistence>`
            function to be launched first.
        """
        bn_result = []
        if self.pcohptr is not None:
            bn_result = self.pcohptr.betti_numbers()
        else:
            print("betti_numbers function requires persistence function"
                  " to be launched first.")
        return bn_result

    def persistent_betti_numbers(self, from_value, to_value):
        """This function returns the persistent Betti numbers of the
        simplicial complex.
        :param from_value: The persistence birth limit to be added in the
            numbers (persistent birth <= from_value).
        :type from_value: float.
        :param to_value: The persistence death limit to be added in the
            numbers (persistent death > to_value).
        :type to_value: float.
        :returns: The persistent Betti numbers ([B0, B1, ..., Bn]).
        :rtype:  list of int
        :note: persistent_betti_numbers function requires
            :func:`persistence()<gudhi.SimplexTree.persistence>`
            function to be launched first.
        """
        pbn_result = []
        if self.pcohptr is not None:
            pbn_result = self.pcohptr.persistent_betti_numbers(from_value,
                                                               to_value)
        else:
            print("persistent_betti_numbers function requires persistence function"
                  " to be launched first.")
        return pbn_result

    def persistence_intervals_in_dimension(self, dimension):
        """This function returns the persistence intervals of the simplicial
        complex in a specific dimension.
        :param dimension: The specific dimension.
        :type dimension: int.
        :returns: The persistence intervals.
        :rtype:  numpy array of dimension 2
        :note: intervals_in_dim function requires
            :func:`persistence()<gudhi.SimplexTree.persistence>`
            function to be launched first.
        """
        intervals_result = []
        if self.pcohptr is not None:
            intervals_result = self.pcohptr.intervals_in_dimension(dimension)
        else:
            print("intervals_in_dim function requires persistence function"
                  " to be launched first.")
        return np.array(intervals_result)

    def persistence_pairs(self):
        """This function returns a list of persistence birth and death simplices pairs.
        :returns: A list of persistence simplices intervals.
        :rtype:  list of pair of list of int
        :note: persistence_pairs function requires
            :func:`persistence()<gudhi.SimplexTree.persistence>`
            function to be launched first.
        """
        persistence_pairs_result = []
        if self.pcohptr is not None:
            persistence_pairs_result = self.pcohptr.persistence_pairs()
        else:
            print("persistence_pairs function requires persistence function"
                  " to be launched first.")
        return persistence_pairs_result

    def write_persistence_diagram(self, persistence_file=''):
        """This function writes the persistence intervals of the simplicial
        complex in a user given file name.
        :param persistence_file: The specific dimension.
        :type persistence_file: string.
        :note: intervals_in_dim function requires
            :func:`persistence()<gudhi.SimplexTree.persistence>`
            function to be launched first.
        """
        if self.pcohptr is not None:
            if persistence_file != '':
                self.pcohptr.write_output_diagram(str.encode(persistence_file))
            else:
                print("persistence_file must be specified")
        else:
            print("intervals_in_dim function requires persistence function"
                  " to be launched first.")
