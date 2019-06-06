from cython cimport numeric
from libcpp.vector cimport vector
from libcpp.utility cimport pair
import os

"""This file is part of the Gudhi Library. The Gudhi library
   (Geometric Understanding in Higher Dimensions) is a generic C++
   library for computational topology.

   Author(s):       Vincent Rouvreau

   Copyright (C) 2016 Inria

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__author__ = "Vincent Rouvreau"
__copyright__ = "Copyright (C) 2016 Inria"
__license__ = "GPL v3"

cdef extern from "Bottleneck_distance_interface.h" namespace "Gudhi::persistence_diagram":
    double bottleneck(vector[pair[double, double]], vector[pair[double, double]], double)
    double bottleneck(vector[pair[double, double]], vector[pair[double, double]])
    #vector[vector[double]] bottleneck_matrix(vector[vector[pair[double, double]]])

def bottleneck_distance(diagram_1, diagram_2, e=None):
    """This function returns the point corresponding to a given vertex.

    :param diagram_1: The first diagram.
    :type diagram_1: vector[pair[double, double]]
    :param diagram_2: The second diagram.
    :type diagram_2: vector[pair[double, double]]
    :param e: If `e` is 0, this uses an expensive algorithm to compute the
        exact distance.
        If `e` is not 0, it asks for an additive `e`-approximation, and
        currently also allows a small multiplicative error (the last 2 or 3
        bits of the mantissa may be wrong). This version of the algorithm takes
        advantage of the limited precision of `double` and is usually a lot
        faster to compute, whatever the value of `e`.

        Thus, by default, `e` is the smallest positive double.
    :type e: float
    :rtype: float
    :returns: the bottleneck distance.
    """
    if e is None:
        # Default value is the smallest double value (not 0, 0 is for exact version)
        return bottleneck(diagram_1, diagram_2)
    else:
        # Can be 0 for exact version
        return bottleneck(diagram_1, diagram_2, e)
