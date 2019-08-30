/*

Copyright (c) 2015, M. Kerber, D. Morozov, A. Nigmetov
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


You are under no obligation whatsoever to provide any bug fixes, patches, or
upgrades to the features, functionality or performance of the source code
(Enhancements) to anyone; however, if you choose to make your Enhancements
available either publicly, or directly to copyright holder,
without imposing a separate written license agreement for such Enhancements,
then you hereby grant the following license: a  non-exclusive, royalty-free
perpetual license to install, use, modify, prepare derivative works, incorporate
into other computer software, distribute, and sublicense such enhancements or
derivative works thereof, in binary and source code form.

*/

#ifndef HERA_BOTTLENECK_DETAIL_H
#define HERA_BOTTLENECK_DETAIL_H


#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>

#include "diagram_traits.h"
#include "basic_defs_bt.h"
#include "bound_match.h"

namespace hera {


    namespace bt {

        // functions taking DiagramPointSet as input.
        // ATTENTION: parameters A and B (diagrams) will be changed after the call
        // (projections added).

        // return the interval (distMin, distMax) such that:
        // a) actual bottleneck distance between A and B is contained in the interval
        // b) if the interval is not (0,0), then  (distMax - distMin) / distMin < epsilon
        template<class Real>
        std::pair<Real, Real> bottleneckDistApproxInterval(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B,
                                                           const Real epsilon, MatchingEdge<Real>& longest_edge,
                                                           bool compute_longest_edge = false);


        // heuristic (sample diagram to estimate the distance)
        template<class Real>
        std::pair<Real, Real>
        bottleneckDistApproxIntervalHeur(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B, const Real epsilon,
                                         bool compute_longest_edge = false);

        // get approximate distance,
        // see bottleneckDistApproxInterval
        template<class Real>
        Real bottleneckDistApprox(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B, const Real epsilon,
                                  MatchingEdge<Real>& longest_edge, bool compute_longest_edge = false);

        // get exact bottleneck distance,
        template<class Real>
        Real bottleneckDistExact(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B, const int decPrecision,
                                 MatchingEdge<Real>& longest_edge, bool compute_longest_edge = false);

        // get exact bottleneck distance,
        template<class Real>
        Real bottleneckDistExact(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B, MatchingEdge<Real>& longest_edge,
                                 bool compute_longest_edge = false);

    } // end namespace bt


} // end namespace hera

#include "bottleneck_detail.hpp"

#endif  // HERA_BOTTLENECK_DETAIL_H
