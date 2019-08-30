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

#ifndef HERA_BOTTLENECK_H
#define HERA_BOTTLENECK_H


#include <fstream>
#include <vector>
#include <algorithm>
#include <limits>
#include <random>

#include "diagram_traits.h"
#include "diagram_reader.h"
#include "bottleneck_detail.h"
#include "basic_defs_bt.h"
#include "bound_match.h"

namespace hera {
    // internal_p defines cost function on edges (use hera::get_infinity(),
    // if you want to explicitly refer to l_inf, but that's default value
    // delta is relative error, default is 1 percent
    template<class Real = double>
    struct BottleneckParams
    {
        Real internal_p { hera::get_infinity() };
        Real delta { 0.01 };
    };

    // functions taking containers as input
    // template parameter PairContainer must be a container of pairs of real
    // numbers (pair.first = x-coordinate, pair.second = y-coordinate)
    // PairContainer class must support iteration of the form
    // for(it = pairContainer.begin(); it != pairContainer.end(); ++it)

    // all functions in this header are wrappers around
    // functions from hera::bt namespace

    // get exact bottleneck distance,
    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistExact(PairContainer& dgm_A, PairContainer& dgm_B, const int decPrecision,
                        hera::bt::MatchingEdge<typename DiagramTraits<PairContainer>::RealType>& longest_edge,
                        bool compute_longest_edge = true)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(dgm_A);
        hera::bt::DiagramPointSet<Real> b(dgm_B);
        return hera::bt::bottleneckDistExact(a, b, decPrecision, longest_edge, compute_longest_edge);
    }

    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistExact(PairContainer& dgm_A, PairContainer& dgm_B, const int decPrecision)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::MatchingEdge<Real> longest_edge;
        return bottleneckDistExact(dgm_A, dgm_B, decPrecision, longest_edge, false);
    }


    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistExact(PairContainer& dgm_A, PairContainer& dgm_B)
    {
        int dec_precision = 14;
        return bottleneckDistExact(dgm_A, dgm_B, dec_precision);
    }


// return the interval (distMin, distMax) such that:
// a) actual bottleneck distance between A and B is contained in the interval
// b) if the interval is not (0,0), then  (distMax - distMin) / distMin < delta
    template<class PairContainer>
    std::pair<typename DiagramTraits<PairContainer>::RealType, typename DiagramTraits<PairContainer>::RealType>
    bottleneckDistApproxInterval(PairContainer& dgm_A, PairContainer& dgm_B,
                                 const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(dgm_A);
        hera::bt::DiagramPointSet<Real> b(dgm_B);
        return hera::bt::bottleneckDistApproxInterval(a, b, delta);
    }

// use sampling heuristic: discard most of the points with small persistency
// to get a good initial approximation of the bottleneck distance
    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistApproxHeur(PairContainer& dgm_A, PairContainer& dgm_B,
                             const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(dgm_A);
        hera::bt::DiagramPointSet<Real> b(dgm_B);
        std::pair<Real, Real> resPair = hera::bt::bottleneckDistApproxIntervalHeur(a, b, delta);
        return resPair.second;
    }

// get approximate distance,
// see bottleneckDistApproxInterval
    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistApprox(PairContainer& A, PairContainer& B,
                         const typename DiagramTraits<PairContainer>::RealType delta,
                         hera::bt::MatchingEdge<typename DiagramTraits<PairContainer>::RealType>& longest_edge,
                         bool compute_longest_edge = true)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::DiagramPointSet<Real> a(A.begin(), A.end());
        hera::bt::DiagramPointSet<Real> b(B.begin(), B.end());
        return hera::bt::bottleneckDistApprox(a, b, delta, longest_edge, compute_longest_edge);
    }

    template<class PairContainer>
    typename DiagramTraits<PairContainer>::RealType
    bottleneckDistApprox(PairContainer& A, PairContainer& B,
                         const typename DiagramTraits<PairContainer>::RealType delta)
    {
        using Real = typename DiagramTraits<PairContainer>::RealType;
        hera::bt::MatchingEdge<Real> longest_edge;
        return hera::bottleneckDistApprox(A, B, delta, longest_edge, false);
    }


} // end namespace hera

#endif
