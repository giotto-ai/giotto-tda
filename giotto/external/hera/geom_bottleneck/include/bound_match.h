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

#ifndef HERA_BOUND_MATCH_H
#define HERA_BOUND_MATCH_H

#include <unordered_map>
#include <memory>

#include "basic_defs_bt.h"
#include "neighb_oracle.h"


namespace hera {
namespace bt {

template<class Real = double>
class Matching {
public:
    using DgmPoint = DiagramPoint<Real>;
    using DgmPointSet = DiagramPointSet<Real>;
    using DgmPointHash = DiagramPointHash<Real>;
    using Path = std::vector<DgmPoint>;

    Matching(const DgmPointSet& AA, const DgmPointSet& BB) : A(AA), B(BB) {};
    DgmPointSet getExposedVertices(bool forA = true) const ;
    bool isExposed(const DgmPoint& p) const;
    void getAllAdjacentVertices(const DgmPointSet& setIn, DgmPointSet& setOut, bool forA = true) const;
    void increase(const Path& augmentingPath);
    void checkAugPath(const Path& augPath) const;
    bool getMatchedVertex(const DgmPoint& p, DgmPoint& result) const;
    bool isPerfect() const;
    void trimMatching(const Real newThreshold);
    MatchingEdge<Real> get_longest_edge() const;
#ifndef FOR_R_TDA
    template<class R>
    friend std::ostream& operator<<(std::ostream& output, const Matching<R>& m);
#endif
private:
    DgmPointSet A;
    DgmPointSet B;
    std::unordered_map<DgmPoint, DgmPoint, DgmPointHash> AToB, BToA;
    void matchVertices(const DgmPoint& pA, const DgmPoint& pB);
    void sanityCheck() const;
};



template<class Real_ = double, class NeighbOracle_ = NeighbOracleDnn<Real_>>
class BoundMatchOracle {
public:
    using Real = Real_;
    using NeighbOracle = NeighbOracle_;
    using DgmPoint = DiagramPoint<Real>;
    using DgmPointSet = DiagramPointSet<Real>;
    using Path = std::vector<DgmPoint>;

    BoundMatchOracle(DgmPointSet psA, DgmPointSet psB, Real dEps, bool useRS = true);
    bool isMatchLess(Real r);
    bool buildMatchingForThreshold(const Real r);
    MatchingEdge<Real> get_longest_edge() const { return M.get_longest_edge(); }
private:
    DgmPointSet A, B;
    Matching<Real> M;
    void printLayerGraph();
    void buildLayerGraph(Real r);
    void buildLayerOracles(Real r);
    bool buildAugmentingPath(const DgmPoint startVertex, Path& result);
    void removeFromLayer(const DgmPoint& p, const int layerIdx);
    std::unique_ptr<NeighbOracle> neighbOracle;
    bool augPathExist;
    std::vector<DgmPointSet> layerGraph;
    std::vector<std::unique_ptr<NeighbOracle>> layerOracles;
    Real distEpsilon;
    bool useRangeSearch;
    Real prevQueryValue;
};

} // end namespace bt
} // end namespace hera

#include "bound_match.hpp"

#endif // HERA_BOUND_MATCH_H
