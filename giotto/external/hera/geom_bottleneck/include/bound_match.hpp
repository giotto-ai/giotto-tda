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

#ifndef HERA_BOUND_MATCH_HPP
#define HERA_BOUND_MATCH_HPP


#ifdef FOR_R_TDA
#undef DEBUG_BOUND_MATCH
#undef DEBUG_MATCHING
#undef VERBOSE_BOTTLENECK
#endif

#include <assert.h>
#include "def_debug_bt.h"
#include "bound_match.h"

#ifdef VERBOSE_BOTTLENECK
#include <chrono>
#endif

#ifndef FOR_R_TDA

#include <iostream>

#endif

namespace hera {
    namespace bt {

#ifndef FOR_R_TDA

        template<class Real>
        std::ostream& operator<<(std::ostream& output, const Matching <Real>& m)
        {
            output << "Matching: " << m.AToB.size() << " pairs (";
            if (!m.isPerfect()) {
                output << "not";
            }
            output << " perfect)" << std::endl;
            for (auto atob : m.AToB) {
                output << atob.first << " <-> " << atob.second << "  distance: " << distLInf(atob.first, atob.second)
                       << std::endl;
            }
            return output;
        }

#endif

        template<class R>
        void Matching<R>::sanityCheck() const
        {
#ifdef DEBUG_MATCHING
            assert( AToB.size() == BToA.size() );
            for(auto aToBPair : AToB) {
                auto bToAPair = BToA.find(aToBPair.second);
                assert(bToAPair != BToA.end());
                assert( aToBPair.first == bToAPair->second);
            }
#endif
        }

        template<class R>
        bool Matching<R>::isPerfect() const
        {
            return AToB.size() == A.size();
        }

        template<class R>
        void Matching<R>::matchVertices(const DgmPoint& pA, const DgmPoint& pB)
        {
            assert(A.hasElement(pA));
            assert(B.hasElement(pB));
            AToB.erase(pA);
            AToB.insert({{ pA, pB }});
            BToA.erase(pB);
            BToA.insert({{ pB, pA }});
        }

        template<class R>
        bool Matching<R>::getMatchedVertex(const DgmPoint& p, DgmPoint& result) const
        {
            sanityCheck();
            auto inA = AToB.find(p);
            if (inA != AToB.end()) {
                result = inA->second;
                return true;
            } else {
                auto inB = BToA.find(p);
                if (inB != BToA.end()) {
                    result = inB->second;
                    return true;
                }
            }
            return false;
        }


        template<class R>
        void Matching<R>::checkAugPath(const Path& augPath) const
        {
            assert(augPath.size() % 2 == 0);
            for (size_t idx = 0; idx < augPath.size(); ++idx) {
                bool mustBeExposed { idx == 0 or idx == augPath.size() - 1 };
                if (isExposed(augPath[idx]) != mustBeExposed) {
#ifndef FOR_R_TDA
                    std::cerr << "mustBeExposed = " << mustBeExposed << ", idx = " << idx << ", point " << augPath[idx]
                              << std::endl;
#endif
                }
                assert(isExposed(augPath[idx]) == mustBeExposed);
                DgmPoint matchedVertex;
                if (idx % 2 == 0) {
                    assert(A.hasElement(augPath[idx]));
                    if (not mustBeExposed) {
                        getMatchedVertex(augPath[idx], matchedVertex);
                        assert(matchedVertex == augPath[idx - 1]);
                    }
                } else {
                    assert(B.hasElement(augPath[idx]));
                    if (not mustBeExposed) {
                        getMatchedVertex(augPath[idx], matchedVertex);
                        assert(matchedVertex == augPath[idx + 1]);
                    }
                }
            }
        }

        // use augmenting path to increase
        // the size of the matching
        template<class R>
        void Matching<R>::increase(const Path& augPath)
        {
            sanityCheck();
            // check that augPath is an augmenting path
            checkAugPath(augPath);
            for (size_t idx = 0; idx < augPath.size() - 1; idx += 2) {
                matchVertices(augPath[idx], augPath[idx + 1]);
            }
            sanityCheck();
        }

        template<class R>
        DiagramPointSet <R> Matching<R>::getExposedVertices(bool forA) const
        {
            sanityCheck();
            DgmPointSet result;
            const DgmPointSet* setToSearch { forA ? &A : &B };
            const std::unordered_map<DgmPoint, DgmPoint, DgmPointHash>* mapToSearch { forA ? &AToB : &BToA };
            for (auto it = setToSearch->cbegin(); it != setToSearch->cend(); ++it) {
                if (mapToSearch->find((*it)) == mapToSearch->cend()) {
                    result.insert((*it));
                }
            }
            return result;
        }

        template<class R>
        void Matching<R>::getAllAdjacentVertices(const DgmPointSet& setIn,
                                                 DgmPointSet& setOut,
                                                 bool forA) const
        {
            sanityCheck();
            //bool isDebug {false};
            setOut.clear();
            const std::unordered_map<DgmPoint, DgmPoint, DgmPointHash>* m;
            m = (forA) ? &BToA : &AToB;
            for (auto pit = setIn.cbegin(); pit != setIn.cend(); ++pit) {
                auto findRes = m->find(*pit);
                if (findRes != m->cend()) {
                    setOut.insert((*findRes).second);
                }
            }
            sanityCheck();
        }

        template<class R>
        bool Matching<R>::isExposed(const DgmPoint& p) const
        {
            return (AToB.find(p) == AToB.end()) && (BToA.find(p) == BToA.end());
        }

        // remove all edges whose length is > newThreshold
        template<class R>
        void Matching<R>::trimMatching(const R newThreshold)
        {
            //bool isDebug { false };
            sanityCheck();
            for (auto aToBIter = AToB.begin(); aToBIter != AToB.end();) {
                if (distLInf(aToBIter->first, aToBIter->second) > newThreshold) {
                    // remove edge from AToB and BToA
                    BToA.erase(aToBIter->second);
                    aToBIter = AToB.erase(aToBIter);
                } else {
                    aToBIter++;
                }
            }
            sanityCheck();
        }

        template<class R>
        MatchingEdge <R> Matching<R>::get_longest_edge() const
        {
            R max_dist = -1.0;
            MatchingEdge<R> edge;
            for (const auto& x : AToB) {
                //std::cout << "max_dist = " << max_dist << std::endl;
                //std::cout << "distance = " << distLInf(x.first, x.second) << std::endl;

                // for now skew edges may appear in the matching
                // but they should not be returned to user
                // if currrent edge is a skew edge, there must another edge
                // with the same cost
                R curr_dist;
                if (x.first.isDiagonal() and x.second.isNormal()) {
                    curr_dist = x.second.get_persistence(hera::get_infinity());
                } else if (x.first.isNormal() and x.second.isDiagonal()) {
                    curr_dist = x.first.get_persistence(hera::get_infinity());
                } else {
                    curr_dist = distLInf(x.first, x.second);
                }
                if (max_dist < curr_dist) {
                    max_dist = curr_dist;
                    edge = x;
                    //std::cout << "updated max_dist = " << max_dist << std::endl;
                    //std::cout << "updated edge = " << x.first << " <-> " << x.second << std::endl;
                }
            }
            return edge;
        }

        // ------- BoundMatchOracle --------------

        template<class R, class NO>
        BoundMatchOracle<R, NO>::BoundMatchOracle(DgmPointSet psA, DgmPointSet psB,
                                                  Real dEps, bool useRS) :
                A(psA), B(psB), M(A, B), distEpsilon(dEps), useRangeSearch(useRS), prevQueryValue(0.0)
        {
            neighbOracle = std::unique_ptr<NeighbOracle>(new NeighbOracle(psB, 0, distEpsilon));
        }

        template<class R, class NO>
        bool BoundMatchOracle<R, NO>::isMatchLess(Real r)
        {
#ifdef VERBOSE_BOTTLENECK
            std::chrono::high_resolution_clock hrClock;
            std::chrono::time_point<std::chrono::high_resolution_clock> startMoment;
            startMoment = hrClock.now();
#endif
            bool result = buildMatchingForThreshold(r);
#ifdef VERBOSE_BOTTLENECK
            auto endMoment = hrClock.now();
            std::chrono::duration<double, std::milli> iterTime = endMoment - startMoment;
            std::cout << "isMatchLess for r = " << r << " finished in " << std::chrono::duration<double, std::milli>(iterTime).count() << " ms." << std::endl;
#endif
            return result;

        }


        template<class R, class NO>
        void BoundMatchOracle<R, NO>::removeFromLayer(const DgmPoint& p, const int layerIdx)
        {
            //bool isDebug {false};
            layerGraph[layerIdx].erase(p);
            if (layerOracles[layerIdx]) {
                layerOracles[layerIdx]->deletePoint(p);
            }
        }

        // return true, if there exists an augmenting path from startVertex
        // in this case the path is returned in result.
        // startVertex must be an exposed vertex from L_1 (layer[0])
        template<class R, class NO>
        bool BoundMatchOracle<R, NO>::buildAugmentingPath(const DgmPoint startVertex, Path& result)
        {
            //bool isDebug {false};
            DgmPoint prevVertexA = startVertex;
            result.clear();
            result.push_back(startVertex);
            size_t evenLayerIdx { 1 };
            while (evenLayerIdx < layerGraph.size()) {
                //for(size_t evenLayerIdx = 1; evenLayerIdx < layerGraph.size(); evenLayerIdx += 2) {
                DgmPoint nextVertexB; // next vertex from even layer
                bool neighbFound = layerOracles[evenLayerIdx]->getNeighbour(prevVertexA, nextVertexB);
                if (neighbFound) {
                    result.push_back(nextVertexB);
                    if (layerGraph.size() == evenLayerIdx + 1) {
                        break;
                    } else {
                        // nextVertexB must be matched with some vertex from the next odd
                        // layer
                        DgmPoint nextVertexA;
                        if (!M.getMatchedVertex(nextVertexB, nextVertexA)) {
#ifndef FOR_R_TDA
                            std::cerr << "Vertices in even layers must be matched! Unmatched: ";
                            std::cerr << nextVertexB << std::endl;
                            std::cerr << evenLayerIdx << "; " << layerGraph.size() << std::endl;
#endif
                            throw std::runtime_error("Unmatched vertex in even layer");
                        } else {
                            assert(!(nextVertexA.getRealX() == 0 and nextVertexA.getRealY() == 0));
                            result.push_back(nextVertexA);
                            prevVertexA = nextVertexA;
                            evenLayerIdx += 2;
                            continue;
                        }
                    }
                } else {
                    // prevVertexA has no neighbours in the next layer,
                    // backtrack
                    if (evenLayerIdx == 1) {
                        // startVertex is not connected to any vertices
                        // in the next layer, augm. path doesn't exist
                        removeFromLayer(startVertex, 0);
                        return false;
                    } else {
                        assert(evenLayerIdx >= 3);
                        assert(result.size() % 2 == 1);
                        result.pop_back();
                        DgmPoint prevVertexB = result.back();
                        result.pop_back();
                        removeFromLayer(prevVertexA, evenLayerIdx - 1);
                        removeFromLayer(prevVertexB, evenLayerIdx - 2);
                        // we should proceed from the previous odd layer
                        assert(result.size() >= 1);
                        prevVertexA = result.back();
                        evenLayerIdx -= 2;
                        continue;
                    }
                }
            } // finished iterating over all layers
            // remove all vertices in the augmenting paths
            // the corresponding layers
            for (size_t layerIdx = 0; layerIdx < result.size(); ++layerIdx) {
                removeFromLayer(result[layerIdx], layerIdx);
            }
            return true;
        }


        template<class R, class NO>
        bool BoundMatchOracle<R, NO>::buildMatchingForThreshold(const Real r)
        {
            //bool isDebug {false};
            if (prevQueryValue > r) {
                M.trimMatching(r);
            }
            prevQueryValue = r;
            while (true) {
                buildLayerGraph(r);
                if (augPathExist) {
                    std::vector<Path> augmentingPaths;
                    DgmPointSet copyLG0;
                    for (DgmPoint p : layerGraph[0]) {
                        copyLG0.insert(p);
                    }
                    for (DgmPoint exposedVertex : copyLG0) {
                        Path augPath;
                        if (buildAugmentingPath(exposedVertex, augPath)) {
                            augmentingPaths.push_back(augPath);
                        }
                    }
                    if (augmentingPaths.empty()) {
#ifndef FOR_R_TDA
                        std::cerr << "augmenting paths must exist, but were not found!" << std::endl;
#endif
                        throw std::runtime_error("bad epsilon?");
                    }
                    // swap all augmenting paths with matching to increase it
                    for (auto& augPath : augmentingPaths) {
                        M.increase(augPath);
                    }
                } else {
                    return M.isPerfect();
                }
            }
        }

        template<class R, class NO>
        void BoundMatchOracle<R, NO>::printLayerGraph(void)
        {
#ifdef DEBUG_BOUND_MATCH
            for(auto& layer : layerGraph) {
                std::cout << "{ ";
                for(auto& p : layer) {
                    std::cout << p << "; ";
                }
                std::cout << "\b\b }" << std::endl;
            }
#endif
        }

        template<class R, class NO>
        void BoundMatchOracle<R, NO>::buildLayerGraph(Real r)
        {
#ifdef VERBOSE_BOTTLENECK
            std::cout << "Entered buildLayerGraph, r = " << r << std::endl;
#endif
            layerGraph.clear();
            DgmPointSet L1 = M.getExposedVertices();
            layerGraph.push_back(L1);
            neighbOracle->rebuild(B, r);
            size_t k = 0;
            DgmPointSet layerNextEven;
            DgmPointSet layerNextOdd;
            bool exposedVerticesFound { false };
            while (true) {
                layerNextEven.clear();
                for (auto p : layerGraph[k]) {
                    bool neighbFound;
                    //DgmPoint neighbour {0.0, 0.0, DgmPoint::DIAG, 1};
                    DgmPoint neighbour;
                    if (useRangeSearch) {
                        std::vector<DgmPoint> neighbVec;
                        neighbOracle->getAllNeighbours(p, neighbVec);
                        neighbFound = !neighbVec.empty();
                        for (auto& neighbPt : neighbVec) {
                            layerNextEven.insert(neighbPt);
                            if (!exposedVerticesFound and M.isExposed(neighbPt)) {
                                exposedVerticesFound = true;
                            }
                        }
                    } else {
                        while (true) {
                            neighbFound = neighbOracle->getNeighbour(p, neighbour);
                            if (neighbFound) {
                                layerNextEven.insert(neighbour);
                                neighbOracle->deletePoint(neighbour);
                                if ((!exposedVerticesFound) && M.isExposed(neighbour)) {
                                    exposedVerticesFound = true;
                                }
                            } else {
                                break;
                            }
                        }
                    } // without range search
                } // all vertices from previous odd layer processed
                if (layerNextEven.empty()) {
                    augPathExist = false;
                    break;
                }
                if (exposedVerticesFound) {
                    for (auto it = layerNextEven.cbegin(); it != layerNextEven.cend();) {
                        if (!M.isExposed(*it)) {
                            layerNextEven.erase(it++);
                        } else {
                            ++it;
                        }

                    }
                    layerGraph.push_back(layerNextEven);
                    augPathExist = true;
                    break;
                }
                layerGraph.push_back(layerNextEven);
                M.getAllAdjacentVertices(layerNextEven, layerNextOdd);
                layerGraph.push_back(layerNextOdd);
                k += 2;
            }
            buildLayerOracles(r);
            printLayerGraph();
        }

        // create geometric oracles for each even layer
        // odd layers have NULL in layerOracles
        template<class R, class NO>
        void BoundMatchOracle<R, NO>::buildLayerOracles(Real r)
        {
            //bool isDebug {false};
            // free previously constructed oracles
            layerOracles.clear();
            for (size_t layerIdx = 0; layerIdx < layerGraph.size(); ++layerIdx) {
                if (layerIdx % 2 == 1) {
                    // even layer, build actual oracle
                    layerOracles.emplace_back(new NeighbOracle(layerGraph[layerIdx], r, distEpsilon));
                } else {
                    // odd layer
                    layerOracles.emplace_back(nullptr);
                }
            }
        }

    } // end namespace bt
} // end namespace hera
#endif // HERA_BOUND_MATCH_HPP
