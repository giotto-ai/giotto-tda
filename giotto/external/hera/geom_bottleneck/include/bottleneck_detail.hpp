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

#ifndef HERA_BOTTLENECK_HPP
#define HERA_BOTTLENECK_HPP

#ifdef FOR_R_TDA
#undef DEBUG_BOUND_MATCH
#undef DEBUG_MATCHING
#undef VERBOSE_BOTTLENECK
#endif


#include <iomanip>
#include <sstream>
#include <string>
#include <cctype>
#include <set>

#include "bottleneck_detail.h"

namespace hera {
    namespace bt {

        template<class Real>
        void binarySearch(const Real epsilon,
                          std::pair<Real, Real>& result,
                          BoundMatchOracle <Real>& oracle,
                          const Real infinityCost,
                          bool isResultInitializedCorrectly,
                          const Real distProbeInit)
        {
            // aliases for result components
            Real& distMin = result.first;
            Real& distMax = result.second;

            distMin = std::max(distMin, infinityCost);
            distMax = std::max(distMax, infinityCost);

            Real distProbe;

            if (not isResultInitializedCorrectly) {
                distProbe = distProbeInit;
                if (oracle.isMatchLess(distProbe)) {
                    // distProbe is an upper bound,
                    // find lower bound with binary search
                    do {
                        distMax = distProbe;
                        distProbe /= 2.0;
                    } while (oracle.isMatchLess(distProbe));
                    distMin = distProbe;
                } else {
                    // distProbe is a lower bound,
                    // find upper bound with exponential search
                    do {
                        distMin = distProbe;
                        distProbe *= 2.0;
                    } while (!oracle.isMatchLess(distProbe));
                    distMax = distProbe;
                }
            }
            // bounds are correct , perform binary search
            distProbe = (distMin + distMax) / 2.0;
            while ((distMax - distMin) / distMin >= epsilon) {

                if (distMax < infinityCost) {
                    distMin = infinityCost;
                    distMax = infinityCost;
                    break;
                }

                if (oracle.isMatchLess(distProbe)) {
                    distMax = distProbe;
                } else {
                    distMin = distProbe;
                }

                distProbe = (distMin + distMax) / 2.0;
            }

            distMin = std::max(distMin, infinityCost);
            distMax = std::max(distMax, infinityCost);
        }

        //        template<class Real>
        //        inline Real getOneDimensionalCost(std::vector<Real>& set_A, std::vector<Real>& set_B)
        //        {
        //            if (set_A.size() != set_B.size()) {
        //                return std::numeric_limits<Real>::infinity();
        //            }
        //
        //            if (set_A.empty()) {
        //                return Real(0.0);
        //            }
        //
        //            std::sort(set_A.begin(), set_A.end());
        //            std::sort(set_B.begin(), set_B.end());
        //
        //            Real result = 0.0;
        //            for (size_t i = 0; i < set_A.size(); ++i) {
        //                result = std::max(result, (std::fabs(set_A[i] - set_B[i])));
        //            }
        //
        //            return result;
        //        }


        template<class Real>
        struct CostEdgePair
        {
            Real cost;
            typename hera::bt::MatchingEdge<Real> edge;
        };

        template<class Real>
        using CoordPointPair = std::pair<Real, typename hera::bt::DiagramPoint<Real>>;

        template<class Real>
        using CoordPointVector = std::vector<typename hera::bt::CoordPointPair<Real>>;

        template<class Real>
        struct CoordPointPairComparator
        {
            bool operator()(const CoordPointPair<Real>& a, const CoordPointPair<Real>& b) const
            {
                return a.first < b.first or (a.first == b.first and a.second.id < b.second.id);
            };
        };

        template<class Real>
        inline typename hera::bt::CostEdgePair<Real>
        getOneDimensionalCost(typename hera::bt::CoordPointVector<Real>& set_A,
                              typename hera::bt::CoordPointVector<Real>& set_B)
        {
            using MatchingEdgeR = hera::bt::MatchingEdge<Real>;
            using CostEdgePairR = CostEdgePair<Real>;

            if (set_A.size() != set_B.size()) {
                return CostEdgePairR { std::numeric_limits<Real>::infinity(), MatchingEdgeR() };
            }

            if (set_A.empty()) {
                return CostEdgePairR { Real(0.0), MatchingEdgeR() };
            }

            std::sort(set_A.begin(), set_A.end(), CoordPointPairComparator<Real>());
            std::sort(set_B.begin(), set_B.end(), CoordPointPairComparator<Real>());

            CostEdgePairR result { -1.0, MatchingEdgeR() };

            for (size_t i = 0; i < set_A.size(); ++i) {
                Real curr_cost = std::fabs(set_A[i].first - set_B[i].first);
                if (curr_cost > result.cost) {
                    result.cost = curr_cost;
                    result.edge = MatchingEdgeR(set_A[i].second, set_B[i].second);
                }
            }
            return result;
        }


        template<class Real>
        inline CostEdgePair<Real> getInfinityCost(const DiagramPointSet <Real>& A, const DiagramPointSet <Real>& B,
                                                  bool compute_longest_edge = false)
        {
            using CostEdgePairR = CostEdgePair<Real>;
            using CoordPointVectorR = CoordPointVector<Real>;

            CoordPointVectorR x_plus_A, x_minus_A, y_plus_A, y_minus_A;
            CoordPointVectorR x_plus_B, x_minus_B, y_plus_B, y_minus_B;

            for (auto iter_A = A.cbegin(); iter_A != A.cend(); ++iter_A) {
                Real x = iter_A->getRealX();
                Real y = iter_A->getRealY();
                if (x == std::numeric_limits<Real>::infinity()) {
                    y_plus_A.emplace_back(y, *iter_A);
                } else if (x == -std::numeric_limits<Real>::infinity()) {
                    y_minus_A.emplace_back(y, *iter_A);
                } else if (y == std::numeric_limits<Real>::infinity()) {
                    x_plus_A.emplace_back(x, *iter_A);
                } else if (y == -std::numeric_limits<Real>::infinity()) {
                    x_minus_A.emplace_back(x, *iter_A);
                }
            }

            for (auto iter_B = B.cbegin(); iter_B != B.cend(); ++iter_B) {
                Real x = iter_B->getRealX();
                Real y = iter_B->getRealY();
                if (x == std::numeric_limits<Real>::infinity()) {
                    y_plus_B.emplace_back(y, *iter_B);
                } else if (x == -std::numeric_limits<Real>::infinity()) {
                    y_minus_B.emplace_back(y, *iter_B);
                } else if (y == std::numeric_limits<Real>::infinity()) {
                    x_plus_B.emplace_back(x, *iter_B);
                } else if (y == -std::numeric_limits<Real>::infinity()) {
                    x_minus_B.emplace_back(x, *iter_B);
                }
            }

            CostEdgePairR result = getOneDimensionalCost(x_plus_A, x_plus_B);

            CostEdgePairR next_cost_edge = getOneDimensionalCost(x_minus_A, x_minus_B);
            if (next_cost_edge.cost > result.cost) {
                result = next_cost_edge;
            }

            next_cost_edge = getOneDimensionalCost(y_plus_A, y_plus_B);
            if (next_cost_edge.cost > result.cost) {
                result = next_cost_edge;
            }

            next_cost_edge = getOneDimensionalCost(y_minus_A, y_minus_B);
            if (next_cost_edge.cost > result.cost) {
                result = next_cost_edge;
            }

            return result;
        }

        // return the interval (distMin, distMax) such that:
        // a) actual bottleneck distance between A and B is contained in the interval
        // b) if the interval is not (0,0), then  (distMax - distMin) / distMin < epsilon
        template<class Real>
        inline std::pair<Real, Real>
        bottleneckDistApproxInterval(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B, const Real epsilon,
                                     MatchingEdge<Real>& edge, bool compute_longest_edge)
        {
            using MatchingEdgeR = MatchingEdge<Real>;
            using CostEdgePairR = CostEdgePair<Real>;

            edge = MatchingEdgeR();
            // empty diagrams are not considered as error
            if (A.empty() and B.empty()) {
                return std::make_pair(0.0, 0.0);
            }

            CostEdgePairR inf_cost_edge = getInfinityCost(A, B, true);

            Real infinity_cost = inf_cost_edge.cost;
            if (infinity_cost == std::numeric_limits<Real>::infinity()) {
                return std::make_pair(infinity_cost, infinity_cost);
            } else {
                edge = inf_cost_edge.edge;
            }

            // link diagrams A and B by adding projections
            addProjections(A, B);

            // TODO: think about that!
            // we need one threshold for checking if the distance is 0,
            // another one for the oracle!
            constexpr Real epsThreshold { 1.0e-10 };
            std::pair<Real, Real> result { 0.0, 0.0 };
            bool useRangeSearch { true };
            // construct an oracle
            BoundMatchOracle<Real> oracle(A, B, epsThreshold, useRangeSearch);
            // check for distance = 0
            if (oracle.isMatchLess(2 * epsThreshold)) {
                if (infinity_cost > epsThreshold) {
                    result.first = infinity_cost;
                    result.second = infinity_cost;
                    edge = inf_cost_edge.edge;
                }
                return result;
            }
            // get a 3-approximation of maximal distance between A and B
            // as a starting value for probe distance
            Real distProbe { getFurthestDistance3Approx<Real, DiagramPointSet<Real>>(A, B) };
            binarySearch(epsilon, result, oracle, infinity_cost, false, distProbe);
            // to compute longest edge a perfect matching is needed
            if (compute_longest_edge and result.first > infinity_cost) {
                oracle.isMatchLess(result.second);
                edge = oracle.get_longest_edge();
            }
            return result;
        }

        template<class Real>
        void sampleDiagramForHeur(const DiagramPointSet <Real>& dgmIn, DiagramPointSet <Real>& dgmOut)
        {
            struct pair_hash
            {
                std::size_t operator()(const std::pair<Real, Real> p) const
                {
                    return std::hash<Real>()(p.first) ^ std::hash<Real>()(p.second);
                }
            };
            std::unordered_map<std::pair<Real, Real>, int, pair_hash> m;
            for (auto ptIter = dgmIn.cbegin(); ptIter != dgmIn.cend(); ++ptIter) {
                if (ptIter->isNormal() and not ptIter->isInfinity()) {
                    m[std::make_pair(ptIter->getRealX(), ptIter->getRealY())]++;
                }
            }
            if (m.size() < 2) {
                dgmOut = dgmIn;
                return;
            }
            std::vector<int> v;
            for (const auto& ptQtyPair : m) {
                v.push_back(ptQtyPair.second);
            }
            std::sort(v.begin(), v.end());
            int maxLeap = v[1] - v[0];
            int cutVal = v[0];
            for (int i = 1; i < static_cast<int>(v.size()) - 1; ++i) {
                int currLeap = v[i + 1] - v[i];
                if (currLeap > maxLeap) {
                    maxLeap = currLeap;
                    cutVal = v[i];
                }
            }
            std::vector<std::pair<Real, Real>> vv;
            // keep points whose multiplicites are at most cutVal
            // quick-and-dirty: fill in vv with copies of each point
            // to construct DiagramPointSet from it later
            for (const auto& ptQty : m) {
                if (ptQty.second < cutVal) {
                    for (int i = 0; i < ptQty.second; ++i) {
                        vv.push_back(std::make_pair(ptQty.first.first, ptQty.first.second));
                    }
                }
            }
            dgmOut.clear();
            dgmOut = DiagramPointSet<Real>(vv.begin(), vv.end());
        }


        // return the interval (distMin, distMax) such that:
        // a) actual bottleneck distance between A and B is contained in the interval
        // b) if the interval is not (0,0), then  (distMax - distMin) / distMin < epsilon
        template<class Real>
        std::pair<Real, Real>
        bottleneckDistApproxIntervalWithInitial(DiagramPointSet <Real>& A, DiagramPointSet <Real>& B,
                                                const Real epsilon,
                                                const std::pair<Real, Real> initialGuess,
                                                const Real infinity_cost,
                                                MatchingEdge <Real>& longest_edge,
                                                bool compute_longest_edge = false)
        {
            // empty diagrams are not considered as error
            if (A.empty() and B.empty()) {
                return std::make_pair(0.0, 0.0);
            }

            // link diagrams A and B by adding projections
            addProjections(A, B);

            constexpr Real epsThreshold { 1.0e-10 };
            std::pair<Real, Real> result { 0.0, 0.0 };
            bool useRangeSearch { true };
            // construct an oracle
            BoundMatchOracle<Real> oracle(A, B, epsThreshold, useRangeSearch);

            Real& distMin { result.first };
            Real& distMax { result.second };

            // initialize search interval from initialGuess
            distMin = initialGuess.first;
            distMax = initialGuess.second;

            assert(distMin <= distMax);

            // make sure that distMin is a lower bound
            while (oracle.isMatchLess(distMin)) {
                // distMin is in fact an upper bound, so assign it to distMax
                distMax = distMin;
                // and decrease distMin by 5 %
                distMin = 0.95 * distMin;
            }

            // make sure that distMax is an upper bound
            while (not oracle.isMatchLess(distMax)) {
                // distMax is in fact a lower bound, so assign it to distMin
                distMin = distMax;
                // and increase distMax by 5 %
                distMax = 1.05 * distMax;
            }

            // bounds are found, perform binary search
            Real distProbe = (distMin + distMax) / 2.0;
            binarySearch(epsilon, result, oracle, infinity_cost, true, distProbe);
            if (compute_longest_edge) {
                longest_edge = oracle.get_longest_edge();
            }
            return result;
        }

        // return the interval (distMin, distMax) such that:
        // a) actual bottleneck distance between A and B is contained in the interval
        // b) if the interval is not (0,0), then  (distMax - distMin) / distMin < epsilon
        // use heuristic: initial estimate on sampled diagrams
        template<class Real>
        std::pair<Real, Real>
        bottleneckDistApproxIntervalHeur(DiagramPointSet <Real>& A, DiagramPointSet <Real>& B, const Real epsilon,
                                         MatchingEdge <Real>& longest_edge)
        {
            // empty diagrams are not considered as error
            if (A.empty() and B.empty()) {
                return std::make_pair(0.0, 0.0);
            }

            Real infinity_cost = getInfinityCost(A, B);
            if (infinity_cost == std::numeric_limits<Real>::infinity()) {
                return std::make_pair(infinity_cost, infinity_cost);
            }

            DiagramPointSet<Real> sampledA, sampledB;
            sampleDiagramForHeur(A, sampledA);
            sampleDiagramForHeur(B, sampledB);

            std::pair<Real, Real> initGuess = bottleneckDistApproxInterval(sampledA, sampledB, epsilon);

            initGuess.first = std::max(initGuess.first, infinity_cost);
            initGuess.second = std::max(initGuess.second, infinity_cost);

            return bottleneckDistApproxIntervalWithInitial<Real>(A, B, epsilon, initGuess, infinity_cost, longest_edge);
        }


        // get approximate distance,
        // see bottleneckDistApproxInterval
        template<class Real>
        Real bottleneckDistApprox(DiagramPointSet <Real>& A, DiagramPointSet <Real>& B, const Real epsilon,
                                  MatchingEdge <Real>& longest_edge, bool compute_longest_edge)
        {
            auto interval = bottleneckDistApproxInterval<Real>(A, B, epsilon, longest_edge, compute_longest_edge);
            return interval.second;
        }


        template<class Real>
        Real bottleneckDistExactFromSortedPwDist(DiagramPointSet <Real>& A, DiagramPointSet <Real>& B,
                                                 const std::vector<Real>& pairwiseDist,
                                                 const int decPrecision, MatchingEdge <Real>& longest_edge,
                                                 bool compute_longest_edge = false)
        {
            // trivial case: we have only one candidate
            if (pairwiseDist.size() == 1) {
                return pairwiseDist[0];
            }

            bool useRangeSearch = true;
            Real distEpsilon = std::numeric_limits<Real>::max();
            Real diffThreshold = 0.1;
            for (int k = 0; k < decPrecision; ++k) {
                diffThreshold /= 10;
            }
            for (size_t k = 0; k < pairwiseDist.size() - 2; ++k) {
                auto diff = pairwiseDist[k + 1] - pairwiseDist[k];
                if (diff > diffThreshold and diff < distEpsilon) {
                    distEpsilon = diff;
                }
            }
            distEpsilon = std::min(diffThreshold, distEpsilon / 3);

            BoundMatchOracle<Real> oracle(A, B, distEpsilon, useRangeSearch);
            // binary search
            size_t iterNum { 0 };
            size_t idxMin { 0 }, idxMax { pairwiseDist.size() - 1 };
            size_t idxMid;
            while (idxMax > idxMin) {
                idxMid = static_cast<size_t>(floor(idxMin + idxMax) / 2);
                iterNum++;
                // not A[imid] < dist <=>  A[imid] >= dist  <=> A[imid[ >= dist + eps
                if (oracle.isMatchLess(pairwiseDist[idxMid] + distEpsilon / 2)) {
                    idxMax = idxMid;
                } else {
                    idxMin = idxMid + 1;
                }
            }
            idxMid = static_cast<size_t>(floor(idxMin + idxMax) / 2);
            Real result = pairwiseDist[idxMid];
            if (compute_longest_edge) {
                oracle.isMatchLess(result + distEpsilon / 2);
                longest_edge = oracle.get_longest_edge();
            }
            return result;
        }


        template<class Real>
        Real
        bottleneckDistExact(DiagramPointSet <Real>& A, DiagramPointSet <Real>& B, MatchingEdge <Real>& longest_edge,
                            bool compute_longest_edge)
        {
            return bottleneckDistExact(A, B, 14, longest_edge, compute_longest_edge);
        }

        template<class Real>
        Real bottleneckDistExact(DiagramPointSet <Real>& A, DiagramPointSet <Real>& B, const int decPrecision,
                                 MatchingEdge <Real>& longest_edge, bool compute_longest_edge)
        {
            using DgmPoint = DiagramPoint<Real>;

            constexpr Real epsilon = 0.001;
            auto interval = bottleneckDistApproxInterval(A, B, epsilon, longest_edge, true);
            // if the longest edge is on infinity, the answer is already exact
            // this will be detected here and all the code after if
            // may assume that the longest edge is on finite points
            if (interval.first == interval.second) {
                return interval.first;
            }
            const Real delta = 0.50001 * (interval.second - interval.first);
            const Real approxDist = 0.5 * (interval.first + interval.second);
            const Real minDist = interval.first;
            const Real maxDist = interval.second;
            if (delta == 0) {
                return interval.first;
            }
            // copy points from A to a vector
            // todo: get rid of this?
            std::vector<DgmPoint> pointsA;
            pointsA.reserve(A.size());
            for (const auto& ptA : A) {
                pointsA.push_back(ptA);
            }

            // in this vector we store the distances between the points
            // that are candidates to realize
            std::set<Real> pairwiseDist;
            {
                // vector to store centers of vertical stripes
                // two for each point in A and the id of the corresponding point
                std::vector<std::pair<Real, DgmPoint>> xCentersVec;
                xCentersVec.reserve(2 * pointsA.size());
                for (auto ptA : pointsA) {
                    xCentersVec.push_back(std::make_pair(ptA.getRealX() - approxDist, ptA));
                    xCentersVec.push_back(std::make_pair(ptA.getRealX() + approxDist, ptA));
                }
                // lambda to compare pairs <coordinate, id> w.r.t coordinate
                auto compLambda = [](std::pair<Real, DgmPoint> a, std::pair<Real, DgmPoint> b) {
                    return a.first < b.first;
                };

                std::sort(xCentersVec.begin(), xCentersVec.end(), compLambda);
                // todo: sort points in B, reduce search range in lower and upper bounds
                for (auto ptB : B) {
                    // iterator to the first stripe such that ptB lies to the left
                    // from its right boundary (x_B <= x_j + \delta iff x_j >= x_B - \delta
                    auto itStart = std::lower_bound(xCentersVec.begin(),
                                                    xCentersVec.end(),
                                                    std::make_pair(ptB.getRealX() - delta, ptB),
                                                    compLambda);

                    for (auto iterA = itStart; iterA < xCentersVec.end(); ++iterA) {
                        if (ptB.getRealX() < iterA->first - delta) {
                            // from that moment x_B >= x_j - delta
                            // is violated: x_B no longer lies to right from the left
                            // boundary of current stripe
                            break;
                        }
                        // we're here => ptB lies in vertical stripe,
                        // check if distance fits into the interval we've found
                        Real pwDist = distLInf(iterA->second, ptB);
                        if (pwDist >= minDist and pwDist <= maxDist) {
                            pairwiseDist.insert(pwDist);
                        }
                    }
                }
            }

            {
                // for y
                // vector to store centers of vertical stripes
                // two for each point in A and the id of the corresponding point
                std::vector<std::pair<Real, DgmPoint>> yCentersVec;
                yCentersVec.reserve(2 * pointsA.size());
                for (auto ptA : pointsA) {
                    yCentersVec.push_back(std::make_pair(ptA.getRealY() - approxDist, ptA));
                    yCentersVec.push_back(std::make_pair(ptA.getRealY() + approxDist, ptA));
                }
                // lambda to compare pairs <coordinate, id> w.r.t coordinate
                auto compLambda = [](std::pair<Real, DgmPoint> a, std::pair<Real, DgmPoint> b) {
                    return a.first < b.first;
                };

                std::sort(yCentersVec.begin(), yCentersVec.end(), compLambda);

                // todo: sort points in B, reduce search range in lower and upper bounds
                for (auto ptB : B) {
                    auto itStart = std::lower_bound(yCentersVec.begin(),
                                                    yCentersVec.end(),
                                                    std::make_pair(ptB.getRealY() - delta, ptB),
                                                    compLambda);


                    for (auto iterA = itStart; iterA < yCentersVec.end(); ++iterA) {
                        if (ptB.getRealY() < iterA->first - delta) {
                            break;
                        }
                        Real pwDist = distLInf(iterA->second, ptB);
                        if (pwDist >= minDist and pwDist <= maxDist) {
                            pairwiseDist.insert(pwDist);
                        }
                    }
                }
            }

            std::vector<Real> pw_dists;
            pw_dists.reserve(pairwiseDist.size());
            for(Real d : pairwiseDist) {
                pw_dists.push_back(d);
            }

            return bottleneckDistExactFromSortedPwDist(A, B, pw_dists, decPrecision, longest_edge,
                                                       compute_longest_edge);
        }

    } // end namespace bt
} // end namespace hera
#endif // HERA_BOTTLENECK_HPP
