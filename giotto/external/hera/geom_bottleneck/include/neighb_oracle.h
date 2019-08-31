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

#ifndef HERA_NEIGHB_ORACLE_H
#define HERA_NEIGHB_ORACLE_H

#include <unordered_map>
#include <algorithm>
#include <memory>

#include "basic_defs_bt.h"
#include "dnn/geometry/euclidean-fixed.h"
#include "dnn/local/kd-tree.h"



namespace hera {
namespace bt {

template<class Real>
class NeighbOracleSimple
{
public:
    using DgmPoint = DiagramPoint<Real>;
    using DgmPointSet = DiagramPointSet<Real>;

private:
    Real r;
    Real distEpsilon;
    DgmPointSet pointSet;

public:

    NeighbOracleSimple() : r(0.0) {}

    NeighbOracleSimple(const DgmPointSet& _pointSet, const Real _r, const Real _distEpsilon) :
        r(_r),
        distEpsilon(_distEpsilon),
        pointSet(_pointSet)
    {}

    void deletePoint(const DgmPoint& p)
    {
        pointSet.erase(p);
    }

    void rebuild(const DgmPointSet& S, const double rr)
    {
        pointSet = S;
        r = rr;
    }

    bool getNeighbour(const DgmPoint& q, DgmPoint& result) const
    {
        for(auto pit = pointSet.cbegin(); pit != pointSet.cend(); ++pit) {
            if ( distLInf(*pit, q) <= r) {
                result = *pit;
                return true;
            }
        }
        return false;
    }

    void getAllNeighbours(const DgmPoint& q, std::vector<DgmPoint>& result)
    {
        result.clear();
        for(const auto& point : pointSet) {
            if ( distLInf(point, q) <= r) {
                result.push_back(point);
            }
        }
        for(auto& pt : result) {
            deletePoint(pt);
        }
    }

};

template<class Real_>
class NeighbOracleDnn
{
public:

    using Real = Real_;
    using DnnPoint = dnn::Point<2, double>;
    using DnnTraits = dnn::PointTraits<DnnPoint>;
    using DgmPoint = DiagramPoint<Real>;
    using DgmPointSet = DiagramPointSet<Real>;
    using DgmPointHash = DiagramPointHash<Real>;

    Real r;
    Real distEpsilon;
    std::vector<DgmPoint> allPoints;
    DgmPointSet diagonalPoints;
    std::unordered_map<DgmPoint, size_t, DgmPointHash> pointIdxLookup;
    // dnn-stuff
    std::unique_ptr<dnn::KDTree<DnnTraits>> kdtree;
    std::vector<DnnPoint> dnnPoints;
    std::vector<DnnPoint*> dnnPointHandles;
    std::vector<size_t> kdtreeItems;

    NeighbOracleDnn(const DgmPointSet& S, const Real rr, const Real dEps) :
        kdtree(nullptr)
    {
        assert(dEps >= 0);
        distEpsilon = dEps;
        rebuild(S, rr);
    }


    void deletePoint(const DgmPoint& p)
    {
        auto findRes = pointIdxLookup.find(p);
        assert(findRes != pointIdxLookup.end());
        //std::cout <<  "Deleting point " <<  p << std::endl;
        size_t pointIdx { (*findRes).second };
        //std::cout <<  "pointIdx =  " << pointIdx << std::endl;
        diagonalPoints.erase(p, false);
        kdtree->delete_point(dnnPointHandles[kdtreeItems[pointIdx]]);
    }

    void rebuild(const DgmPointSet& S, const Real rr)
    {
        //std::cout <<  "Entered rebuild, r = " <<  rr << std::endl;
        r = rr;
        size_t dnnNumPoints = S.size();
        //printDebug(isDebug, "S = ", S);
        if (dnnNumPoints  > 0) {
            pointIdxLookup.clear();
            pointIdxLookup.reserve(S.size());
            allPoints.clear();
            allPoints.reserve(S.size());
            diagonalPoints.clear();
            diagonalPoints.reserve(S.size() / 2);
            for(auto pit = S.cbegin(); pit != S.cend(); ++pit) {
                allPoints.push_back(*pit);
                if (pit->isDiagonal()) {
                    diagonalPoints.insert(*pit);
                }
            }

            size_t pointIdx = 0;
            for(auto& dataPoint : allPoints) {
                pointIdxLookup.insert( { dataPoint, pointIdx++ } );
            }

            size_t dnnItemIdx { 0 };
            size_t trueIdx { 0 };
            dnnPoints.clear();
            kdtreeItems.clear();
            dnnPointHandles.clear();
            dnnPoints.clear();
            kdtreeItems.reserve(S.size() );
            // store normal items in kd-tree
            for(const auto& g : allPoints) {
                if (true) {
                    kdtreeItems.push_back(dnnItemIdx);
                    // index of items is id of dnn-point
                    DnnPoint p(trueIdx);
                    p[0] = g.getRealX();
                    p[1] = g.getRealY();
                    dnnPoints.push_back(p);
                    assert(dnnItemIdx == dnnPoints.size() - 1);
                    dnnItemIdx++;
                }
                trueIdx++;
            }
            assert(dnnPoints.size() == allPoints.size() );
            for(size_t i = 0; i < dnnPoints.size(); ++i) {
                dnnPointHandles.push_back(&dnnPoints[i]);
            }
            DnnTraits traits;
            //std::cout << "kdtree: " << dnnPointHandles.size() << " points" << std::endl;
            kdtree.reset(new dnn::KDTree<DnnTraits>(traits, dnnPointHandles));
        }
    }


    bool getNeighbour(const DgmPoint& q, DgmPoint& result) const
    {
        //std::cout << "getNeighbour for q = " << q << ", r = " << r << std::endl;
        //std::cout << *this << std::endl;
        // distance between two diagonal points
        // is  0
        if (q.isDiagonal()) {
            if (!diagonalPoints.empty()) {
                result = *diagonalPoints.cbegin();
                //std::cout <<  "Neighbour found in diagonal points, res =  " <<  result;
                return true;
            }
        }
        // check if kdtree is not empty
        if (0 == kdtree->get_num_points() ) {
            //std::cout << "empty tree, no neighb." << std::endl;
            return false;
        }
        // if no neighbour found among diagonal points,
        // search in kd_tree
        DnnPoint queryPoint;
        queryPoint[0] = q.getRealX();
        queryPoint[1] = q.getRealY();
        auto kdtreeResult = kdtree->findFirstR(queryPoint, r);
        if (kdtreeResult.empty()) {
            //std::cout << "no neighbour within " << r << "found." << std::endl;
            return false;
        }
        if (kdtreeResult[0].d <= r + distEpsilon) {
            result = allPoints[kdtreeResult[0].p->id()];
            //std::cout << "Neighbour found with kd-tree, index =  " << kdtreeResult[0].p->id() << std::endl;
            //std::cout << "result =  " <<  result << std::endl;
            return true;
        }
        //std::cout << "No neighbour found for r =  " << r << std::endl;
        return false;
    }



    void getAllNeighbours(const DgmPoint& q, std::vector<DgmPoint>& result)
    {
        //std::cout <<  "Entered getAllNeighbours for q = " << q << std::endl;
        result.clear();
        // add diagonal points, if necessary
        if (  q.isDiagonal() ) {
            for( auto& diagPt : diagonalPoints ) {
                result.push_back(diagPt);
            }
        }
        // delete diagonal points we found
        // to prevent finding them again
        for(auto& pt : result) {
            //std::cout << "deleting DIAG point pt = " << pt << std::endl;
            deletePoint(pt);
        }
        size_t diagOffset = result.size();
        std::vector<size_t> pointIndicesOut;
        // perorm range search on kd-tree
        DnnPoint queryPoint;
        queryPoint[0] = q.getRealX();
        queryPoint[1] = q.getRealY();
        auto kdtreeResult = kdtree->findR(queryPoint, r);
        pointIndicesOut.reserve(kdtreeResult.size());
        for(auto& handleDist : kdtreeResult) {
            if (handleDist.d <= r + distEpsilon) {
                pointIndicesOut.push_back(handleDist.p->id());
            } else {
                break;
            }
        }
        // get actual points in result
        for(auto& ptIdx : pointIndicesOut) {
            result.push_back(allPoints[ptIdx]);
        }
        // delete all points we found
        for(auto ptIt = result.begin() + diagOffset; ptIt != result.end(); ++ptIt) {
            //printDebug(isDebug, "deleting point pt = ", *ptIt);
            deletePoint(*ptIt);
        }
    }

    //DgmPointSet originalPointSet;
    template<class R>
    friend std::ostream& operator<<(std::ostream& out, const NeighbOracleDnn<R>& oracle);

};

} // end namespace bt
} // end namespace hera

#endif // HERA_NEIGHB_ORACLE_H
