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

#ifndef HERA_BASIC_DEFS_BT_H
#define HERA_BASIC_DEFS_BT_H

#ifdef _WIN32
#include <ciso646>
#endif

#include <utility>
#include <vector>
#include <stdexcept>
#include <math.h>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <assert.h>

#include "def_debug_bt.h"

#ifndef FOR_R_TDA

#include <iostream>

#endif

namespace hera {

    template<class Real = double>
    Real get_infinity()
    {
        return Real(-1.0);
    }

    namespace bt {


        typedef int IdType;
        constexpr IdType MinValidId = 10;

        template<class Real = double>
        struct Point
        {
            Real x, y;

            bool operator==(const Point<Real>& other) const
            {
                return ((this->x == other.x) and (this->y == other.y));
            }

            bool operator!=(const Point<Real>& other) const
            {
                return !(*this == other);
            }

            Point(Real ax, Real ay) :
                    x(ax), y(ay)
            {}

            Point() :
                    x(0.0), y(0.0)
            {}

#ifndef FOR_R_TDA

            template<class R>
            friend std::ostream& operator<<(std::ostream& output, const Point<R>& p)
            {
                output << "(" << p.x << ", " << p.y << ")";
                return output;
            }

#endif
        };

        template<class Real = double>
        struct DiagramPoint
        {
            // Points above the diagonal have type NORMAL
            // Projections onto the diagonal have type DIAG
            // for DIAG points only x-coordinate is relevant
            // to-do: add getters/setters, checks in constructors, etc
            enum Type
            {
                NORMAL, DIAG
            };
            // data members
        private:
            Real x, y;
        public:
            Type type;
            IdType id;
            IdType user_id;

            // operators, constructors
            bool operator==(const DiagramPoint<Real>& other) const
            {
                // compare by id only
                assert(this->id >= MinValidId);
                assert(other.id >= MinValidId);
                bool areEqual { this->id == other.id };
                assert(!areEqual or ((this->x == other.x) and (this->y == other.y) and (this->type == other.type)));
                return areEqual;
            }

            bool operator!=(const DiagramPoint& other) const
            {
                return !(*this == other);
            }

            DiagramPoint() :
                    x(0.0),
                    y(0.0),
                    type(DiagramPoint::DIAG),
                    id(MinValidId - 1),
                    user_id(-1)
            {
            }

            DiagramPoint(Real _x, Real _y, Type _type, IdType _id, IdType _user_id) :
                    x(_x),
                    y(_y),
                    type(_type),
                    id(_id),
                    user_id(_user_id)
            {
                if (_y == _x and _type != DIAG) {
                    throw std::runtime_error("Point on the main diagonal must have DIAG type");
                }

            }


            bool isDiagonal() const
            { return type == DIAG; }

            bool isNormal() const
            { return type == NORMAL; }

            bool isInfinity() const
            {
                return x == std::numeric_limits<Real>::infinity() or
                       x == -std::numeric_limits<Real>::infinity() or
                       y == std::numeric_limits<Real>::infinity() or
                       y == -std::numeric_limits<Real>::infinity();
            }

            Real inline getRealX() const // return the x-coord
            {
                return x;
            }

            Real inline getRealY() const // return the y-coord
            {
                return y;
            }

            IdType inline get_user_id() const
            {
                if (isNormal())
                    return user_id;
                else
                    return -1;
            }

            Real inline get_persistence(const Real internal_p = get_infinity()) const
            {
                if (isDiagonal())
                    return 0.0;
                Real pers = fabs(y - x) / 2;
                if (internal_p == get_infinity()) {
                    return pers;
                } else if (internal_p == 1.0) {
                    return 2 * pers;
                } else {
                    return std::pow(static_cast<Real>(2), static_cast<Real>(1) / internal_p);
                }
            }

#ifndef FOR_R_TDA

            friend std::ostream& operator<<(std::ostream& output, const DiagramPoint& p)
            {
                if (p.isDiagonal()) {
                    output << "(" << p.x << ", " << p.y << ", " << 0.5 * (p.x + p.y) << ", " << p.id << " DIAG )";
                } else {
                    output << "(" << p.x << ", " << p.y << ", " << p.id << " NORMAL)";
                }
                return output;
            }
#endif
        };

        template<class Real>
        using MatchingEdge = std::pair<DiagramPoint<Real>, DiagramPoint<Real>>;

        // compute l-inf distance between two diagram points
        template<class Real>
        inline Real distLInf(const DiagramPoint<Real>& a, const DiagramPoint<Real>& b)
        {
            if (a.isDiagonal() and b.isDiagonal()) {
                // distance between points on the diagonal is 0
                return 0.0;
            }
            // otherwise distance is a usual l-inf distance
            return std::max(fabs(a.getRealX() - b.getRealX()), fabs(a.getRealY() - b.getRealY()));
        }

        // this function works with points at infinity as well
        // not needed in actual computation, since these points are processed
        // separately, but is useful in tests
        template<class Real>
        inline Real dist_l_inf_slow(const DiagramPoint<Real>& a, const DiagramPoint<Real>& b)
        {
            if (a.isDiagonal() and b.isDiagonal()) {
                // distance between points on the diagonal is 0
                return 0.0;
            }
            // otherwise distance is a usual l-inf distance
            //
            Real dx = (a.getRealX() == b.getRealX()) ? 0.0 : fabs(a.getRealX() - b.getRealX());
            Real dy = (a.getRealY() == b.getRealY()) ? 0.0 : fabs(a.getRealY() - b.getRealY());
            Real result = std::max(dx, dy);
            if (std::isnan(result))
                result = std::numeric_limits<Real>::infinity();
            return result;
        }



        template<class Real = double>
        inline Real get_infinity()
        {
            return Real(-1.0);
        }

        template<class T>
        inline void hash_combine(std::size_t& seed, const T& v)
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        template<class Real = double>
        struct DiagramPointHash
        {
            size_t operator()(const DiagramPoint<Real>& p) const
            {
                assert(p.id >= MinValidId);
                return std::hash<int>()(p.id);
            }
        };

        template<class Real = double>
        Real distLInf(const DiagramPoint<Real>& a, const DiagramPoint<Real>& b);

        //template<class Real = double>
        //typedef std::unordered_set<Point, PointHash> PointSet;
        template<class Real_ = double>
        class DiagramPointSet;

        template<class Real>
        void addProjections(DiagramPointSet<Real>& A, DiagramPointSet<Real>& B);

        template<class Real_>
        class DiagramPointSet
        {
        public:

            using Real = Real_;
            using DgmPoint = DiagramPoint<Real>;
            using DgmPointHash = DiagramPointHash<Real>;
            using const_iterator = typename std::unordered_set<DgmPoint, DgmPointHash>::const_iterator;
            using iterator = typename std::unordered_set<DgmPoint, DgmPointHash>::iterator;

        private:

            bool isLinked { false };
            IdType maxId { MinValidId + 1 };
            std::unordered_set<DgmPoint, DgmPointHash> points;

        public:

            void insert(const DgmPoint& p)
            {
                points.insert(p);
                if (p.id > maxId) {
                    maxId = p.id + 1;
                }
            }

            void erase(const DgmPoint& p, bool doCheck = true)
            {
                // if doCheck, erasing non-existing elements causes assert
                auto it = points.find(p);
                if (it != points.end()) {
                    points.erase(it);
                } else {
                    assert(!doCheck);
                }
            }


            void erase(const const_iterator it)
            {
                points.erase(it);
            }

            void removeDiagonalPoints()
            {
                if (isLinked) {
                    auto ptIter = points.begin();
                    while (ptIter != points.end()) {
                        if (ptIter->isDiagonal()) {
                            ptIter = points.erase(ptIter);
                        } else {
                            ptIter++;
                        }
                    }
                    isLinked = false;
                }
            }

            size_t size() const
            {
                return points.size();
            }

            void reserve(const size_t newSize)
            {
                points.reserve(newSize);
            }

            void clear()
            {
                points.clear();
            }

            bool empty() const
            {
                return points.empty();
            }

            bool hasElement(const DgmPoint& p) const
            {
                return points.find(p) != points.end();
            }

            iterator find(const DgmPoint& p)
            {
                return points.find(p);
            }

            iterator begin()
            {
                return points.begin();
            }

            iterator end()
            {
                return points.end();
            }

            const_iterator cbegin() const
            {
                return points.cbegin();
            }

            const_iterator cend() const
            {
                return points.cend();
            }


            const_iterator find(const DgmPoint& p) const
            {
                return points.find(p);
            }

#ifndef FOR_R_TDA

            friend std::ostream& operator<<(std::ostream& output, const DiagramPointSet& ps)
            {
                output << "{ ";
                for (auto pit = ps.cbegin(); pit != ps.cend(); ++pit) {
                    output << *pit << ", ";
                }
                output << "\b\b }";
                return output;
            }

#endif

            friend void addProjections<Real_>(DiagramPointSet<Real_>& A, DiagramPointSet<Real_>& B);

            template<class PairIterator>
            void fillIn(PairIterator begin_iter, PairIterator end_iter)
            {
                isLinked = false;
                clear();
                IdType uniqueId = MinValidId + 1;
                IdType user_id = 0;
                for (auto iter = begin_iter; iter != end_iter; ++iter) {
                    insert(DgmPoint(iter->first, iter->second, DgmPoint::NORMAL, uniqueId++, user_id++));
                }
            }

            template<class PointContainer>
            void fillIn(const PointContainer& dgm_cont)
            {
                using Traits = DiagramTraits<PointContainer>;
                isLinked = false;
                clear();
                IdType uniqueId = MinValidId + 1;
                IdType user_id = 0;
                for (const auto& pt : dgm_cont) {
                    Real x = Traits::get_x(pt);
                    Real y = Traits::get_y(pt);
                    insert(DgmPoint(x, y, DgmPoint::NORMAL, uniqueId++, user_id++));
                }
            }


            // ctor from range
            template<class PairIterator>
            DiagramPointSet(PairIterator begin_iter, PairIterator end_iter)
            {
                fillIn(begin_iter, end_iter);
            }

            // ctor from container, uses DiagramTraits
            template<class PointContainer>
            DiagramPointSet(const PointContainer& dgm)
            {
                fillIn(dgm);
            }


            // default ctor, empty diagram
            DiagramPointSet(IdType minId = MinValidId + 1) :
                    maxId(minId + 1)
            {};

            IdType nextId()
            { return maxId + 1; }

        }; // DiagramPointSet


        template<class Real, class DiagPointContainer>
        Real getFurthestDistance3Approx(DiagPointContainer& A, DiagPointContainer& B)
        {
            Real result { 0.0 };
            DiagramPoint<Real> begA = *(A.begin());
            DiagramPoint<Real> optB = *(B.begin());
            for (const auto& pointB : B) {
                if (distLInf(begA, pointB) > result) {
                    result = distLInf(begA, pointB);
                    optB = pointB;
                }
            }
            for (const auto& pointA : A) {
                if (distLInf(pointA, optB) > result) {
                    result = distLInf(pointA, optB);
                }
            }
            return result;
        }

        // preprocess diagrams A and B by adding projections onto diagonal of points of
        // A to B and vice versa. Also removes points at infinity!
        // NB: ids of points will be changed!
        template<class Real_>
        void addProjections(DiagramPointSet<Real_>& A, DiagramPointSet<Real_>& B)
        {

            using Real = Real_;
            using DgmPoint = DiagramPoint<Real>;
            using DgmPointSet = DiagramPointSet<Real>;

            IdType uniqueId { MinValidId + 1 };
            DgmPointSet newA, newB;

            // copy normal points from A to newA
            // add projections to newB
            for (auto& pA : A) {
                if (pA.isNormal() and not pA.isInfinity()) {
                    // add pA's projection to B
                    DgmPoint dpA { pA.getRealX(), pA.getRealY(), DgmPoint::NORMAL, uniqueId++, pA.get_user_id() };
                    DgmPoint dpB { (pA.getRealX() + pA.getRealY()) / 2, (pA.getRealX() + pA.getRealY()) / 2,
                                   DgmPoint::DIAG, uniqueId++, -1 };
                    newA.insert(dpA);
                    newB.insert(dpB);
                }
            }

            for (auto& pB : B) {
                if (pB.isNormal() and not pB.isInfinity()) {
                    // add pB's projection to A
                    DgmPoint dpB { pB.getRealX(), pB.getRealY(), DgmPoint::NORMAL, uniqueId++, pB.get_user_id() };
                    DgmPoint dpA { (pB.getRealX() + pB.getRealY()) / 2, (pB.getRealX() + pB.getRealY()) / 2,
                                   DgmPoint::DIAG, uniqueId++, -1 };
                    newB.insert(dpB);
                    newA.insert(dpA);
                }
            }

            A = newA;
            B = newB;
            A.isLinked = true;
            B.isLinked = true;
        }

        //#ifndef FOR_R_TDA

        //template<class Real>
        //std::ostream& operator<<(std::ostream& output, const DiagramPoint<Real>& p)
        //{
        //    if ( p.isDiagonal() ) {
        //        output << "(" << p.x << ", " << p.y << ", " <<  0.5 * (p.x + p.y) << ", "  << p.id << " DIAG )";
        //    } else {
        //        output << "(" << p.x << ", " << p.y << ", " << p.id << " NORMAL)";
        //    }
        //    return output;
        //}

        //template<class Real>
        //std::ostream& operator<<(std::ostream& output, const DiagramPointSet<Real>& ps)
        //{
        //    output << "{ ";
        //    for(auto pit = ps.cbegin(); pit != ps.cend(); ++pit) {
        //        output << *pit << ", ";
        //    }
        //    output << "\b\b }";
        //    return output;
        //}
        //#endif // FOR_R_TDA


    } // end namespace bt
} // end namespace hera
#endif  // HERA_BASIC_DEFS_BT_H
