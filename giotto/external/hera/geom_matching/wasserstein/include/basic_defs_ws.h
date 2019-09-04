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

#ifndef BASIC_DEFS_WS_H
#define BASIC_DEFS_WS_H

#include <vector>
#include <math.h>
#include <cstddef>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <iomanip>
#include <locale>
#include <cassert>
#include <limits>
#include <ostream>
#include <typeinfo>

#ifdef _WIN32
#include <ciso646>
#endif

#ifndef FOR_R_TDA
#include "spdlog/spdlog.h"
#include "spdlog/fmt/fmt.h"
#include "spdlog/fmt/ostr.h"
#endif

#include "hera_infinity.h"
#include "dnn/geometry/euclidean-dynamic.h"
#include "def_debug_ws.h"

#define MIN_VALID_ID 10

namespace hera
{

//template<class Real = double>
//inline bool is_infinity(const Real& x)
//{
//    return x == Real(-1);
//};
//
//template<class Real = double>
//inline Real get_infinity()
//{
//    return Real( -1 );
//}

template<class Real = double>
inline bool is_p_valid_norm(const Real& p)
{
    return is_infinity<Real>(p) or p >= Real(1);
}

template<class Real = double>
struct AuctionParams
{
    Real wasserstein_power { 1.0 };
    Real delta { 0.01 }; // relative error
    Real internal_p { get_infinity<Real>() };
    Real initial_epsilon { 0.0 }; // 0.0 means maxVal / 4.0
    Real epsilon_common_ratio { 5.0 };
    Real gamma_threshold { 0.0 };  // for experiments, not in use now
    int max_num_phases { std::numeric_limits<decltype(max_num_phases)>::max() };
    int max_bids_per_round { 1 };  // imitate Gauss-Seidel is default behaviour
    unsigned int dim { 2 }; // for pure geometric version only; ignored in persistence diagrams
    Real final_relative_error;  // out parameter - after auction terminates, contains the real relative error
    bool tolerate_max_iter_exceeded { false }; // whether auction should throw an exception on max. iterations exceeded
};

namespace ws
{

    using IdxType = int;

    constexpr size_t k_invalid_index = std::numeric_limits<IdxType>::max();

    template<class Real = double>
    using IdxValPair = std::pair<IdxType, Real>;

    template<class R>
    inline std::ostream& operator<<(std::ostream& output, const IdxValPair<R> p)
    {
        output << fmt::format("({0}, {1})", p.first, p.second);
        return output;
    }

    enum class OwnerType { k_none, k_normal, k_diagonal };

    inline std::ostream& operator<<(std::ostream& s, const OwnerType t)
    {
        switch(t)
        {
            case OwnerType::k_none : s << "NONE"; break;
            case OwnerType::k_normal: s << "NORMAL"; break;
            case OwnerType::k_diagonal: s << "DIAGONAL"; break;
        }
        return s;
    }

    template<class Real = double>
    struct Point {
        Real x, y;
        bool operator==(const Point& other) const;
        bool operator!=(const Point& other) const;
        Point(Real _x, Real _y) : x(_x), y(_y) {}
        Point() : x(0.0), y(0.0) {}
    };

#ifndef FOR_R_TDA
    template<class Real = double>
    std::ostream& operator<<(std::ostream& output, const Point<Real> p);
#endif

    template <class T>
    inline void hash_combine(std::size_t & seed, const T & v)
    {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    template<class Real_ = double>
    struct DiagramPoint
    {
        using Real = Real_;
        // data members
        // Points above the diagonal have type NORMAL
        // Projections onto the diagonal have type DIAG
        // for DIAG points only x-coordinate is relevant
        enum Type { NORMAL, DIAG};
        Real x, y;
        Type type;
        // methods
        DiagramPoint(Real xx, Real yy, Type ttype);
        bool is_diagonal() const { return type == DIAG; }
        bool is_normal() const { return type == NORMAL; }
        Real getRealX() const; // return the x-coord
        Real getRealY() const; // return the y-coord
        Real persistence_lp(const Real p) const;
        struct LexicographicCmp
        {
            bool    operator()(const DiagramPoint& p1, const DiagramPoint& p2) const
            { return p1.type < p2.type || (p1.type == p2.type && (p1.x < p2.x || (p1.x == p2.x && p1.y < p2.y))); }
        };

        const Real& operator[](const int idx) const
        {
            switch(idx)
            {
                case 0 : return x;
                         break;
                case 1 : return y;
                         break;
                default: throw std::out_of_range("DiagramPoint has dimension 2");
            }
        }

        Real& operator[](const int idx)
        {
            switch(idx)
            {
                case 0 : return x;
                         break;
                case 1 : return y;
                         break;
                default: throw std::out_of_range("DiagramPoint has dimension 2");
            }
        }

    };


    template<class Real>
    struct DiagramPointHash {
        size_t operator()(const DiagramPoint<Real> &p) const
        {
            std::size_t seed = 0;
            hash_combine(seed, std::hash<Real>(p.x));
            hash_combine(seed, std::hash<Real>(p.y));
            hash_combine(seed, std::hash<bool>(p.is_diagonal()));
            return seed;
        }
    };


#ifndef FOR_R_TDA
    template <class Real = double>
    inline std::ostream& operator<<(std::ostream& output, const DiagramPoint<Real> p);
#endif

    template<class Real>
    inline void format_arg(fmt::BasicFormatter<char> &f, const char *&format_str, const DiagramPoint<Real>&p) {
        if (p.is_diagonal()) {
            f.writer().write("({0},{1}, DIAG)", p.x, p.y);
        } else {
            f.writer().write("({0},{1}, NORM)", p.x, p.y);
        }
    }


    template<class Real, class Pt>
    struct DistImpl
    {
        Real operator()(const Pt& a, const Pt& b, const Real p, const int dim)
        {
            Real result = 0.0;
            if (hera::is_infinity(p)) {
                for(int d = 0; d < dim; ++d) {
                    result = std::max(result, std::fabs(a[d] - b[d]));
                }
            } else if (p == 1.0) {
                for(int d = 0; d < dim; ++d) {
                    result += std::fabs(a[d] - b[d]);
                }
            } else {
                assert(p > 1.0);
                for(int d = 0; d < dim; ++d) {
                    result += std::pow(std::fabs(a[d] - b[d]), p);
                }
                result = std::pow(result, 1.0 / p);
            }
            return result;
        }
    };

    template<class Real>
    struct DistImpl<Real, DiagramPoint<Real>>
    {
        Real operator()(const DiagramPoint<Real>& a, const DiagramPoint<Real>& b, const Real p, const int dim)
        {
            Real result = 0.0;
            if ( a.is_diagonal() and b.is_diagonal()) {
                return result;
            } else if (hera::is_infinity(p)) {
               result = std::max(std::fabs(a.getRealX() - b.getRealX()), std::fabs(a.getRealY() - b.getRealY()));
            } else if (p == 1.0) {
                result = std::fabs(a.getRealX() - b.getRealX()) + std::fabs(a.getRealY() - b.getRealY());
            } else {
                assert(p > 1.0);
                result = std::pow(std::pow(std::fabs(a.getRealX() - b.getRealX()), p) + std::pow(std::fabs(a.getRealY() - b.getRealY()), p), 1.0 / p);
            }
            return result;
        }
    };

    template<class R, class Pt>
    inline R dist_lp(const Pt& a, const Pt& b, const R p, const int dim)
    {
        return DistImpl<R, Pt>()(a, b, p, dim);
    }

    // TODO
    template<class Real, typename DiagPointContainer>
    inline double getFurthestDistance3Approx(DiagPointContainer& A, DiagPointContainer& B, const Real p)
    {
        int dim = 2;
        Real result { 0.0 };
        DiagramPoint<Real> begA = *(A.begin());
        DiagramPoint<Real> optB = *(B.begin());
        for(const auto& pointB : B) {
            if (dist_lp(begA, pointB, p, dim) > result) {
                result = dist_lp(begA, pointB, p, dim);
                optB = pointB;
            }
        }
        for(const auto& pointA : A) {
            if (dist_lp(pointA, optB, p, dim) > result) {
                result = dist_lp(pointA, optB, p, dim);
            }
        }
        return result;
    }

    template<class Real>
    inline Real getFurthestDistance3Approx_pg(const hera::ws::dnn::DynamicPointVector<Real>& A, const hera::ws::dnn::DynamicPointVector<Real>& B, const Real p, const int dim)
    {
        Real result { 0.0 };
        int opt_b_idx = 0;
        for(size_t b_idx = 0; b_idx < B.size(); ++b_idx) {
            if (dist_lp(A[0], B[b_idx], p, dim) > result) {
                result = dist_lp(A[0], B[b_idx], p, dim);
                opt_b_idx = b_idx;
            }
        }

        for(size_t a_idx = 0; a_idx < A.size(); ++a_idx) {
            result = std::max(result,  dist_lp(A[a_idx], B[opt_b_idx], p, dim));
        }

        return result;
    }


    template<class Container>
    inline std::string format_container_to_log(const Container& cont);

    template<class Real, class IndexContainer>
    inline std::string format_point_set_to_log(const IndexContainer& indices, const std::vector<DiagramPoint<Real>>& points);

    template<class T>
    inline std::string format_int(T i);

} // ws
} // hera



#include "basic_defs_ws.hpp"


#endif
