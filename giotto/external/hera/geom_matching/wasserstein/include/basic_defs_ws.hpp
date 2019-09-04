/*

Copyright (c) 2015, M. Kerber, D. Morozov, A. Nigmetov
Copyright (c) 2018, G. Spreemann
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

#pragma once

#include <algorithm>
#include <cfloat>
#include <set>
#include <algorithm>
#include <istream>
#include <cstdint>
#include "basic_defs_ws.h"

#ifndef FOR_R_TDA
#include <iostream>
#endif

#include <sstream>

namespace hera {
static const int64_t DIPHA_MAGIC = 8067171840;
static const int64_t DIPHA_PERSISTENCE_DIAGRAM = 2;

namespace ws {
// Point

template <class Real>
bool Point<Real>::operator==(const Point<Real>& other) const
{
    return ((this->x == other.x) and (this->y == other.y));
}

template <class Real>
bool Point<Real>::operator!=(const Point<Real>& other) const
{
    return !(*this == other);
}


#ifndef FOR_R_TDA
template <class Real>
inline std::ostream& operator<<(std::ostream& output, const Point<Real> p)
{
    output << "(" << p.x << ", " << p.y << ")";
    return output;
}
#endif

template <class Real>
inline Real sqr_dist(const Point<Real>& a, const Point<Real>& b)
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
}

template <class Real>
inline Real dist(const Point<Real>& a, const Point<Real>& b)
{
    return sqrt(sqr_dist(a, b));
}


template <class Real>
inline Real DiagramPoint<Real>::persistence_lp(const Real p) const
{
    if (is_diagonal())
        return 0.0;
    else {
        Real u { (getRealY() + getRealX())/2 };
        int dim = 2;
        DiagramPoint<Real> a_proj(u, u, DiagramPoint<Real>::DIAG);
        return dist_lp(*this, a_proj, p, dim);
    }
}


#ifndef FOR_R_TDA
template <class Real>
inline std::ostream& operator<<(std::ostream& output, const DiagramPoint<Real> p)
{
    if ( p.type == DiagramPoint<Real>::DIAG ) {
        output << "(" << p.x << ", " << p.y << ", " <<  0.5 * (p.x + p.y) << " DIAG )";
    } else {
        output << "(" << p.x << ", " << p.y << ", " << " NORMAL)";
    }
    return output;
}
#endif

template <class Real>
DiagramPoint<Real>::DiagramPoint(Real xx, Real yy, Type ttype) :
    x(xx),
    y(yy),
    type(ttype)
{
    //if ( yy < xx )
        //throw "Point is below the diagonal";
    //if ( yy == xx and ttype != DiagramPoint<Real>::DIAG)
        //throw "Point on the main diagonal must have DIAG type";
}

template <class Real>
Real DiagramPoint<Real>::getRealX() const
{
    if (is_normal())
        return x;
    else
        return Real(0.5) * (x + y);
}

template <class Real>
Real DiagramPoint<Real>::getRealY() const
{
    if (is_normal())
        return y;
    else
        return Real(0.5) * (x + y);
}

template<class Container>
inline std::string format_container_to_log(const Container& cont)
{
    std::stringstream result;
    result << "[";
    for(auto iter = cont.begin(); iter != cont.end(); ++iter) {
        result << *iter;
        if (std::next(iter) != cont.end()) {
            result << ", ";
        }
    }
    result << "]";
    return result.str();
}

template<class Container>
inline std::string format_pair_container_to_log(const Container& cont)
{
    std::stringstream result;
    result << "[";
    for(auto iter = cont.begin(); iter != cont.end(); ++iter) {
        result << "(" << iter->first << ", " << iter->second << ")";
        if (std::next(iter) != cont.end()) {
            result << ", ";
        }
    }
    result << "]";
    return result.str();
}


template<class Real, class IndexContainer>
inline std::string format_point_set_to_log(const IndexContainer& indices,
                                    const std::vector<DiagramPoint<Real>>& points)
{
    std::stringstream result;
    result << "[";
    for(auto iter = indices.begin(); iter != indices.end(); ++iter) {
        DiagramPoint<Real> p = points[*iter];
        result << "(" << p.getRealX() << ", " << p.getRealY() << ")";
        if (std::next(iter) != indices.end())
            result << ", ";
    }
    result << "]";
    return result.str();
}

template<class T>
inline std::string format_int(T i)
{
    std::stringstream ss;
    ss.imbue(std::locale(""));
    ss << std::fixed << i;
    return ss.str();
}


} // end of namespace ws


template <typename T> inline void reverse_endianness(T & x)
{
    uint8_t * p = reinterpret_cast<uint8_t *>(&x);
    std::reverse(p, p + sizeof(T));
}

template <typename T> inline T read_le(std::istream & s)
{
    T result;
    s.read(reinterpret_cast<char *>(&result), sizeof(T));
    #ifdef BIGENDIAN
    reverse_endianness(result);
    #endif
    return result;
}

} // hera
