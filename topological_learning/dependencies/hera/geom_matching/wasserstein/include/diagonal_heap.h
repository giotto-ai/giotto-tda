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

#ifndef DIAGONAL_HEAP_H
#define DIAGONAL_HEAP_H

//#define USE_BOOST_HEAP

#include <map>
#include <memory>
#include <set>
#include <list>

#ifdef USE_BOOST_HEAP
#include <boost/heap/d_ary_heap.hpp>
#endif

#include "basic_defs_ws.h"

namespace hera {
namespace ws {

template <typename T>
struct CompPairsBySecondLexStruct {
    bool operator()(const IdxValPair<T>& a, const IdxValPair<T>& b) const
    {
        return a.second < b.second or (a.second == b.second and a.first > b.first);
    }
};


template <typename T>
struct CompPairsBySecondGreaterStruct {
    bool operator()(const IdxValPair<T>& a, const IdxValPair<T>& b) const
    {
        return a.second > b.second;
    }
};

#ifdef USE_BOOST_HEAP
template <class Real>
using LossesHeapOld = boost::heap::d_ary_heap<IdxValPair<Real>, boost::heap::arity<2>, boost::heap::mutable_<true>, boost::heap::compare<CompPairsBySecondGreaterStruct<Real>>>;
#else
template<typename T, class ComparisonStruct>
class IdxValHeap {
public:
    using InternalKeeper = std::set<IdxValPair<T>, ComparisonStruct>;
    using handle_type = typename InternalKeeper::iterator;
    using const_handle_type = typename InternalKeeper::const_iterator;
    // methods
    handle_type push(const IdxValPair<T>& val)
    {
        auto res_pair = _heap.insert(val);
        assert(res_pair.second);
        assert(res_pair.first != _heap.end());
        return res_pair.first;
    }

    void decrease(handle_type& handle, const IdxValPair<T>& new_val)
    {
        _heap.erase(handle);
        handle = push(new_val);
    }

    void increase(handle_type& handle, const IdxValPair<T>& new_val)
    {
        _heap.erase(handle);
        handle = push(new_val);
    }

    size_t size() const
    {
        return _heap.size();
    }

    handle_type ordered_begin()
    {
        return _heap.begin();
    }

    handle_type ordered_end()
    {
        return _heap.end();
    }

    const_handle_type ordered_begin() const
    {
        return _heap.cbegin();
    }

    const_handle_type ordered_end() const
    {
        return _heap.cend();
    }


private:
    std::set<IdxValPair<T>, ComparisonStruct> _heap;
};

// if we store losses, the minimal value should come first
template <class Real>
using LossesHeapOld = IdxValHeap<Real, CompPairsBySecondLexStruct<Real>>;
#endif

template <class Real>
inline std::string losses_heap_to_string(const LossesHeapOld<Real>& h)
{
    std::stringstream result;
    result << "[";
    for(auto iter = h.ordered_begin(); iter != h.ordered_end(); ++iter) {
        result << *iter;
        if (std::next(iter) != h.ordered_end()) {
            result << ", ";
        }
    }
    result << "]";
    return result.str();
}

} // ws
} // hera

#endif // DIAGONAL_HEAP_H
