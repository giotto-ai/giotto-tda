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

#ifndef AUCTION_ORACLE_LAZY_HEAP_H
#define AUCTION_ORACLE_LAZY_HEAP_H


#define USE_BOOST_HEAP

#include <map>
#include <memory>
#include <set>
#include <list>

#ifdef USE_BOOST_HEAP
#include <boost/heap/d_ary_heap.hpp>
#endif

#include "basic_defs_ws.h"

namespace ws {

template <typename T>
struct CompPairsBySecondStruct {
    bool operator()(const IdxValPair<T>& a, const IdxValPair<T>& b) const
    {
        return a.second < b.second;
    }
};


template <typename T>
struct CompPairsBySecondGreaterStruct {
    bool operator()(const IdxValPair<T>& a, const IdxValPair<T>& b) const
    {
        return a.second > b.second;
    }
};

template <typename T>
struct CompPairsBySecondLexStruct {
    bool operator()(const IdxValPair<T>& a, const IdxValPair<T>& b) const
    {
        return a.second < b.second or (a.second == b.second and a.first > b.first);
    }
};

template <typename T>
struct CompPairsBySecondLexGreaterStruct {
    bool operator()(const IdxValPair<T>& a, const IdxValPair<T>& b) const
    {
        return a.second > b.second or (a.second == b.second and a.first > b.first);
    }
};

using ItemsTimePair = std::pair<IdxType, int>;
using UpdateList = std::list<ItemsTimePair>;
using UpdateListIter = UpdateList::iterator;


#ifdef USE_BOOST_HEAP
template <class Real>
using LossesHeap = boost::heap::d_ary_heap<IdxValPair<Real>, boost::heap::arity<2>, boost::heap::mutable_<true>, boost::heap::compare<CompPairsBySecondGreaterStruct<Real>>>;
#else
template<typename T, class ComparisonStruct>
class IdxValHeap {
public:
    using InternalKeeper = std::set<IdxValPair<T>, ComparisonStruct>;
    using handle_type = typename InternalKeeper::iterator;
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


private:
    std::set<IdxValPair<T>, ComparisonStruct> _heap;
};

// if we store losses, the minimal value should come first
template <class Real>
using LossesHeap = IdxValHeap<Real, CompPairsBySecondLexStruct>;
#endif


template <class Real = double>
struct AuctionOracleLazyHeapRestricted : AuctionOracleBase<Real> {

    using LossesHeapR = typename ws::LossesHeap<Real>;
    using LossesHeapRHandle = typename ws::LossesHeap<Real>::handle_type;
    using DiagramPointR = typename ws::DiagramPoint<Real>;


     AuctionOracleLazyHeapRestricted(const std::vector<DiagramPointR>& bidders, const std::vector<DiagramPointR>& items, const Real wasserstein_power, const Real _internal_p = get_infinity<Real>());
    ~AuctionOracleLazyHeapRestricted();
    // data members
    // temporarily make everything public
    std::vector<std::vector<Real>> weight_matrix;
    //Real weight_adj_const;
    Real max_val;
    // vector of heaps to find the best items
    std::vector<LossesHeapR*> losses_heap;
    std::vector<std::vector<size_t>> items_indices_for_heap_handles;
    std::vector<std::vector<LossesHeapRHandle>> losses_heap_handles;
    // methods
    void fill_in_losses_heap();
    void set_price(const IdxType items_idx, const Real new_price);
    IdxValPair<Real> get_optimal_bid(const IdxType bidder_idx);
    Real get_matching_weight(const std::vector<IdxType>& bidders_to_items) const;
    void adjust_prices();
    // to update the queue in lazy fashion
    std::vector<UpdateListIter> items_iterators;
    UpdateList update_list;
    std::vector<int> bidders_update_moments;
    int update_counter;
    void update_queue_for_bidder(const IdxType bidder_idx);
    LossesHeapR diag_items_heap;
    std::vector<LossesHeapRHandle> diag_heap_handles;
    std::vector<size_t> heap_handles_indices;
    // debug

    DebugOptimalBid<Real> get_optimal_bid_debug(const IdxType bidder_idx);

    // for diagonal points
    bool best_diagonal_items_computed;
    size_t best_diagonal_item_idx;
    Real best_diagonal_item_value;
    size_t second_best_diagonal_item_idx;
    Real second_best_diagonal_item_value;
};

} // end of namespace ws

#include "auction_oracle_lazy_heap.h"

#endif
