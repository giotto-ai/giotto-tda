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

#ifndef AUCTION_ORACLE_KDTREE_SINGLE_DIAG_H
#define AUCTION_ORACLE_KDTREE_SINGLE_DIAG_H


#include <map>
#include <memory>
#include <set>
#include <list>
#include <ostream>

#include "basic_defs_ws.h"
#include "dnn/geometry/euclidean-fixed.h"
#include "dnn/local/kd-tree.h"

namespace hera {
namespace ws {


template <class Real = double>
struct ItemSlice;

template <class Real = double>
bool operator<(const ItemSlice<Real>& s_1, const ItemSlice<Real>& s_2);

template <class Real = double>
bool operator>(const ItemSlice<Real>& s_1, const ItemSlice<Real>& s_2);

template <class Real>
struct ItemSlice {
public:
    using RealType = Real;

    size_t item_idx;
    Real loss;
    ItemSlice(size_t _item_idx, const Real _loss);

    void set_loss(const Real new_loss) { loss = new_loss; }
    void adjust_loss(const Real delta) { loss -= delta; }

    friend bool operator< <>(const ItemSlice<Real>&, const ItemSlice<Real>&);
    friend bool operator> <>(const ItemSlice<Real>&, const ItemSlice<Real>&);

private:
};


template <class Real = double>
class LossesHeap {
public:
    using ItemSliceR  = ItemSlice<Real>;
    using KeeperTypeR = std::set<ItemSliceR, std::less<ItemSliceR> >;
    using IterTypeR   = typename KeeperTypeR::iterator;

    LossesHeap() {}
    LossesHeap(const std::vector<ItemSliceR>&);
    void adjust_prices(const Real delta);  // subtract delta from all values
    ItemSliceR get_best_slice() const;
    ItemSliceR get_second_best_slice() const;

    template<typename ...Args>
    decltype(auto) emplace(Args&&... args)
    {
        return keeper.emplace(std::forward<Args>(args)...);
    }


    IterTypeR begin()  { return keeper.begin(); }
    IterTypeR end()    { return keeper.end(); }
    void erase(IterTypeR iter) { assert(iter != keeper.end()); keeper.erase(iter); }
    decltype(auto) insert(const ItemSliceR& item) { return keeper.insert(item); }
    size_t size() const { return keeper.size(); }
    bool empty() const { return keeper.empty(); }
//private:
    std::set<ItemSliceR, std::less<ItemSliceR> > keeper;
};

template <class Real>
struct DiagonalBid {
    DiagonalBid() {}

    std::vector<size_t> assigned_normal_items;
    std::vector<Real> assigned_normal_items_bid_values;

    std::vector<size_t> best_item_indices;
    std::vector<Real> bid_values;

    // common bid value for diag-diag
    Real diag_to_diag_value { 0.0 };

    // analogous to second best item value; denoted by w in Bertsekas's paper on auction for transportation problem
    Real almost_best_value { 0.0 };

    // how many points to get from unassigned diagonal chunk
    int num_from_unassigned_diag { 0 };
};

template <class Real_ = double, class PointContainer_ = std::vector<DiagramPoint<Real_>>>
struct AuctionOracleKDTreeSingleDiag : AuctionOracleBase<Real_, PointContainer_> {

    using PointContainer        = PointContainer_;
    using Real                  = Real_;

    using DnnPoint              = dnn::Point<2, Real>;
    using DnnTraits             = dnn::PointTraits<DnnPoint>;

    using IdxValPairR           = typename ws::IdxValPair<Real>;
    using ItemSliceR            = typename ws::ItemSlice<Real>;
    using LossesHeapR           = typename ws::LossesHeap<Real>;
    using LossesHeapIterR       = typename ws::LossesHeap<Real>::IterTypeR;
    using DiagramPointR         = typename ws::DiagramPoint<Real>;
    using DiagonalBidR          = typename ws::DiagonalBid<Real>;

    AuctionOracleKDTreeSingleDiag(const PointContainer& bidders,
                                  const PointContainer& items,
                                  const AuctionParams<Real>& params);
    ~AuctionOracleKDTreeSingleDiag();
    // data members
    // temporarily make everything public
    Real max_val_;
    size_t num_diag_items_;
    size_t num_normal_items_;
    size_t num_diag_bidders_;
    size_t num_normal_bidders_;
    dnn::KDTree<DnnTraits>* kdtree_;
    std::vector<DnnPoint> dnn_points_;
    std::vector<DnnPoint*> dnn_point_handles_;
    std::vector<size_t> kdtree__items_;

    // this heap is used by off-diagonal bidders to get the cheapest diagonal
    // item; index in the IdxVal is a valid item index in the vector of items
    // items in diag_assigned_to_diag_slice_ and in diag_unassigned_slice_
    // are not stored in this heap
    LossesHeapR diag_items_heap_;
    // vector of iterators; if item_idx is in diag_assigned_to_diag_slice_ or
    // in diag_unassigned_slice_, then diag_items_heap__iters_[item_idx] ==
    // diag_items_heap_.end()
    std::vector<LossesHeapIterR> diag_items_heap__iters_;


    // this heap is used by _the_ diagonal bidder to get the cheapest items
    // * value in IdxValPair is price + persistence (i.e., price for
    // diagonal items)
    // * index in IdxValPair is a valid item index in the vector of items
    // items in diag_assigned_to_diag_slice_ and in diag_unassigned_slice_
    // are not stored in this heap
    LossesHeapR all_items_heap_;
    std::vector<LossesHeapIterR> all_items_heap__iters_;

    std::unordered_set<size_t> diag_assigned_to_diag_slice_;
    std::unordered_set<size_t> diag_unassigned_slice_;


    std::unordered_set<size_t> normal_items_assigned_to_diag_;

    Real diag_to_diag_price_;
    Real diag_unassigned_price_;

    // methods
    Real get_price(const size_t item_idx) const override;
    void set_price(const size_t item_idx,
                   const Real new_price,
                   const bool item_is_diagonal,
                   const bool bidder_is_diagonal,
                   const OwnerType old_owner_type);

    IdxValPair<Real> get_optimal_bid(const IdxType bidder_idx);

    DiagonalBidR get_optimal_bids_for_diagonal(int unassigned_mass);
    void process_unassigned_diagonal(const int unassigned_mass,
                                     int& accumulated_mass,
                                     bool& saw_diagonal_slice,
                                     int& num_classes,
                                     Real& w,
                                     DiagonalBidR& result,
                                     bool& found_w);

    void adjust_prices();
    void flush_assignment();
    void sanity_check();

    bool is_item_diagonal(const size_t item_idx) const;
    bool is_item_normal(const size_t item_idx) const { return not is_item_diagonal(item_idx); }

};

} // ws
} // hera

#include "auction_oracle_kdtree_single_diag.hpp"

#endif
