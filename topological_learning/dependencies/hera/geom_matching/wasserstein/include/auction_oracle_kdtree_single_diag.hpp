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
#ifndef AUCTION_ORACLE_KDTREE_RESTRICTED_SINGLE_DIAG_HPP
#define AUCTION_ORACLE_KDTREE_RESTRICTED_SINGLE_DIAG_HPP

#include <assert.h>
#include <algorithm>
#include <functional>
#include <iterator>

#include "def_debug_ws.h"
#include "auction_oracle.h"


#ifdef FOR_R_TDA
#undef DEBUG_AUCTION
#endif

namespace hera {
namespace ws {

// *****************************
// AuctionOracleKDTreeSingleDiag
// *****************************



template <class Real>
ItemSlice<Real>::ItemSlice(size_t _item_idx,
                           const Real _loss) :
    item_idx(_item_idx),
    loss(_loss)
{
}


template <class Real>
bool operator<(const ItemSlice<Real>& s_1, const ItemSlice<Real>& s_2)
{
    return s_1.loss < s_2.loss
           or (s_1.loss == s_2.loss and s_1.item_idx < s_2.item_idx);
}

template <class Real>
bool operator>(const ItemSlice<Real>& s_1, const ItemSlice<Real>& s_2)
{
    return s_1.loss > s_2.loss
           or (s_1.loss == s_2.loss and s_1.item_idx > s_2.item_idx);
}

template<class Real>
std::ostream& operator<<(std::ostream& s, const ItemSlice<Real>& x)
{
    s << "(" << x.item_idx << ", " << x.loss << ")";
    return s;
}

// *****************************
// LossesHeap
// *****************************


template <class Real>
void LossesHeap<Real>::adjust_prices(const Real delta)
{
    throw std::runtime_error("not implemented");
}

template <class Real>
typename LossesHeap<Real>::ItemSliceR LossesHeap<Real>::get_best_slice() const
{
    return *(keeper.begin());
}

template <class Real>
typename LossesHeap<Real>::ItemSliceR LossesHeap<Real>::get_second_best_slice() const
{
    if (keeper.size() > 1) {
        return *std::next(keeper.begin());
    } else {
        return ItemSliceR(k_invalid_index, std::numeric_limits<Real>::max());
    }
}

template<class Real>
std::ostream& operator<<(std::ostream& s, const LossesHeap<Real>& x)
{
    s << "Heap[ ";
    for(auto iter = x.keeper.begin(); iter != x.keeper.end(); ++iter) {
        s << *iter << "\n";
    }
    s << "]\n";
    return s;
}

// *****************************
// DiagonalBid
// *****************************

template <class Real>
std::ostream& operator<<(std::ostream& s, const DiagonalBid<Real>& b)
{
    s << "DiagonalBid { num_from_unassigned_diag = " << b.num_from_unassigned_diag;
    s << ", diag_to_diag_value = " << b.diag_to_diag_value;
    s << ", almost_best_value =  " << b.almost_best_value;
    s << ",\nbest_item_indices =  [";
    for(const auto i : b.best_item_indices) {
        s << i << ", ";
    }
    s << "]\n";

    s << ",\nbid_values=  [";
    for(const auto v : b.bid_values) {
        s << v << ", ";
    }
    s << "]\n";

    s << ",\nassigned_normal_items=  [";
    for(const auto i : b.assigned_normal_items) {
        s << i << ", ";
    }
    s << "]\n";

    s << ",\nassigned_normal_items_bid_values =  [";
    for(const auto v : b.assigned_normal_items_bid_values) {
        s << v << ", ";
    }
    s << "]\n";

    return s;
}

// *****************************
// AuctionOracleKDTreeSingleDiag
// *****************************

template<class Real_, class PointContainer_>
std::ostream& operator<<(std::ostream& s, const AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>& x)
{
    s << "oracle: bidders" << std::endl;
    for(const auto& p : x.bidders) {
        s << p << "\n";
    }
    s << "items:";

    for(const auto& p : x.items) {
        s << p << "\n";
    }

    s << "diag_unassigned_slice_.size = " << x.diag_unassigned_slice_.size() << ", ";
    s << "diag_unassigned_price_ = " << x.diag_unassigned_price_ << ", ";
    s << "diag unassigned slice [";

    for(const auto& i : x.diag_unassigned_slice_) {
        s << i << ", ";
    }
    s << "]\n ";

    s << "diag_assigned_to_diag_slice_.size = " << x.diag_assigned_to_diag_slice_.size() << ",  ";
    s << "diag_assigned_to_diag_price = " << x.diag_to_diag_price_ << "\n";
    s << "diag_assigned_to_diag_slice_ [";

    for(const auto& i : x.diag_assigned_to_diag_slice_) {
        s << i << ", ";
    }
    s << "]\n ";

    s << "diag_items_heap_.size = " << x.diag_items_heap_.size() << "\n ";
    s << x.diag_items_heap_;

    s << "all_items_heap_.size = " << x.all_items_heap_.size() << "\n ";
    s << x.all_items_heap_;

    s << "epsilon = " << x.epsilon << std::endl;

    return s;
}


template<class Real_, class PointContainer_>
AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::AuctionOracleKDTreeSingleDiag(const PointContainer_& _bidders,
                                                                                     const PointContainer_& _items,
                                                                                     const AuctionParams<Real>& params) :
    AuctionOracleBase<Real_, PointContainer_>(_bidders, _items, params),
    max_val_(std::pow( 3.0 * getFurthestDistance3Approx<>(_bidders, _items, params.internal_p), params.wasserstein_power)),
    num_diag_items_(0),
    kdtree__items_(_items.size(), k_invalid_index)
{
    size_t dnn_item_idx { 0 };
    dnn_points_.clear();

    all_items_heap__iters_.clear();
    all_items_heap__iters_.reserve( 4 * _items.size() / 7);


    for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        const auto& item = this->items[item_idx];
        if (item.is_normal() ) {
            // store normal items in kd-tree
            kdtree__items_[item_idx] = dnn_item_idx;
            // index of items is id of dnn-point
            DnnPoint p(item_idx);
            p[0] = item.x;
            p[1] = item.y;
            dnn_points_.push_back(p);
            assert(dnn_item_idx == dnn_points_.size() - 1);
            dnn_item_idx++;
            // add slice to vector
            auto ins_res = all_items_heap_.emplace(item_idx, this->get_value_for_diagonal_bidder(item_idx));
            all_items_heap__iters_.push_back(ins_res.first);
            assert(ins_res.second);
        } else {
            // all diagonal items are initially in the unassigned slice
            diag_unassigned_slice_.insert(item_idx);
            all_items_heap__iters_.push_back(all_items_heap_.end());
            diag_items_heap__iters_.push_back(diag_items_heap_.end());
            ++num_diag_items_;
        }
    }

    num_normal_items_ = this->items.size() - num_diag_items_;
    num_normal_bidders_ = num_diag_items_;
    num_diag_bidders_ = this->bidders.size() - num_normal_bidders_;

    assert(dnn_points_.size() < _items.size() );
    for(size_t i = 0; i < dnn_points_.size(); ++i) {
        dnn_point_handles_.push_back(&dnn_points_[i]);
    }

    DnnTraits traits;
    traits.internal_p = params.internal_p;

    kdtree_ = new dnn::KDTree<DnnTraits>(traits, dnn_point_handles_, this->wasserstein_power);

    sanity_check();

    //std::cout << "IN CTOR: " << *this << std::endl;

}


template<class Real_, class PointContainer_>
void AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::process_unassigned_diagonal(const int unassigned_mass, int& accumulated_mass, bool& saw_diagonal_slice, int& num_classes, Real& w, DiagonalBidR& result, bool& found_w)
{
    result.num_from_unassigned_diag = std::min(static_cast<int>(diag_unassigned_slice_.size()), static_cast<int>(unassigned_mass - accumulated_mass));
    if (not saw_diagonal_slice) {
        saw_diagonal_slice = true;
        ++num_classes;
    }

    accumulated_mass += result.num_from_unassigned_diag;
    //std::cout << "got mass from diagunassigned_slice, result.num_from_unassigned_diag = " << result.num_from_unassigned_diag << ", accumulated_mass = " << accumulated_mass << std::endl;

    if (static_cast<int>(diag_unassigned_slice_.size()) > result.num_from_unassigned_diag and num_classes >= 2) {
        found_w = true;
        w = diag_unassigned_price_;
        //std::cout << "w found from diag_unassigned_slice_, too, w = " << w << std::endl;
        result.almost_best_value = w;
    }

}


template<class Real_, class PointContainer_>
typename AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::DiagonalBidR AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::get_optimal_bids_for_diagonal(int unassigned_mass)
{
    sanity_check();

    assert(unassigned_mass == static_cast<decltype(unassigned_mass)>(num_diag_bidders_)
                              - static_cast<decltype(unassigned_mass)>(normal_items_assigned_to_diag_.size())
                              - static_cast<decltype(unassigned_mass)>(diag_assigned_to_diag_slice_.size()) );
    assert(unassigned_mass > 0);

    DiagonalBidR result;


    // number of similarity classes already assigned to diagonal bidder
    // each normal point is a single class
    // all diagonal points are in one class
    int num_classes = normal_items_assigned_to_diag_.size() + ( diag_assigned_to_diag_slice_.empty() ? 0 : 1 );
    bool saw_diagonal_slice = not diag_assigned_to_diag_slice_.empty();
    bool found_w = false;

    //std::cout << "Enter get_optimal_bids_for_diagonal, unassigned_mass = " << unassigned_mass <<", num_classes = " << num_classes << ", saw_diagonal_slice = " << std::boolalpha << saw_diagonal_slice << std::endl;

    decltype(unassigned_mass) accumulated_mass = 0;


    Real w { std::numeric_limits<Real>::max() };
    bool unassigned_not_processed = not diag_unassigned_slice_.empty();

    for(auto slice_iter = all_items_heap_.begin(); slice_iter != all_items_heap_.end(); ++slice_iter) {

        auto slice = *slice_iter;

        if ( is_item_normal(slice.item_idx) and normal_items_assigned_to_diag_.count(slice.item_idx) == 1) {
            //std::cout << __LINE__ << ": skipping slice " << slice << std::endl;
            // this item is already assigned to diagonal bidder, skip
            continue;
        }

        if (unassigned_not_processed and slice.loss >= diag_unassigned_price_) {
            // diag_unassigned slice is better,
            // process it first
            process_unassigned_diagonal(unassigned_mass, accumulated_mass, saw_diagonal_slice, num_classes, w, result, found_w);
            unassigned_not_processed = false;
            if (accumulated_mass >= unassigned_mass and found_w) {
                break;
            }
        }


        if (is_item_normal(slice.item_idx)) {
            // all off-diagonal items are distinct
            ++num_classes;
        } else if (not saw_diagonal_slice) {
            saw_diagonal_slice = true;
            ++num_classes;
        }

        if (accumulated_mass < unassigned_mass) {
            //std::cout << __LINE__ << ": added slice to best items " << slice << std::endl;
            result.best_item_indices.push_back(slice.item_idx);
        }

        if (accumulated_mass >= unassigned_mass and num_classes >= 2) {
            //std::cout << "Found w, slice = " << slice << std::endl;
            w = slice.loss;
            found_w = true;
            result.almost_best_value = w;
            break;
        }

        // all items in all_items heap have mass 1
        ++accumulated_mass;
        //std::cout << "accumulated_mass = " << accumulated_mass << std::endl;

    }

    if (unassigned_not_processed and (accumulated_mass < unassigned_mass or not found_w)) {
        process_unassigned_diagonal(unassigned_mass, accumulated_mass, saw_diagonal_slice, num_classes, w, result, found_w);
    }

    assert(found_w);

    //if (w == std::numeric_limits<Real>::max()) { std::cout << "HERE: " << *this << std::endl; }
    assert(w != std::numeric_limits<Real>::max());

    result.assigned_normal_items.clear();
    result.assigned_normal_items_bid_values.clear();

    result.assigned_normal_items.reserve(normal_items_assigned_to_diag_.size());
    result.assigned_normal_items_bid_values.reserve(normal_items_assigned_to_diag_.size());

    // add already assigned normal items and their new prices to bid
    for(const auto item_idx : normal_items_assigned_to_diag_) {
        assert( all_items_heap__iters_[item_idx] != all_items_heap_.end() );
        assert( is_item_normal(item_idx) );

        result.assigned_normal_items.push_back(item_idx);
        Real bid_value = w - this->get_cost_for_diagonal_bidder(item_idx) + this->epsilon;
        //if ( bid_value <= this->get_price(item_idx) ) {
            //std::cout << bid_value << " vs price " << this->get_price(item_idx) << std::endl;
            //std::cout << *this << std::endl;
        //}
        assert( bid_value >= this->get_price(item_idx) );
        result.assigned_normal_items_bid_values.push_back(bid_value);
    }

    // calculate bid values
    // diag-to-diag items all have the same bid value
    if (saw_diagonal_slice) {
        result.diag_to_diag_value = w + this->epsilon;
    } else {
        result.diag_to_diag_value = std::numeric_limits<Real>::max();
    }

    result.bid_values.reserve(result.best_item_indices.size());
    for(const auto item_idx : result.best_item_indices) {
        Real bid_value = w - this->get_cost_for_diagonal_bidder(item_idx) + this->epsilon;
        result.bid_values.push_back(bid_value);
    }

    return result;
}


template<class Real_, class PointContainer_>
IdxValPair<Real_> AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::get_optimal_bid(IdxType bidder_idx)
{
    //std::cout << "enter get_optimal_bid" << std::endl;
    sanity_check();

    auto bidder = this->bidders[bidder_idx];

    size_t best_item_idx;
    Real best_item_price;
    Real best_item_value;
    Real second_best_item_value;

    // this function is for normal bidders only
    assert(bidder.is_normal());


    // get 2 best items among non-diagonal points from kdtree_
    DnnPoint bidder_dnn;
    bidder_dnn[0] = bidder.getRealX();
    bidder_dnn[1] = bidder.getRealY();
    auto two_best_items = kdtree_->findK(bidder_dnn, 2);
    size_t best_normal_item_idx { two_best_items[0].p->id() };
    Real best_normal_item_value { two_best_items[0].d };
    // if there is only one off-diagonal point in the second diagram,
    // kd-tree will not return the second candidate.
    // Set its price to inf, so it will always lose to the price of the projection
    Real second_best_normal_item_value { two_best_items.size() == 1 ? std::numeric_limits<Real>::max() : two_best_items[1].d };

    size_t best_diag_item_idx;
    Real best_diag_value;
    Real best_diag_price;

    {
        Real diag_edge_cost = std::pow(bidder.persistence_lp(this->internal_p), this->wasserstein_power);
        auto best_diag_price_in_heap = diag_items_heap_.empty() ? std::numeric_limits<Real>::max() : diag_items_heap_.get_best_slice().loss;
        auto best_diag_idx_in_heap = diag_items_heap_.empty() ? k_invalid_index : diag_items_heap_.get_best_slice().item_idx;
        // if unassigned_diag_slice is empty, its price is max,
        // same for diag-diag assigned slice, so the ifs below will work

        if (best_diag_price_in_heap <= diag_to_diag_price_ and best_diag_price_in_heap <= diag_unassigned_price_) {
            best_diag_item_idx = best_diag_idx_in_heap;
            best_diag_value = diag_edge_cost + best_diag_price_in_heap;
            best_diag_price = best_diag_price_in_heap;
        } else if (diag_to_diag_price_ < best_diag_price_in_heap and diag_to_diag_price_ < diag_unassigned_price_) {
            best_diag_item_idx = *diag_assigned_to_diag_slice_.begin();
            best_diag_value = diag_edge_cost + diag_to_diag_price_;
            best_diag_price = diag_to_diag_price_;
        } else {
            best_diag_item_idx = *diag_unassigned_slice_.begin();
            best_diag_value = diag_edge_cost + diag_unassigned_price_;
            best_diag_price = diag_unassigned_price_;
        }

    }

    if ( best_diag_value < best_normal_item_value) {
        best_item_idx = best_diag_item_idx;
        best_item_price = best_diag_price;
        best_item_value = best_diag_value;
        second_best_item_value = best_normal_item_value;
    } else if (best_diag_value < second_best_normal_item_value) {
        best_item_idx = best_normal_item_idx;
        best_item_price = this->get_price(best_item_idx);
        best_item_value = best_normal_item_value;
        second_best_item_value = best_diag_value;
    } else {
        best_item_idx = best_normal_item_idx;
        best_item_price = this->get_price(best_item_idx);
        best_item_value = best_normal_item_value;
        second_best_item_value = second_best_normal_item_value;
    }

    IdxValPair<Real> result;

    result.first = best_item_idx;
    result.second = ( second_best_item_value - best_item_value ) + best_item_price + this->epsilon;

    //std::cout << "bidder_idx = " << bidder_idx << ", best_item_idx = " << best_item_idx << ", best_item_value = " << best_item_value << ", second_best_item_value = " << second_best_item_value << ", eps = " << this->epsilon << std::endl;
    assert( second_best_item_value >= best_item_value );
    //assert( best_item_price == this->get_price(best_item_idx) );
    assert(result.second >= best_item_price);
    sanity_check();

    return result;
}
/*
a_{ij} = d_{ij}
price_{ij} = a_{ij} + price_j
*/



//template<class Real_, class PointContainer_>
//std::vector<IdxValPairR> AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::increase_price_of_assigned_to_diag(WHAT)
//{
    //WHAT;
//}
//

template<class Real_, class PointContainer_>
Real_ AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::get_price(const size_t item_idx) const
{
    if (is_item_diagonal(item_idx)) {
        if (diag_assigned_to_diag_slice_.count(item_idx) == 1) {
            return diag_to_diag_price_;
        } else if (diag_unassigned_slice_.count(item_idx) == 1) {
            return diag_unassigned_price_;
        }
    }
    return this-> prices[item_idx];
}

template<class Real_, class PointContainer_>
void AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::set_price(const size_t item_idx,
                                          const Real new_price,
                                          const bool item_is_diagonal,
                                          const bool bidder_is_diagonal,
                                          const OwnerType old_owner_type)
{

    //std::cout << std::boolalpha << "enter set_price, item_idx = " << item_idx << ", new_price = " << new_price << ", old price = " << this->get_price(item_idx);
    //std::cout << ", item_is_diagonal = " << item_is_diagonal << ", bidder_is_diagonal  = " << bidder_is_diagonal << ", old_owner_type  = " << old_owner_type << std::endl;

    bool item_is_normal = not item_is_diagonal;
    bool bidder_is_normal = not bidder_is_diagonal;

    assert( new_price >= this->get_price(item_idx) );

    // update vector prices
    if (item_is_normal or bidder_is_normal) {
        this->prices[item_idx] = new_price;
    }

    // update kdtree_
    if (item_is_normal) {
        assert(0 <= item_idx and item_idx < kdtree__items_.size());
        assert(0 <= kdtree__items_[item_idx] and kdtree__items_[item_idx] < dnn_point_handles_.size());
        kdtree_->change_weight( dnn_point_handles_[kdtree__items_[item_idx]], new_price);
    }

    // update all_items_heap_
    if (bidder_is_diagonal and item_is_diagonal) {
        // remove slice (item is buried in diag_assigned_to_diag_slice_)
        assert(old_owner_type != OwnerType::k_diagonal);
        auto iter = all_items_heap__iters_[item_idx];
        assert(iter != all_items_heap_.end());
        all_items_heap_.erase(iter);
        all_items_heap__iters_[item_idx] = all_items_heap_.end();
    } else {
        auto iter = all_items_heap__iters_[item_idx];
        if (iter != all_items_heap_.end()) {
            // update existing element
            ItemSliceR x = *iter;
            x.set_loss( this->get_value_for_diagonal_bidder(item_idx) );
            all_items_heap_.erase(iter);
            auto ins_res = all_items_heap_.insert(x);
            all_items_heap__iters_[item_idx] = ins_res.first;
            assert(ins_res.second);
         } else {
            // insert new slice
            // for diagonal items value = price
            ItemSliceR x { item_idx, new_price };
            auto ins_res = all_items_heap_.insert(x);
            all_items_heap__iters_[item_idx] = ins_res.first;
            assert(ins_res.second);
         }
    }

    // update diag_items_heap_
    if (item_is_diagonal and bidder_is_normal) {
        // update existing element
        auto iter = diag_items_heap__iters_[item_idx];
        if (iter != diag_items_heap_.end()) {
            ItemSliceR x = *iter;
            x.set_loss( new_price );
            diag_items_heap_.erase(iter);
            auto ins_res = diag_items_heap_.insert(x);
            diag_items_heap__iters_[item_idx] = ins_res.first;
            assert(ins_res.second);
       } else {
            // insert new slice
            // for diagonal items value = price
            ItemSliceR x { item_idx, new_price };
            auto ins_res = diag_items_heap_.insert(x);
            diag_items_heap__iters_[item_idx] = ins_res.first;
            assert(ins_res.second);
        }
    } else if (bidder_is_diagonal and item_is_diagonal ) {
        // remove slice (item is buried in diag_assigned_to_diag_slice_)
        assert(old_owner_type != OwnerType::k_diagonal);
        auto iter = diag_items_heap__iters_[item_idx];
        assert(iter != diag_items_heap_.end());
        diag_items_heap_.erase(iter);
        diag_items_heap__iters_[item_idx] = diag_items_heap_.end();
    }

    // update diag_unassigned_price_
    if (item_is_diagonal and old_owner_type == OwnerType::k_none and diag_unassigned_slice_.empty()) {
        diag_unassigned_price_ = std::numeric_limits<Real>::max();
    }

}


template<class Real_, class PointContainer_>
bool AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::is_item_diagonal(const size_t item_idx) const
{
    return item_idx < this->num_diag_items_;
}


template<class Real_, class PointContainer_>
void AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::flush_assignment()
{
    //std::cout << "enter oracle->flush_assignment" << std::endl;
    sanity_check();

    for(const auto item_idx : diag_assigned_to_diag_slice_) {
        diag_unassigned_slice_.insert(item_idx);
    }
    diag_assigned_to_diag_slice_.clear();

    // common price of diag-diag items becomes price of diag-unassigned-slice
    // diag_to_diag_slice is now empty, set its price to max
    // so that get_optimal_bid works correctly
    diag_unassigned_price_ = diag_to_diag_price_;
    diag_to_diag_price_ = std::numeric_limits<Real>::max();

    normal_items_assigned_to_diag_.clear();

    sanity_check();
}


template<class Real_, class PointContainer_>
void AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::adjust_prices()
{
    return;

    throw std::runtime_error("not implemented");
    auto pr_begin = this->prices.begin();
    auto pr_end = this->prices.end();

    Real min_price = *(std::min_element(pr_begin, pr_end));

    for(auto& p : this->prices) {
        p -= min_price;
    }

    kdtree_->adjust_weights(min_price);
    diag_items_heap_.adjust_prices(min_price);
    all_items_heap_.adjust_prices(min_price);
}


template<class Real_, class PointContainer_>
AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::~AuctionOracleKDTreeSingleDiag()
{
    delete kdtree_;
}

template<class Real_, class PointContainer_>
void AuctionOracleKDTreeSingleDiag<Real_, PointContainer_>::sanity_check()
{
#ifdef DEBUG_AUCTION

    //std::cout << "ORACLE CURRENT STATE IN SANITY CHECK" <<  *this << std::endl;

    assert( diag_items_heap_.size() + diag_assigned_to_diag_slice_.size() + diag_unassigned_slice_.size() == num_diag_items_ );
    assert( diag_items_heap__iters_.size() == num_diag_items_ );
    for(size_t i = 0; i < num_diag_items_; ++i) {
        if (diag_items_heap__iters_.at(i) != diag_items_heap_.end()) {
            assert(diag_items_heap__iters_[i]->item_idx == i);
        }
    }

    assert( all_items_heap_.size() + diag_assigned_to_diag_slice_.size() + diag_unassigned_slice_.size() == this->num_items_ );
    assert( all_items_heap__iters_.size() == this->num_items_ );
    for(size_t i = 0; i < this->num_items_; ++i) {
        if (all_items_heap__iters_.at(i) != all_items_heap_.end()) {
            assert(all_items_heap__iters_[i]->item_idx == i);
        } else {
            assert( i < num_diag_items_ );
        }
    }

    for(size_t i = 0; i < num_diag_items_; ++i) {
        int is_in_assigned_slice = diag_assigned_to_diag_slice_.count(i);
        int is_in_unassigned_slice = diag_unassigned_slice_.count(i);
        int is_in_heap = diag_items_heap__iters_[i] != diag_items_heap_.end();
        assert( is_in_assigned_slice + is_in_unassigned_slice + is_in_heap == 1);
    }

    //assert((diag_assigned_to_diag_slice_.empty() and diag_to_diag_price_ == std::numeric_limits<Real>::max()) or (not diag_assigned_to_diag_slice_.empty() and diag_to_diag_price_ != std::numeric_limits<Real>::max()));
    //assert((diag_unassigned_slice_.empty() and diag_unassigned_price_ == std::numeric_limits<Real>::max()) or (not diag_unassigned_slice_.empty() and diag_unassigned_price_ != std::numeric_limits<Real>::max()));

    assert(diag_assigned_to_diag_slice_.empty() or diag_to_diag_price_ != std::numeric_limits<Real>::max());
    assert(diag_unassigned_slice_.empty() or diag_unassigned_price_ != std::numeric_limits<Real>::max());
#endif
}


} // ws
} // hera
#endif
