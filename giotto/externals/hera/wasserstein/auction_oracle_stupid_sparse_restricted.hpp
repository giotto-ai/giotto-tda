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
#ifndef AUCTION_ORACLE_STUPID_SPARSE_HPP
#define AUCTION_ORACLE_STUPID_SPARSE_HPP

#include <assert.h>
#include <algorithm>
#include <functional>
#include <iterator>

#include "def_debug_ws.h"
#include "basic_defs_ws.h"
#include "auction_oracle_stupid_sparse_restricted.h"

#ifdef FOR_R_TDA
#undef DEBUG_AUCTION
#endif

namespace hera {
namespace ws {


// *****************************
// AuctionOracleStupidSparseRestricted
// *****************************


template <int k_max_nn, class Real_, class PointContainer_>
std::ostream& operator<<(std::ostream& output, const AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>& oracle)
{
    output << "Oracle " << &oracle << std::endl;
    output << fmt::format("       max_val_ = {0}, best_diagonal_items_computed_ = {1}, best_diagonal_item_value_ = {2}, second_best_diagonal_item_idx_ = {3}, second_best_diagonal_item_value_ = {4}\n",
                          oracle.max_val_,
                          oracle.best_diagonal_items_computed_,
                          oracle.best_diagonal_item_value_,
                          oracle.second_best_diagonal_item_idx_,
                          oracle.second_best_diagonal_item_value_);

    output << fmt::format("       prices = {0}\n",
                          format_container_to_log(oracle.prices));

    output << fmt::format("       diag_items_heap_ = {0}\n",
                          losses_heap_to_string(oracle.diag_items_heap_));


    output << fmt::format("       top_diag_indices_ = {0}\n",
                          format_container_to_log(oracle.top_diag_indices_));

    output << fmt::format("       top_diag_counter_ = {0}\n",
                          oracle.top_diag_counter_);

    output << fmt::format("       top_diag_lookup_ = {0}\n",
                          format_container_to_log(oracle.top_diag_lookup_));


    output << "end of oracle " << &oracle << std::endl;
    return output;
}


template<int k_max_nn, class Real_, class PointContainer_>
AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::AuctionOracleStupidSparseRestricted(const PointContainer_& _bidders,
                                                                   const PointContainer_& _items,
                                                                   const AuctionParams<Real_>& params) :
    AuctionOracleBase<Real_, PointContainer_>(_bidders, _items, params),
    admissible_items_(_bidders.size(), std::vector<size_t>()),
    heap_handles_indices_(_items.size(), k_invalid_index),
    top_diag_lookup_(_items.size(), k_invalid_index)
{
    // initialize admissible edges
    std::vector<size_t> kdtree_items_(_items.size(), k_invalid_index);
    std::vector<DnnPoint> dnn_points_;
    std::vector<DnnPoint*> dnn_point_handles_;
    size_t dnn_item_idx { 0 };
    size_t true_idx { 0 };
    dnn_points_.reserve(this->items.size());
    // store normal items in kd-tree
    for(const auto& g : this->items) {
        if (g.is_normal() ) {
            kdtree_items_[true_idx] = dnn_item_idx;
            // index of items is id of dnn-point
            DnnPoint p(true_idx);
            p[0] = g.getRealX();
            p[1] = g.getRealY();
            dnn_points_.push_back(p);
            assert(dnn_item_idx == dnn_points_.size() - 1);
            dnn_item_idx++;
        }
        true_idx++;
    }
    assert(dnn_points_.size() < _items.size() );
    for(size_t i = 0; i < dnn_points_.size(); ++i) {
        dnn_point_handles_.push_back(&dnn_points_[i]);
    }
    DnnTraits traits;
    traits.internal_p = params.internal_p;
    dnn::KDTree<DnnTraits> kdtree_(traits, dnn_point_handles_, params.wasserstein_power);

    // loop over normal bidders, find nearest neighbours
    size_t bidder_idx = 0;
    for(const auto& b : this->bidders) {
        if (b.is_normal()) {
            admissible_items_[bidder_idx].reserve(k_max_nn);
            DnnPoint bidder_dnn;
            bidder_dnn[0] = b.getRealX();
            bidder_dnn[1] = b.getRealY();
            auto nearest_neighbours = kdtree_.findK(bidder_dnn, k_max_nn);
            assert(nearest_neighbours.size() == k_max_nn);
            for(const auto& x : nearest_neighbours) {
                admissible_items_[bidder_idx].push_back(x.p->id());
            }
        }
        bidder_idx++;
    }

    size_t handle_idx {0};
    for(size_t item_idx = 0; item_idx < _items.size(); ++item_idx) {
        if (this->items[item_idx].is_diagonal()) {
            heap_handles_indices_[item_idx] = handle_idx++;
            diag_heap_handles_.push_back(diag_items_heap_.push(std::make_pair(item_idx, 0.0)));
        }
    }
    max_val_ = 3*getFurthestDistance3Approx<>(_bidders, _items, params.internal_p);
    max_val_ = std::pow(max_val_, params.wasserstein_power);

    console_logger = spdlog::get("console");
    if (not console_logger) {
        console_logger = spdlog::stdout_logger_st("console");
    }
    console_logger->set_pattern("[%H:%M:%S.%e] %v");
    console_logger->info("Stupid sparse oracle ctor done, k = {0}", k_max_nn);
}


template<int k_max_nn, class Real_, class PointContainer_>
bool AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::is_in_top_diag_indices(const size_t item_idx) const
{
    return top_diag_lookup_[item_idx] != k_invalid_index;
}


template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::add_top_diag_index(const size_t item_idx)
{
    assert(find(top_diag_indices_.begin(), top_diag_indices_.end(), item_idx) == top_diag_indices_.end());
    assert(this->items[item_idx].is_diagonal());

    top_diag_indices_.push_back(item_idx);
    top_diag_lookup_[item_idx] = top_diag_indices_.size() - 1;
}

template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::remove_top_diag_index(const size_t item_idx)
{
    if (top_diag_indices_.size() > 1) {
        // remove item_idx from top_diag_indices after swapping
        // it with the last element, update index lookup appropriately
        auto old_index = top_diag_lookup_[item_idx];
        auto end_element = top_diag_indices_.back();
        std::swap(top_diag_indices_[old_index], top_diag_indices_.back());
        top_diag_lookup_[end_element] = old_index;
    }

    top_diag_indices_.pop_back();
    top_diag_lookup_[item_idx] = k_invalid_index;
    if (top_diag_indices_.size() < 2) {
        recompute_second_best_diag();
    }
    best_diagonal_items_computed_ = not top_diag_indices_.empty();
    reset_top_diag_counter();
}


template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::increment_top_diag_counter()
{
    assert(top_diag_counter_ >= 0 and top_diag_counter_ < top_diag_indices_.size());

    ++top_diag_counter_;
    if (top_diag_counter_ >= top_diag_indices_.size()) {
        top_diag_counter_ -= top_diag_indices_.size();
    }

    assert(top_diag_counter_ >= 0 and top_diag_counter_ < top_diag_indices_.size());
}


template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::reset_top_diag_counter()
{
    top_diag_counter_ = 0;
}

template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::recompute_top_diag_items(bool hard)
{
    console_logger->debug("Enter recompute_top_diag_items, hard = {0}", hard);
    assert(hard or top_diag_indices_.empty());

    if (hard) {
        std::fill(top_diag_lookup_.begin(), top_diag_lookup_.end(), k_invalid_index);
        top_diag_indices_.clear();
    }

    auto top_diag_iter = diag_items_heap_.ordered_begin();
    best_diagonal_item_value_ = top_diag_iter->second;
    add_top_diag_index(top_diag_iter->first);

    ++top_diag_iter;

    // traverse the heap while we see the same value
    while(top_diag_iter != diag_items_heap_.ordered_end()) {
        if ( top_diag_iter->second != best_diagonal_item_value_) {
            break;
        } else {
            add_top_diag_index(top_diag_iter->first);
        }
        ++top_diag_iter;
    }

    recompute_second_best_diag();

    best_diagonal_items_computed_ = true;
    reset_top_diag_counter();
    console_logger->debug("Exit recompute_top_diag_items, hard = {0}", hard);
}

template<int k_max_nn, class Real_, class PointContainer_>
typename AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::DebugOptimalBidR
AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::get_optimal_bid_debug(IdxType bidder_idx) const
{
    DebugOptimalBidR result;
    throw std::runtime_error("Not implemented");
    return result;
}


template<int k_max_nn, class Real_, class PointContainer_>
IdxValPair<Real_> AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::get_optimal_bid(IdxType bidder_idx)
{
    auto bidder = this->bidders[bidder_idx];

    // corresponding point is always considered as a candidate
    // if bidder is a diagonal point, proj_item is a normal point,
    // and vice versa.

    size_t best_item_idx { k_invalid_index };
    size_t second_best_item_idx { k_invalid_index };
    size_t best_diagonal_item_idx { k_invalid_index };
    Real best_item_value;
    Real second_best_item_value;


    size_t proj_item_idx = bidder_idx;
    assert( 0 <= proj_item_idx and proj_item_idx < this->items.size() );
    assert(this->items[proj_item_idx].type != bidder.type);
    Real proj_item_value = this->get_value_for_bidder(bidder_idx, proj_item_idx);

    if (bidder.is_diagonal()) {
        // for diagonal bidder the only normal point has already been added
        // the other 2 candidates are diagonal items only, get from the heap
        // with prices

        if (not best_diagonal_items_computed_) {
            recompute_top_diag_items();
        }

        best_diagonal_item_idx = top_diag_indices_[top_diag_counter_];
        increment_top_diag_counter();

        if ( proj_item_value < best_diagonal_item_value_) {
            best_item_idx = proj_item_idx;
            best_item_value = proj_item_value;
            second_best_item_value = best_diagonal_item_value_;
            second_best_item_idx = best_diagonal_item_idx;
        } else if (proj_item_value < second_best_diagonal_item_value_) {
            best_item_idx = best_diagonal_item_idx;
            best_item_value = best_diagonal_item_value_;
            second_best_item_value = proj_item_value;
            second_best_item_idx = proj_item_idx;
        } else {
            best_item_idx = best_diagonal_item_idx;
            best_item_value = best_diagonal_item_value_;
            second_best_item_value = second_best_diagonal_item_value_;
            second_best_item_idx = second_best_diagonal_item_idx_;
        }
    } else {

        size_t best_normal_item_idx { k_invalid_index };
        size_t second_best_normal_item_idx { k_invalid_index };
        Real best_normal_item_value { std::numeric_limits<Real>::max() };
        Real second_best_normal_item_value { std::numeric_limits<Real>::max() };

        // find best item
        for(const auto curr_item_idx : admissible_items_[bidder_idx]) {
            auto curr_item_value = this->get_value_for_bidder(bidder_idx, curr_item_idx);
            if (curr_item_value < best_normal_item_value) {
                best_normal_item_idx = curr_item_idx;
                best_normal_item_value = curr_item_value;
            }
        }

        // find second-best item
        for(const auto curr_item_idx : admissible_items_[bidder_idx]) {
            if (curr_item_idx == best_normal_item_idx) {
                continue;
            }
            auto curr_item_value = this->get_value_for_bidder(bidder_idx, curr_item_idx);
            if (curr_item_value < second_best_normal_item_value) {
                second_best_normal_item_idx = curr_item_idx;
                second_best_normal_item_value = curr_item_value;
            }
        }

        if ( proj_item_value < best_normal_item_value) {
            best_item_idx = proj_item_idx;
            increment_top_diag_counter();
            best_item_value = proj_item_value;
            second_best_item_value = best_normal_item_value;
        } else if (proj_item_value < second_best_normal_item_value) {
            best_item_idx = best_normal_item_idx;
            best_item_value = best_normal_item_value;
            second_best_item_value = proj_item_value;
        } else {
            best_item_idx = best_normal_item_idx;
            best_item_value = best_normal_item_value;
            second_best_item_value = second_best_normal_item_value;
        }
    }

    IdxValPair<Real> result;

    assert( second_best_item_value >= best_item_value );

    result.first = best_item_idx;
    result.second = ( second_best_item_value - best_item_value ) + this->prices[best_item_idx] + this->epsilon;

    return result;
}

template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::recompute_second_best_diag()
{

    console_logger->debug("Enter recompute_second_best_diag");

    if (top_diag_indices_.size() > 1) {
        second_best_diagonal_item_value_ = best_diagonal_item_value_;
        second_best_diagonal_item_idx_ = top_diag_indices_[0];
    } else {
        if (diag_items_heap_.size() == 1) {
            second_best_diagonal_item_value_ == std::numeric_limits<Real>::max();
            second_best_diagonal_item_idx_ = k_invalid_index;
        } else {
            auto diag_iter = diag_items_heap_.ordered_begin();
            ++diag_iter;
            second_best_diagonal_item_value_ = diag_iter->second;
            second_best_diagonal_item_idx_ = diag_iter->first;
        }
    }

    console_logger->debug("Exit recompute_second_best_diag, second_best_diagonal_item_value_ = {0}, second_best_diagonal_item_idx_ = {1}", second_best_diagonal_item_value_, second_best_diagonal_item_idx_);
}


template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::set_price(IdxType item_idx,
                                                    Real new_price,
                                                    const bool update_diag)
{

    console_logger->debug("Enter set_price, item_idx = {0}, new_price = {1}, old price = {2}, update_diag = {3}", item_idx, new_price, this->prices[item_idx], update_diag);

    assert(this->prices.size() == this->items.size());
    assert( 0 < diag_heap_handles_.size() and diag_heap_handles_.size() <= this->items.size());
	// adjust_prices decreases prices,
    // also this variable must be true in reverse phases of FR-auction
	bool item_goes_down = new_price > this->prices[item_idx];

    this->prices[item_idx] = new_price;
    if ( this->items[item_idx].is_diagonal() ) {
        assert(diag_heap_handles_.size() > heap_handles_indices_.at(item_idx));
		if (item_goes_down) {
			diag_items_heap_.decrease(diag_heap_handles_[heap_handles_indices_[item_idx]], std::make_pair(item_idx, new_price));
		} else {
			diag_items_heap_.increase(diag_heap_handles_[heap_handles_indices_[item_idx]], std::make_pair(item_idx, new_price));
		}
        if (update_diag) {
            // Update top_diag_indices_ only if necessary:
            // normal bidders take their projections, which might not be on top
            // also, set_price is called by adjust_prices, and we may have already
            // removed the item from top_diag
            if (is_in_top_diag_indices(item_idx)) {
                remove_top_diag_index(item_idx);
            }

            if (item_idx == second_best_diagonal_item_idx_) {
                recompute_second_best_diag();
            }
        }
    }

    console_logger->debug("Exit set_price, item_idx = {0}, new_price = {1}", item_idx, new_price);
}


template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::adjust_prices(Real delta)
{
    //console_logger->debug("Enter adjust_prices, delta = {0}", delta);
    //std::cerr << *this << std::endl;

    if (delta == 0.0)
        return;

    for(auto& p : this->prices) {
        p -= delta;
    }

    bool price_goes_up = delta < 0;

	for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        if (this->items[item_idx].is_diagonal()) {
            auto new_price = this->prices[item_idx];
            if (price_goes_up) {
                diag_items_heap_.decrease(diag_heap_handles_[heap_handles_indices_[item_idx]], std::make_pair(item_idx, new_price));
            } else {
                diag_items_heap_.increase(diag_heap_handles_[heap_handles_indices_[item_idx]], std::make_pair(item_idx, new_price));
            }
        }
	}
    best_diagonal_item_value_ -= delta;
    second_best_diagonal_item_value_ -= delta;

    //std::cerr << *this << std::endl;
    //console_logger->debug("Exit adjust_prices, delta = {0}", delta);
}

template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::adjust_prices()
{
    auto pr_begin = this->prices.begin();
    auto pr_end = this->prices.end();
    Real min_price = *(std::min_element(pr_begin, pr_end));
    adjust_prices(min_price);
}

template<int k_max_nn, class Real_, class PointContainer_>
size_t AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::get_heap_top_size() const
{
    return top_diag_indices_.size();
}

template<int k_max_nn, class Real_, class PointContainer_>
std::pair<Real_, Real_> AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::get_minmax_price() const
{
    auto r = std::minmax_element(this->prices.begin(), this->prices.end());
    return std::make_pair(*r.first, *r.second);
}

template<int k_max_nn, class Real_, class PointContainer_>
void AuctionOracleStupidSparseRestricted<k_max_nn, Real_, PointContainer_>::sanity_check()
{
#ifdef DEBUG_STUPID_SPARSE_RESTR_ORACLE

    assert(admissible_items_.size() == this->bidders.size());

    for(size_t bidder_idx = 0; bidder_idx < this->bidders.size(); ++bidder_idx) {
        if (this->bidders[bidder_idx].is_normal()) {
            assert(admissible_items_[bidder_idx].size() == k_max_nn);
        } else {
            assert(admissible_items_[bidder_idx].size() == 0);
        }
    }

    if (best_diagonal_items_computed_) {
        std::vector<Real> diag_items_price_vec;
        diag_items_price_vec.reserve(this->items.size());

        for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
            if (this->items.at(item_idx).is_diagonal()) {
                diag_items_price_vec.push_back(this->prices.at(item_idx));
            } else {
                diag_items_price_vec.push_back(std::numeric_limits<Real>::max());
            }
        }

        auto best_iter =  std::min_element(diag_items_price_vec.begin(), diag_items_price_vec.end());
        assert(best_iter != diag_items_price_vec.end());
        Real true_best_diag_value = *best_iter;
        size_t true_best_diag_idx = best_iter - diag_items_price_vec.begin();
        assert(true_best_diag_value != std::numeric_limits<Real>::max());

        Real true_second_best_diag_value = std::numeric_limits<Real>::max();
        size_t true_second_best_diag_idx = k_invalid_index;
        for(size_t item_idx = 0; item_idx < diag_items_price_vec.size(); ++item_idx) {
            if (this->items.at(item_idx).is_normal()) {
                assert(top_diag_lookup_.at(item_idx) == k_invalid_index);
                continue;
            }

            auto i_iter = std::find(top_diag_indices_.begin(), top_diag_indices_.end(), item_idx);
            if (diag_items_price_vec.at(item_idx) == true_best_diag_value) {
                assert(i_iter != top_diag_indices_.end());
                assert(top_diag_lookup_.at(item_idx) == i_iter - top_diag_indices_.begin());
            } else {
                assert(top_diag_lookup_.at(item_idx) == k_invalid_index);
                assert(i_iter == top_diag_indices_.end());
            }

            if (item_idx == true_best_diag_idx) {
                continue;
            }
            if (diag_items_price_vec.at(item_idx) < true_second_best_diag_value) {
                true_second_best_diag_value = diag_items_price_vec.at(item_idx);
                true_second_best_diag_idx = item_idx;
            }
        }

        if (true_best_diag_value != best_diagonal_item_value_) {
            console_logger->debug("best_diagonal_item_value_ = {0}, true value = {1}", best_diagonal_item_value_, true_best_diag_value);
            std::cerr << *this;
            //console_logger->debug("{0}", *this);
        }

        assert(true_best_diag_value == best_diagonal_item_value_);

        assert(true_second_best_diag_idx != k_invalid_index);

        if (true_second_best_diag_value != second_best_diagonal_item_value_) {
            console_logger->debug("second_best_diagonal_item_value_ = {0}, true value = {1}", second_best_diagonal_item_value_, true_second_best_diag_value);
            //console_logger->debug("{0}", *this);
        }

        assert(true_second_best_diag_value == second_best_diagonal_item_value_);
    }
#endif
}


} // ws
} // hera

#endif
