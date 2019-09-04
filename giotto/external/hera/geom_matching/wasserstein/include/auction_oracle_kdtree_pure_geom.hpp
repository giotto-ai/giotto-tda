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
#ifndef AUCTION_ORACLE_KDTREE_PURE_GEOM_HPP
#define AUCTION_ORACLE_KDTREE_PURE_GEOM_HPP

#include <assert.h>
#include <algorithm>
#include <functional>
#include <iterator>

#include "def_debug_ws.h"
#include "auction_oracle_kdtree_restricted.h"


#ifdef FOR_R_TDA
#undef DEBUG_AUCTION
#endif

namespace hera {
namespace ws {


// *****************************
// AuctionOracleKDTreePureGeom
// *****************************



template <class Real_, class PointContainer_>
std::ostream& operator<<(std::ostream& output, const AuctionOracleKDTreePureGeom<Real_, PointContainer_>& oracle)
{
    output << "Oracle " << &oracle << std::endl;
    output << fmt::format("       max_val_ = {0}\n",
                          oracle.max_val_);

    output << fmt::format("       prices = {0}\n",
                          format_container_to_log(oracle.prices));

    output << "end of oracle " << &oracle << std::endl;
    return output;
}


template<class Real_, class PointContainer_>
AuctionOracleKDTreePureGeom<Real_, PointContainer_>::AuctionOracleKDTreePureGeom(const PointContainer_& _bidders,
                                                                                 const PointContainer_& _items,
                                                                                 const AuctionParams<Real_>& params) :
    AuctionOracleBase<Real_, PointContainer_>(_bidders, _items, params),
    traits(params.dim)
{

    traits.internal_p = params.internal_p;

    std::vector<PointHandleR> item_handles(this->num_items_);
    for(size_t i = 0; i < this->num_items_; ++i) {
        item_handles[i] = traits.handle(this->items[i]);
    }

    //kdtree_ = std::unique_ptr<KDTreeR>(new KDTreeR(traits,
    //        this->items | ba::transformed([this](const DiagramPointR& p) { return traits.handle(p); }),
    //        params.wasserstein_power));

    kdtree_ = std::unique_ptr<KDTreeR>(new KDTreeR(traits, item_handles, params.wasserstein_power));


    max_val_ = 3*getFurthestDistance3Approx_pg(this->bidders, this->items, params.internal_p, params.dim);
    max_val_ = std::pow(max_val_, params.wasserstein_power);
    weight_adj_const_ = max_val_;

    console_logger = spdlog::get("console");
    if (not console_logger) {
        console_logger = spdlog::stdout_logger_st("console");
    }
    console_logger->set_pattern("[%H:%M:%S.%e] %v");
    console_logger->debug("KDTree Restricted oracle ctor done");
}


template<class Real_, class PointContainer_>
typename AuctionOracleKDTreePureGeom<Real_, PointContainer_>::DebugOptimalBidR
AuctionOracleKDTreePureGeom<Real_, PointContainer_>::get_optimal_bid_debug(IdxType bidder_idx) const
{
    auto bidder = this->bidders[bidder_idx];

    size_t best_item_idx = k_invalid_index;
    size_t second_best_item_idx = k_invalid_index;
    Real best_item_value = std::numeric_limits<Real>::max();
    Real second_best_item_value = std::numeric_limits<Real>::max();

    for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        auto item = this->items[item_idx];
        auto item_value = std::pow(traits.distance(bidder, item), this->wasserstein_power) + this->prices[item_idx];
        if (item_value < best_item_value) {
            best_item_value = item_value;
            best_item_idx = item_idx;
        }
    }

    assert(best_item_idx != k_invalid_index);

    for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        auto item = this->items[item_idx];
        if (item_idx == best_item_idx)
            continue;

        auto item_value = std::pow(traits.distance(bidder, item), this->wasserstein_power) + this->prices[item_idx];
        if (item_value < second_best_item_value) {
            second_best_item_value = item_value;
            second_best_item_idx = item_idx;
        }
    }

    assert(second_best_item_idx != k_invalid_index);
    assert(second_best_item_value >= best_item_value);

    DebugOptimalBidR result;

    result.best_item_idx = best_item_idx;
    result.best_item_value = best_item_value;
    result.second_best_item_idx = second_best_item_idx;
    result.second_best_item_value = second_best_item_value;

    return result;
}


template<class Real_, class PointContainer_>
IdxValPair<Real_> AuctionOracleKDTreePureGeom<Real_, PointContainer_>::get_optimal_bid(IdxType bidder_idx)
{
    auto two_best_items = kdtree_->findK(this->bidders[bidder_idx], 2);
    size_t best_item_idx = traits.id(two_best_items[0].p);
    Real best_item_value = two_best_items[0].d;
    Real second_best_item_value = two_best_items[1].d;

    IdxValPair<Real> result;

    assert( second_best_item_value >= best_item_value );

    result.first = best_item_idx;
    result.second = ( second_best_item_value - best_item_value ) + this->prices[best_item_idx] + this->epsilon;

#ifdef DEBUG_KDTREE_RESTR_ORACLE
    auto bid_debug = get_optimal_bid_debug(bidder_idx);
    assert(fabs(bid_debug.best_item_value - best_item_value) < 0.000000001);
    assert(fabs(bid_debug.second_best_item_value - second_best_item_value) < 0.000000001);
#endif

    return result;
}

/*
a_{ij} = d_{ij}
value_{ij} = a_{ij} + price_j
*/

template<class Real_, class PointContainer_>
void AuctionOracleKDTreePureGeom<Real_, PointContainer_>::set_price(IdxType item_idx,
                                                    Real new_price)
{

    console_logger->debug("Enter set_price, item_idx = {0}, new_price = {1}, old price = {2}", item_idx, new_price, this->prices[item_idx]);

    assert(this->prices.size() == this->items.size());
	// adjust_prices decreases prices,
    // also this variable must be true in reverse phases of FR-auction

    this->prices[item_idx] = new_price;
    kdtree_->change_weight( traits.handle(this->items[item_idx]), new_price);

    console_logger->debug("Exit set_price, item_idx = {0}, new_price = {1}", item_idx, new_price);
}


template<class Real_, class PointContainer_>
void AuctionOracleKDTreePureGeom<Real_, PointContainer_>::adjust_prices(Real delta)
{
    //console_logger->debug("Enter adjust_prices, delta = {0}", delta);
    //std::cerr << *this << std::endl;

    if (delta == 0.0)
        return;

    for(auto& p : this->prices) {
        p -= delta;
    }

    kdtree_->adjust_weights(delta);

    //std::cerr << *this << std::endl;
    //console_logger->debug("Exit adjust_prices, delta = {0}", delta);
}

template<class Real_, class PointContainer_>
void AuctionOracleKDTreePureGeom<Real_, PointContainer_>::adjust_prices()
{
    auto pr_begin = this->prices.begin();
    auto pr_end = this->prices.end();
    Real min_price = *(std::min_element(pr_begin, pr_end));
    adjust_prices(min_price);
}

template<class Real_, class PointContainer_>
std::pair<Real_, Real_> AuctionOracleKDTreePureGeom<Real_, PointContainer_>::get_minmax_price() const
{
    auto r = std::minmax_element(this->prices.begin(), this->prices.end());
    return std::make_pair(*r.first, *r.second);
}

template<class Real_, class PointContainer_>
AuctionOracleKDTreePureGeom<Real_, PointContainer_>::~AuctionOracleKDTreePureGeom()
{
}

template<class Real_, class PointContainer_>
void AuctionOracleKDTreePureGeom<Real_, PointContainer_>::sanity_check()
{
}


} // ws
} // hera

#endif
