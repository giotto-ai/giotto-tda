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

#ifndef AUCTION_ORACLE_BASE_HPP
#define AUCTION_ORACLE_BASE_HPP

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

template<class Real, class PointContainer>
AuctionOracleBase<Real, PointContainer>::AuctionOracleBase(const PointContainer& _bidders,
                                           const PointContainer& _items,
                                           const AuctionParams<Real>& params) :
    bidders(_bidders),
    items(_items),
    num_bidders_(_bidders.size()),
    num_items_(_items.size()),
    prices(items.size(), Real(0.0)),
    wasserstein_power(params.wasserstein_power),
    internal_p(params.internal_p),
    dim(params.dim)
{
    assert(bidders.size() == items.size() );
}


template<class Real, class PointContainer>
Real AuctionOracleBase<Real, PointContainer>::get_value_for_bidder(size_t bidder_idx, size_t item_idx) const
{
    return std::pow(dist_lp<Real>(bidders[bidder_idx], items[item_idx], internal_p, dim), wasserstein_power) + get_price(item_idx);
}

template<class Real, class PointContainer>
Real AuctionOracleBase<Real, PointContainer>::get_value_for_diagonal_bidder(size_t item_idx) const
{
    return get_cost_for_diagonal_bidder(item_idx) + get_price(item_idx);
}

template<class Real, class PointContainer>
Real AuctionOracleBase<Real, PointContainer>::get_cost_for_diagonal_bidder(size_t item_idx) const
{
    return std::pow(items[item_idx].persistence_lp(internal_p), wasserstein_power);
}



template<class Real>
std::ostream& operator<< (std::ostream& output, const DebugOptimalBid<Real>& db)
{
    output << "best_item_value = " << db.best_item_value;
    output << "; best_item_idx = " << db.best_item_idx;
    output << "; second_best_item_value = " << db.second_best_item_value;
    output << "; second_best_item_idx = " << db.second_best_item_idx;
    return output;
}

} // ws
} // hera

#endif
