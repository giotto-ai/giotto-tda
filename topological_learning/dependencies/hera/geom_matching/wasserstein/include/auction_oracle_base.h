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

#ifndef AUCTION_ORACLE_BASE_H
#define AUCTION_ORACLE_BASE_H

#include <map>
#include <memory>
#include <set>
#include <list>

#include "basic_defs_ws.h"

namespace hera {
namespace ws {


template <class Real>
struct DebugOptimalBid {
    DebugOptimalBid() : best_item_idx(k_invalid_index), best_item_value(-666.666), second_best_item_idx(k_invalid_index), second_best_item_value(-666.666) {};
    IdxType best_item_idx;
    Real best_item_value;
    IdxType second_best_item_idx;
    Real second_best_item_value;
};

template <class Real = double, class PointContainer_ = std::vector<DiagramPoint<Real>>>
struct AuctionOracleBase {
    AuctionOracleBase(const PointContainer_& _bidders, const PointContainer_& _items, const AuctionParams<Real>& params);
    ~AuctionOracleBase() {}
    Real get_epsilon() const { return epsilon; };
    void set_epsilon(Real new_epsilon) { assert(new_epsilon >= 0.0); epsilon = new_epsilon; };
    const std::vector<Real>& get_prices() const { return prices; }
    virtual Real get_price(const size_t item_idx) const { return prices[item_idx]; } // TODO make virtual?
//protected:
    const PointContainer_& bidders;
    const PointContainer_& items;
    const size_t num_bidders_;
    const size_t num_items_;
    std::vector<Real> prices;
    const Real wasserstein_power;
    Real epsilon;
    const Real internal_p;
    unsigned int dim;  // used only in pure geometric version, not for persistence diagrams
    Real get_value_for_bidder(size_t bidder_idx, size_t item_idx) const;
    Real get_value_for_diagonal_bidder(size_t item_idx) const;
    Real get_cost_for_diagonal_bidder(size_t item_idx) const;
};


template<class Real>
std::ostream& operator<< (std::ostream& output, const DebugOptimalBid<Real>& db);

} // ws
} // hera


#include "auction_oracle_base.hpp"

#endif
