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

#ifndef AUCTION_ORACLE_KDTREE_RESTRICTED_H
#define AUCTION_ORACLE_KDTREE_RESTRICTED_H


//#define USE_BOOST_HEAP

#include <map>
#include <memory>
#include <set>


#include "spdlog/spdlog.h"
#include "basic_defs_ws.h"
#include "diagonal_heap.h"
#include "auction_oracle_base.h"
#include "dnn/geometry/euclidean-fixed.h"
#include "dnn/local/kd-tree.h"

namespace hera {
namespace ws {

template <class Real_ = double, class PointContainer_ = std::vector<DiagramPoint<Real_>>>
struct AuctionOracleKDTreeRestricted : AuctionOracleBase<Real_, PointContainer_> {

    using PointContainer    = PointContainer_;
    using Real              = Real_;

    using LossesHeapR       = typename ws::LossesHeapOld<Real>;
    using LossesHeapRHandle = typename ws::LossesHeapOld<Real>::handle_type;
    using DiagramPointR     = typename ws::DiagramPoint<Real>;
    using DebugOptimalBidR  = typename ws::DebugOptimalBid<Real>;

    using DnnPoint          = dnn::Point<2, Real>;
    using DnnTraits         = dnn::PointTraits<DnnPoint>;

    AuctionOracleKDTreeRestricted(const PointContainer& bidders, const PointContainer& items, const AuctionParams<Real>& params);
    ~AuctionOracleKDTreeRestricted();
    // data members
    // temporarily make everything public
    Real max_val_;
    Real weight_adj_const_;
    dnn::KDTree<DnnTraits>* kdtree_;
    std::vector<DnnPoint> dnn_points_;
    std::vector<DnnPoint*> dnn_point_handles_;
    LossesHeapR diag_items_heap_;
    std::vector<LossesHeapRHandle> diag_heap_handles_;
    std::vector<size_t> heap_handles_indices_;
    std::vector<size_t> kdtree_items_;
    std::vector<size_t> top_diag_indices_;
    std::vector<size_t> top_diag_lookup_;
    size_t top_diag_counter_ { 0 };
    bool best_diagonal_items_computed_ { false };
    Real best_diagonal_item_value_;
    size_t second_best_diagonal_item_idx_ { k_invalid_index };
    Real second_best_diagonal_item_value_ { std::numeric_limits<Real>::max() };


    // methods
    void set_price(const IdxType items_idx, const Real new_price, const bool update_diag = true);
    IdxValPair<Real> get_optimal_bid(const IdxType bidder_idx);
    void adjust_prices();
    void adjust_prices(const Real delta);

    // debug routines
    DebugOptimalBidR get_optimal_bid_debug(IdxType bidder_idx) const;
    void sanity_check();


    // heap top vector
    size_t get_heap_top_size() const;
    void recompute_top_diag_items(bool hard = false);
    void recompute_second_best_diag();
    void reset_top_diag_counter();
    void increment_top_diag_counter();
    void add_top_diag_index(const size_t item_idx);
    void remove_top_diag_index(const size_t item_idx);
    bool is_in_top_diag_indices(const size_t item_idx) const;

    std::shared_ptr<spdlog::logger> console_logger;

    std::pair<Real, Real> get_minmax_price() const;

};

template<class Real>
std::ostream& operator<< (std::ostream& output, const DebugOptimalBid<Real>& db);

} // ws
} // hera


#include "auction_oracle_kdtree_restricted.hpp"

#endif
