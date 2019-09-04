/*

Copyright (c) 2016, M. Kerber, D. Morozov, A. Nigmetov
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

#ifndef AUCTION_RUNNER_GS_H
#define AUCTION_RUNNER_GS_H

#include <memory>
#include <unordered_set>

#include "spdlog/spdlog.h"
#include "auction_oracle.h"

namespace hera {
namespace ws {

template<class RealType_ = double, class AuctionOracle_ = AuctionOracleKDTreeRestricted<RealType_>, class PointContainer_ = std::vector<DiagramPoint<RealType_>> >      // alternatively: AuctionOracleLazyHeap --- TODO
class AuctionRunnerGS {
public:
    using Real          = RealType_;
    using AuctionOracle = AuctionOracle_;
    using DgmPoint      = typename AuctionOracle::DiagramPointR;
    using IdxValPairR   = IdxValPair<Real>;
    using PointContainer = PointContainer_;


    AuctionRunnerGS(const PointContainer& A,
                    const PointContainer& B,
                    const AuctionParams<Real>& params,
                    const std::string& _log_filename_prefix = "");

    void set_epsilon(Real new_val) { assert(epsilon > 0.0); epsilon = new_val; };
    Real get_epsilon() const { return oracle.get_epsilon(); }
    Real get_wasserstein_cost();
    Real get_wasserstein_distance();
    Real get_relative_error() const { return relative_error; };
    void enable_logging(const char* log_filename, const size_t _max_unassigned_to_log);
//private:
    // private data
    PointContainer bidders, items;
    const size_t num_bidders;
    const size_t num_items;
    std::vector<IdxType> items_to_bidders;
    std::vector<IdxType> bidders_to_items;
    Real wasserstein_power;
    Real epsilon;
    Real delta;
    Real internal_p;
    Real initial_epsilon;
    Real epsilon_common_ratio; // next epsilon = current epsilon / epsilon_common_ratio
    const int max_num_phases; // maximal number of iterations of epsilon-scaling
    bool tolerate_max_iter_exceeded;
    Real weight_adj_const;
    Real wasserstein_cost;
    Real relative_error;
    int dimension;
    // to get the 2 best items
    AuctionOracle oracle;
    std::unordered_set<size_t> unassigned_bidders;
    // private methods
    void assign_item_to_bidder(const IdxType bidder_idx, const IdxType items_idx);
    void run_auction();
    void run_auction_phases(const int max_num_phases, const Real _initial_epsilon);
    void run_auction_phase();
    void flush_assignment();
    // return 0, if item_idx is invalid
    Real get_item_bidder_cost(const size_t item_idx, const size_t bidder_idx, const bool tolerate_invalid_idx = false) const;

    // for debug only
    void sanity_check();
    void print_debug();
    int count_unhappy();
    void print_matching();
    Real getDistanceToQthPowerInternal();
    int num_phase { 0 };
    int num_rounds { 0 };
    bool is_distance_computed {false};
#ifdef LOG_AUCTION
    bool log_auction { false };
    std::shared_ptr<spdlog::logger> console_logger;
    std::shared_ptr<spdlog::logger> plot_logger;
    std::unordered_set<size_t> unassigned_items;
    size_t max_unassigned_to_log { 0 };
    const char* logger_name = "auction_detailed_logger"; // the name in spdlog registry; filename is provided as parameter in enable_logging
    const Real total_items_persistence;
    const Real total_bidders_persistence;
    Real partial_cost;
    Real unassigned_bidders_persistence;
    Real unassigned_items_persistence;
#endif
};

} // ws
} // hera


#include "auction_runner_gs.hpp"

#endif
