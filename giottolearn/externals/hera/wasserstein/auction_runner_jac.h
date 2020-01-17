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

#ifndef HERA_AUCTION_RUNNER_JAC_H
#define HERA_AUCTION_RUNNER_JAC_H

#ifdef WASSERSTEIN_PURE_GEOM
#undef LOG_AUCTION
#undef ORDERED_BY_PERSISTENCE
#endif

//#define ORDERED_BY_PERSISTENCE

#include <unordered_set>

#include "auction_oracle.h"

namespace hera {
namespace ws {

// the two parameters that you can tweak in auction algorithm are:
// 1. epsilon_common_ratio
// 2. max_num_phases

template<class RealType_ = double, class AuctionOracle_ = AuctionOracleKDTreeRestricted<RealType_>, class PointContainer_ = std::vector<DiagramPoint<RealType_>> >      // alternatively: AuctionOracleLazyHeap --- TODO
class AuctionRunnerJac {
public:

    using Real          = RealType_;
    using AuctionOracle = AuctionOracle_;
    using DgmPoint      = typename AuctionOracle::DiagramPointR;
    using IdxValPairR   = IdxValPair<Real>;
    using PointContainer = PointContainer_;

    const Real k_lowest_bid_value = -1; // all bid values must be positive


    AuctionRunnerJac(const PointContainer& A,
                     const PointContainer& B,
                     const AuctionParams<Real>& params,
                     const std::string& _log_filename_prefix = "");

    void set_epsilon(Real new_val);
    Real get_epsilon() const { return epsilon; }
    void run_auction();
    template<class Range>
    void run_bidding_step(const Range& r);
    bool is_done() const;
    void decrease_epsilon();
    Real get_wasserstein_distance();
    Real get_wasserstein_cost();
    Real get_relative_error(const bool debug_output = false) const;
//private:
    // private data
    PointContainer bidders;
    PointContainer items;
    const size_t num_bidders;
    const size_t num_items;
    std::vector<IdxType> items_to_bidders;
    std::vector<IdxType> bidders_to_items;
    Real wasserstein_power;
    Real epsilon;
    Real delta;
    Real internal_p;
    Real initial_epsilon;
    const Real epsilon_common_ratio; // next epsilon = current epsilon / epsilon_common_ratio
    const int max_num_phases; // maximal number of phases of epsilon-scaling
    Real weight_adj_const;
    Real wasserstein_cost;
    std::vector<IdxValPairR> bid_table;
    // to get the 2 best items
    AuctionOracle oracle;
    std::unordered_set<size_t> unassigned_bidders;
    std::unordered_set<size_t> items_with_bids;
    // to imitate Gauss-Seidel
    const size_t max_bids_per_round;
    Real partial_cost { 0.0 };
    bool is_distance_computed { false };
    int num_rounds { 0 };
    int num_phase { 0 };
    int dimension;

    size_t unassigned_threshold; // for experiments

#ifndef WASSERSTEIN_PURE_GEOM
    std::unordered_set<size_t> unassigned_normal_bidders;
    std::unordered_set<size_t> unassigned_diag_bidders;
    bool diag_first {true};
    size_t batch_size { 1000 };
#ifdef ORDERED_BY_PERSISTENCE
    // to process unassigned by persistence
    using RealIdxPair = std::pair<Real, size_t>;
    std::set<RealIdxPair, std::greater<RealIdxPair>> unassigned_normal_bidders_by_persistence;
#endif

    // to stop earlier in the last phase
    const Real total_items_persistence;
    const Real total_bidders_persistence;
    Real unassigned_bidders_persistence;
    Real unassigned_items_persistence;
    Real gamma_threshold;


    size_t num_diag_items { 0 };
    size_t num_normal_items { 0 };
    size_t num_diag_bidders { 0 };
    size_t num_normal_bidders { 0 };


#endif



    // private methods
    void assign_item_to_bidder(const IdxType bidder_idx, const IdxType items_idx);
    void assign_to_best_bidder(const IdxType items_idx);
    void clear_bid_table();
    void run_auction_phases(const int max_num_phases, const Real _initial_epsilon);
    void run_auction_phase();
    void submit_bid(IdxType bidder_idx, const IdxValPairR& items_bid_value_pair);
    void flush_assignment();
    Real get_item_bidder_cost(const size_t item_idx, const size_t bidder_idx) const;
#ifndef WASSERSTEIN_PURE_GEOM
    Real get_cost_to_diagonal(const DgmPoint& pt) const;
    Real get_gamma() const;
#endif
    bool continue_auction_phase() const;

    void add_unassigned_bidder(const size_t bidder_idx);
    void remove_unassigned_bidder(const size_t bidder_idx);
    void remove_unassigned_item(const size_t item_idx);

#ifndef WASSERSTEIN_PURE_GEOM
    bool is_item_diagonal(const size_t item_idx) const { return item_idx < num_diag_items; }
    bool is_item_normal(const size_t item_idx) const { return not is_item_diagonal(item_idx); }
    bool is_bidder_diagonal(const size_t bidder_idx) const { return bidder_idx >= num_normal_bidders; }
    bool is_bidder_normal(const size_t bidder_idx) const { return not is_bidder_diagonal(bidder_idx); }
#endif



    // for debug only
    void sanity_check();
    void print_debug();
    void print_matching();

    std::string log_filename_prefix;
    const Real k_max_relative_error = 2.0; // if relative error cannot be estimated or is too large, use this value

#ifdef LOG_AUCTION

    size_t parallel_threshold { 5000 };
    bool is_step_parallel {false};
    std::unordered_set<size_t> unassigned_items;
    std::unordered_set<size_t> unassigned_normal_items;
    std::unordered_set<size_t> unassigned_diag_items;
    std::unordered_set<size_t> never_assigned_bidders;
    size_t all_assigned_round { 0 };
    size_t all_assigned_round_found { false };

    int num_rounds_non_cumulative { 0 }; // set to 0 in the beginning of each phase
    int num_diag_assignments { 0 };
    int num_diag_assignments_non_cumulative { 0 };
    int num_diag_bids_submitted { 0 };
    int num_diag_stole_from_diag { 0 };
    int num_normal_assignments { 0 };
    int num_normal_assignments_non_cumulative { 0 };
    int num_normal_bids_submitted { 0 };

    std::vector<std::vector<size_t>> price_change_cnt_vec;


    const char* plot_logger_name = "plot_logger";
    const char* price_state_logger_name = "price_stat_logger";
    std::string plot_logger_file_name;
    std::string price_stat_logger_file_name;
    std::shared_ptr<spdlog::logger> plot_logger;
    std::shared_ptr<spdlog::logger> price_stat_logger;
    std::shared_ptr<spdlog::logger> console_logger;


    int num_parallel_bids { 0 };
    int num_total_bids { 0 };

    int num_parallel_diag_bids { 0 };
    int num_total_diag_bids { 0 };

    int num_parallel_normal_bids { 0 };
    int num_total_normal_bids { 0 };

    int num_parallel_assignments { 0 };
    int num_total_assignments { 0 };
#endif

}; // AuctionRunnerJac


} // ws
} // hera

#include "auction_runner_jac.hpp"

#undef ORDERED_BY_PERSISTENCE

#endif
