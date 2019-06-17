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

#ifndef AUCTION_RUNNER_FR_H
#define AUCTION_RUNNER_FR_H

#define ORDERED_BY_PERSISTENCE

#include <unordered_set>

#include "auction_oracle.h"

namespace hera {
namespace ws {

// the two parameters that you can tweak in auction algorithm are:
// 1. epsilon_common_ratio
// 2. max_num_phases

template<class RealType_, class AuctionOracle_ = AuctionOracleKDTreeRestricted<RealType_>>      // alternatively: AuctionOracleLazyHeap --- TODO
class AuctionRunnerFR {
public:

    using Real          = RealType_;
    using AuctionOracle = AuctionOracle_;
    using DgmPoint      = DiagramPoint<Real>;
    using DgmPointVec   = std::vector<DgmPoint>;
    using IdxValPairR   = IdxValPair<Real>;

    const Real k_lowest_bid_value = -(std::numeric_limits<Real>::max() - 1); // all bid values must be positive


    AuctionRunnerFR(const std::vector<DgmPoint>& A,
                     const std::vector<DgmPoint>& B,
                     const Real q,
                     const Real _delta,
                     const Real _internal_p,
                     const Real _initial_epsilon = 0.0,
                     const Real _eps_factor = 5.0,
                     const int _max_num_phases = std::numeric_limits<int>::max(),
                     const Real _gamma_threshold = 0.0,
                     const size_t _max_bids_per_round = std::numeric_limits<size_t>::max(),
                     const std::string& _log_filename_prefix = "");

    void set_epsilon(Real new_val);
    Real get_epsilon() const { return epsilon; }
    void run_auction();
    Real get_wasserstein_distance();
    Real get_wasserstein_cost();
    Real get_relative_error(const bool debug_output = false) const;
    bool phase_can_be_final() const;
private:
    // private data
    std::vector<DgmPoint> bidders, items;
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
    Real cumulative_epsilon_factor { 1.0 };
    const int max_num_phases; // maximal number of phases of epsilon-scaling
    bool is_forward { true }; // to use in distributed version only
    Real weight_adj_const;
    Real wasserstein_cost;
    std::vector<IdxValPairR> forward_bid_table;
    std::vector<IdxValPairR> reverse_bid_table;
    // to get the 2 best items
    AuctionOracle forward_oracle;
    AuctionOracle reverse_oracle;
    std::unordered_set<size_t> unassigned_bidders;
    std::unordered_set<size_t> unassigned_items;
    std::unordered_set<size_t> items_with_bids;
    std::unordered_set<size_t> bidders_with_bids;

#ifdef ORDERED_BY_PERSISTENCE
    // to process unassigned by persistence
    size_t batch_size;
    using RealIdxPair = std::pair<Real, size_t>;
    std::set<RealIdxPair, std::greater<RealIdxPair>> unassigned_bidders_by_persistence;
    std::set<RealIdxPair, std::greater<RealIdxPair>> unassigned_items_by_persistence;
#endif


    // to imitate Gauss-Seidel
    const size_t max_bids_per_round;

    // to stop earlier in the last phase
    const Real total_items_persistence;
    const Real total_bidders_persistence;
    Real partial_cost;
    Real unassigned_bidders_persistence;
    Real unassigned_items_persistence;
    Real gamma_threshold;
    size_t unassigned_threshold;

    bool is_distance_computed { false };
    int num_rounds { 0 };
    int num_rounds_non_cumulative { 0 };
    int num_phase { 0 };


    size_t num_diag_items { 0 };
    size_t num_normal_items { 0 };
    size_t num_diag_bidders { 0 };
    size_t num_normal_bidders { 0 };



    // private methods
    void assign_forward(const IdxType item_idx, const IdxType bidder_idx);
    void assign_reverse(const IdxType item_idx, const IdxType bidder_idx);
    void assign_to_best_bidder(const IdxType item_idx);
    void assign_to_best_item(const IdxType bidder_idx);
    void clear_forward_bid_table();
    void clear_reverse_bid_table();
    void assign_diag_to_diag();
    void run_auction_phases(const int max_num_phases, const Real _initial_epsilon);
    void run_auction_phase();
    void run_forward_auction_phase();
    void run_reverse_auction_phase();
    void submit_forward_bid(IdxType bidder_idx, const IdxValPairR& bid);
    void submit_reverse_bid(IdxType item_idx, const IdxValPairR& bid);
    void flush_assignment();
    Real get_item_bidder_cost(const size_t item_idx, const size_t bidder_idx) const;
    Real get_cost_to_diagonal(const DgmPoint& pt) const;
    Real get_gamma() const;

    template<class Range>
    void run_forward_bidding_step(const Range& r);

    template<class Range>
    void run_reverse_bidding_step(const Range& r);

    void add_unassigned_bidder(const size_t bidder_idx);
    void add_unassigned_item(const size_t item_idx);
    void remove_unassigned_bidder(const size_t bidder_idx);
    void remove_unassigned_item(const size_t item_idx);

    bool is_item_diagonal(const size_t item_idx) const { return item_idx < num_diag_items; }
    bool is_item_normal(const size_t item_idx) const { return not is_item_diagonal(item_idx); }
    bool is_bidder_diagonal(const size_t bidder_idx) const { return bidder_idx >= num_normal_bidders; }
    bool is_bidder_normal(const size_t bidder_idx) const { return not is_bidder_diagonal(bidder_idx); }

    size_t num_forward_bids_submitted { 0 };
    size_t num_reverse_bids_submitted { 0 };

    void decrease_epsilon();
    // stopping criteria
    bool continue_forward(const size_t, const size_t);
    bool continue_reverse(const size_t, const size_t);
    bool continue_phase();



    // for debug only
    void sanity_check();
    void check_epsilon_css();
    void print_debug();
    void print_matching();

    std::string log_filename_prefix;
    const Real k_max_relative_error = 2.0; // if relative error cannot be estimated or is too large, use this value
    void reset_round_stat(); // empty, if logging is disable
    void reset_phase_stat();

    std::unordered_set<size_t> never_assigned_bidders;
    std::unordered_set<size_t> never_assigned_items;

    std::shared_ptr<spdlog::logger> console_logger;
#ifdef LOG_AUCTION
    std::unordered_set<size_t> unassigned_normal_bidders;
    std::unordered_set<size_t> unassigned_diag_bidders;

    std::unordered_set<size_t> unassigned_normal_items;
    std::unordered_set<size_t> unassigned_diag_items;


    size_t all_assigned_round { 0 };
    size_t all_assigned_round_found { false };

    int num_forward_rounds_non_cumulative { 0 };
    int num_forward_rounds { 0 };

    int num_reverse_rounds_non_cumulative { 0 };
    int num_reverse_rounds { 0 };

    // all per-round vars are non-cumulative

    // forward rounds
    int num_normal_forward_bids_submitted { 0 };
    int num_diag_forward_bids_submitted { 0 };

    int num_forward_diag_to_diag_assignments { 0 };
    int num_forward_diag_to_normal_assignments { 0 };
    int num_forward_normal_to_diag_assignments { 0 };
    int num_forward_normal_to_normal_assignments { 0 };

    int num_forward_diag_from_diag_thefts { 0 };
    int num_forward_diag_from_normal_thefts { 0 };
    int num_forward_normal_from_diag_thefts { 0 };
    int num_forward_normal_from_normal_thefts { 0 };

    // reverse rounds
    int num_normal_reverse_bids_submitted { 0 };
    int num_diag_reverse_bids_submitted { 0 };

    int num_reverse_diag_to_diag_assignments { 0 };
    int num_reverse_diag_to_normal_assignments { 0 };
    int num_reverse_normal_to_diag_assignments { 0 };
    int num_reverse_normal_to_normal_assignments { 0 };

    int num_reverse_diag_from_diag_thefts { 0 };
    int num_reverse_diag_from_normal_thefts { 0 };
    int num_reverse_normal_from_diag_thefts { 0 };
    int num_reverse_normal_from_normal_thefts { 0 };

    // price change statistics
    std::vector<std::vector<size_t>> forward_price_change_cnt_vec;
    std::vector<std::vector<size_t>> reverse_price_change_cnt_vec;

    const char* forward_plot_logger_name = "forward_plot_logger";
    const char* reverse_plot_logger_name = "reverse_plot_logger";
    const char* forward_price_stat_logger_name = "forward_price_stat_logger";
    const char* reverse_price_stat_logger_name = "reverse_price_stat_logger";

    std::string forward_plot_logger_file_name;
    std::string reverse_plot_logger_file_name;
    std::string forward_price_stat_logger_file_name;
    std::string reverse_price_stat_logger_file_name;

    std::shared_ptr<spdlog::logger> forward_plot_logger;
    std::shared_ptr<spdlog::logger> reverse_plot_logger;
    std::shared_ptr<spdlog::logger> forward_price_stat_logger;
    std::shared_ptr<spdlog::logger> reverse_price_stat_logger;


    size_t parallel_threshold = 5000;
    int num_parallelizable_rounds { 0 };
    int num_parallelizable_forward_rounds { 0 };
    int num_parallelizable_reverse_rounds { 0 };

    int num_parallel_bids { 0 };
    int num_total_bids { 0 };

    int num_parallel_assignments { 0 };
    int num_total_assignments { 0 };
#endif

};



} // ws
} // hera

#include "auction_runner_fr.hpp"

#undef ORDERED_BY_PERSISTENCE
#endif
