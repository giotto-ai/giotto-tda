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

#ifndef AUCTION_RUNNER_JAC_HPP
#define AUCTION_RUNNER_JAC_HPP

#include <assert.h>
#include <algorithm>
#include <functional>
#include <iterator>

#include "def_debug_ws.h"
#include "auction_runner_jac.h"


#ifdef FOR_R_TDA
#include "Rcpp.h"
#undef DEBUG_AUCTION
#endif

namespace hera {
namespace ws {


// *****************************
// AuctionRunnerJac
// *****************************

    template<class R, class AO, class PC>
    AuctionRunnerJac<R, AO, PC>::AuctionRunnerJac(const PointContainer& A,
                                              const PointContainer& B,
                                              const AuctionParams<Real>& params,
                                              const std::string &_log_filename_prefix
    ) :
            bidders(A),
            items(B),
            num_bidders(A.size()),
            num_items(A.size()),
            items_to_bidders(A.size(), k_invalid_index),
            bidders_to_items(A.size(), k_invalid_index),
            wasserstein_power(params.wasserstein_power),
            delta(params.delta),
            internal_p(params.internal_p),
            initial_epsilon(params.initial_epsilon),
            epsilon_common_ratio(params.epsilon_common_ratio == 0.0 ? 5.0 : params.epsilon_common_ratio),
            max_num_phases(params.max_num_phases),
            bid_table(A.size(), std::make_pair(k_invalid_index, k_lowest_bid_value)),
            oracle(bidders, items, params),
            max_bids_per_round(params.max_bids_per_round),
            dimension(params.dim),
#ifndef WASSERSTEIN_PURE_GEOM
            total_items_persistence(std::accumulate(items.begin(),
                                                    items.end(),
                                                    R(0.0),
                                                    [params](const Real &ps, const DgmPoint &item) {
                                                        return ps + std::pow(item.persistence_lp(params.internal_p), params.wasserstein_power);
                                                    }
            )),
            total_bidders_persistence(std::accumulate(bidders.begin(),
                                                      bidders.end(),
                                                      R(0.0),
                                                      [params](const Real &ps, const DgmPoint &bidder) {
                                                          return ps + std::pow(bidder.persistence_lp(params.internal_p), params.wasserstein_power);
                                                      }
            )),
            unassigned_bidders_persistence(total_bidders_persistence),
            unassigned_items_persistence(total_items_persistence),
            gamma_threshold(params.gamma_threshold),
#endif
            log_filename_prefix(_log_filename_prefix)
    {
        assert(A.size() == B.size());

#ifndef WASSERSTEIN_PURE_GEOM
        for (const auto &p : bidders) {
            if (p.is_normal()) {
                num_normal_bidders++;
                num_diag_items++;
            } else {
                num_normal_items++;
                num_diag_bidders++;
            }
        }
#endif
        // for experiments
        unassigned_threshold = 100;

#ifdef ORDERED_BY_PERSISTENCE
        batch_size = 1000;
        for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
            if (is_bidder_normal(bidder_idx)) {
                unassigned_normal_bidders_by_persistence.insert(
                        std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx));
            }
        }
#endif

#ifdef LOG_AUCTION
        parallel_threshold = 16;
        console_logger = spdlog::get("console");
        if (not console_logger) {
            console_logger = spdlog::stdout_logger_st("console");
        }
        console_logger->set_pattern("[%H:%M:%S.%e] %v");
#ifdef ORDERED_BY_PERSISTENCE
        if (max_bids_per_round == 1) {
            console_logger->info("Gauss-Seidel imitated by Jacobi runner, q = {0}, max_bids_per_round = {1}, batch_size = {4}, gamma_threshold = {2}, diag_first = {3} ORDERED_BY_PERSISTENCE",
                              wasserstein_power,
                              max_bids_per_round,
                              gamma_threshold,
                              diag_first,
                              batch_size);
        } else {
            console_logger->info("Jacobi runner, q = {0}, max_bids_per_round = {1}, batch_size = {4},  gamma_threshold = {2}, diag_first = {3} ORDERED_BY_PERSISTENCE",
                              wasserstein_power,
                              max_bids_per_round,
                              gamma_threshold,
                              diag_first,
                              batch_size);
        }

#else
        if (max_bids_per_round == 1) {
            console_logger->info(
                    "Gauss-Seidel imitated by Jacobi runner, q = {0}, max_bids_per_round = {1}, batch_size = {4}, gamma_threshold = {2}, diag_first = {3}",
                    wasserstein_power,
                    max_bids_per_round,
                    gamma_threshold,
                    diag_first,
                    batch_size);
        } else {
            console_logger->info(
                    "Jacobi runner, q = {0}, max_bids_per_round = {1}, batch_size = {4},  gamma_threshold = {2}, diag_first = {3}",
                    wasserstein_power,
                    max_bids_per_round,
                    gamma_threshold,
                    diag_first,
                    batch_size);
        }
#endif

        plot_logger_file_name = log_filename_prefix + "_plot.txt";
        plot_logger = spdlog::get(plot_logger_name);
        if (not plot_logger) {
            plot_logger = spdlog::basic_logger_st(plot_logger_name, plot_logger_file_name);
        }
        plot_logger->info("New plot starts here, diagram size = {0}, gamma_threshold = {1}, epsilon_common_ratio = {2}",
                          bidders.size(),
                          gamma_threshold,
                          epsilon_common_ratio);
        plot_logger->set_pattern("%v");

        price_stat_logger_file_name = log_filename_prefix + "_price_change_stat";
        price_stat_logger = spdlog::get(price_state_logger_name);
        if (not price_stat_logger) {
            price_stat_logger = spdlog::basic_logger_st(price_state_logger_name,
                                                        price_stat_logger_file_name);
        }
        price_stat_logger->info(
                "New price statistics starts here, diagram size = {0}, gamma_threshold = {1}, epsilon_common_ratio = {2}",
                bidders.size(),
                gamma_threshold,
                epsilon_common_ratio);
        price_stat_logger->set_pattern("%v");
#endif
    }

#ifndef WASSERSTEIN_PURE_GEOM
    template<class R, class AO, class PC>
    typename AuctionRunnerJac<R, AO, PC>::Real
    AuctionRunnerJac<R, AO, PC>::get_cost_to_diagonal(const DgmPoint &pt) const {
        return std::pow(pt.persistence_lp(internal_p), wasserstein_power);
    }

    template<class R, class AO, class PC>
    typename AuctionRunnerJac<R, AO, PC>::Real
    AuctionRunnerJac<R, AO, PC>::get_gamma() const {
        return std::pow(std::fabs(unassigned_items_persistence + unassigned_bidders_persistence),
                        1.0 / wasserstein_power);
    }
#endif

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::assign_item_to_bidder(IdxType item_idx, IdxType bidder_idx)
    {
        //sanity_check();
        // only unassigned bidders submit bids
        assert(bidders_to_items[bidder_idx] == k_invalid_index);

        IdxType old_item_owner = items_to_bidders[item_idx];

        // set new owner
        bidders_to_items[bidder_idx] = item_idx;
        items_to_bidders[item_idx] = bidder_idx;

        // remove bidder and item from the sets of unassigned bidders/items
        remove_unassigned_bidder(bidder_idx);

        if (k_invalid_index != old_item_owner) {
            // old owner of item becomes unassigned
            bidders_to_items[old_item_owner] = k_invalid_index;
            add_unassigned_bidder(old_item_owner);
            // existing edge was removed, decrease partial_cost
            partial_cost -= get_item_bidder_cost(item_idx, old_item_owner);
        } else {
            // item was unassigned before
            remove_unassigned_item(item_idx);
        }

        // new edge was added to matching, increase partial cost
        partial_cost += get_item_bidder_cost(item_idx, bidder_idx);

#ifdef LOG_AUCTION
        if (is_item_diagonal(item_idx)) {
            num_diag_assignments++;
            num_diag_assignments_non_cumulative++;
        } else {
            num_normal_assignments++;
            num_normal_assignments_non_cumulative++;
        }

        if (k_invalid_index != old_item_owner) {
            if (is_bidder_diagonal(bidder_idx) and is_bidder_diagonal(old_item_owner)) {
                num_diag_stole_from_diag++;
            }
        }
#endif

        //sanity_check();
    }

    template<class R, class AO, class PC>
    typename AuctionRunnerJac<R, AO, PC>::Real
    AuctionRunnerJac<R, AO, PC>::get_item_bidder_cost(const size_t item_idx, const size_t bidder_idx) const
    {
        return std::pow(dist_lp(bidders[bidder_idx], items[item_idx], internal_p, dimension),
                        wasserstein_power);
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::assign_to_best_bidder(IdxType item_idx) {
        assert(item_idx >= 0 and item_idx < static_cast<IdxType>(num_items));
        assert(bid_table[item_idx].first != k_invalid_index);
        IdxValPairR best_bid{bid_table[item_idx]};
        assign_item_to_bidder(item_idx, best_bid.first);
        oracle.set_price(item_idx, best_bid.second);
#ifdef LOG_AUCTION

        if (is_step_parallel) {
            num_parallel_assignments++;
        }
        num_total_assignments++;

        price_change_cnt_vec.back()[item_idx]++;
#endif
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::clear_bid_table() {
        auto iter = items_with_bids.begin();
        while (iter != items_with_bids.end()) {
            auto item_with_bid_idx = *iter;
            bid_table[item_with_bid_idx].first = k_invalid_index;
            bid_table[item_with_bid_idx].second = k_lowest_bid_value;
            iter = items_with_bids.erase(iter);
        }
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::submit_bid(IdxType bidder_idx, const IdxValPairR &bid) {
        IdxType item_idx = bid.first;
        Real bid_value = bid.second;
        assert(item_idx >= 0);
        if (bid_table[item_idx].second < bid_value) {
            bid_table[item_idx].first = bidder_idx;
            bid_table[item_idx].second = bid_value;
        }
        items_with_bids.insert(item_idx);

#ifdef LOG_AUCTION

        num_total_bids++;


        if (is_bidder_diagonal(bidder_idx)) {
            num_diag_bids_submitted++;
        } else {
            num_normal_bids_submitted++;
        }
#endif
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::print_debug() {
#ifdef DEBUG_AUCTION
        sanity_check();
        std::cout << "**********************" << std::endl;
        std::cout << "Current assignment:" << std::endl;
        for(size_t idx = 0; idx < bidders_to_items.size(); ++idx) {
            std::cout << idx << " <--> " << bidders_to_items[idx] << std::endl;
        }
        std::cout << "Weights: " << std::endl;
        //for(size_t i = 0; i < num_bidders; ++i) {
            //for(size_t j = 0; j < num_items; ++j) {
                //std::cout << oracle.weight_matrix[i][j] << " ";
            //}
            //std::cout << std::endl;
        //}
        std::cout << "Prices: " << std::endl;
        for(const auto price : oracle.get_prices()) {
            std::cout << price << std::endl;
        }
        //std::cout << "Value matrix: " << std::endl;
        //for(size_t i = 0; i < num_bidders; ++i) {
            //for(size_t j = 0; j < num_items; ++j) {
                //std::cout << oracle.weight_matrix[i][j] - oracle.prices[j] << " ";
            //}
            //std::cout << std::endl;
        //}
        std::cout << "**********************" << std::endl;
#endif
    }

    template<class R, class AO, class PC>
    typename AuctionRunnerJac<R, AO, PC>::Real
    AuctionRunnerJac<R, AO, PC>::get_relative_error(const bool debug_output) const
    {
        if (partial_cost == 0.0 and unassigned_bidders.empty())
            return 0.0;
        Real result;
#ifndef WASSERSTEIN_PURE_GEOM
        Real gamma = get_gamma();
#else
        Real gamma = 0.0;
#endif
        // cost minus n epsilon
        Real reduced_cost = partial_cost - num_bidders * get_epsilon();
        if (reduced_cost < 0) {
#ifdef LOG_AUCTION
            if (debug_output) {
                console_logger->info("Epsilon too large, reduced_cost = {0}, gamma = {1}", reduced_cost, gamma);
            }
#endif
            result = k_max_relative_error;
        } else {
            Real denominator = std::pow(reduced_cost, 1.0 / wasserstein_power) - gamma;
            if (denominator <= 0) {
#ifdef LOG_AUCTION
                if (debug_output) {
                    console_logger->info("Epsilon too large, reduced_cost = {0}, denominator = {1}, gamma = {2}",
                                         reduced_cost, denominator, gamma);
                }
#endif
                result = k_max_relative_error;
            } else {
                Real numerator = 2 * gamma +
                                 std::pow(partial_cost, 1.0 / wasserstein_power) -
                                 std::pow(reduced_cost, 1.0 / wasserstein_power);

                result = numerator / denominator;
#ifdef LOG_AUCTION
                if (debug_output) {
                    console_logger->info(
                            "Reduced_cost = {0}, denominator = {1}, numerator {2}, error = {3},  gamma = {4}",
                            reduced_cost,
                            denominator,
                            numerator,
                            result,
                            gamma);
                }
#endif
            }
        }
        return result;
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::flush_assignment() {
        for (auto &b2i : bidders_to_items) {
            b2i = k_invalid_index;
        }
        for (auto &i2b : items_to_bidders) {
            i2b = k_invalid_index;
        }

        // all bidders and items become unassigned
        for (size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
            unassigned_bidders.insert(bidder_idx);
        }

#ifdef ORDERED_BY_PERSISTENCE
        for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
            if (is_bidder_normal(bidder_idx)) {
                unassigned_normal_bidders_by_persistence.insert(
                        std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx));
            }
        }
#endif
        oracle.adjust_prices();

        partial_cost = 0.0;


#ifndef WASSERSTEIN_PURE_GEOM
        for (size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
            if (is_bidder_normal(bidder_idx)) {
                unassigned_normal_bidders.insert(bidder_idx);
            } else {
                unassigned_diag_bidders.insert(bidder_idx);
            }
        }

        unassigned_bidders_persistence = total_bidders_persistence;
        unassigned_items_persistence = total_items_persistence;

#ifdef LOG_AUCTION

        price_change_cnt_vec.push_back(std::vector<size_t>(num_items, 0));

        never_assigned_bidders = unassigned_bidders;

        for (size_t item_idx = 0; item_idx < items.size(); ++item_idx) {
            unassigned_items.insert(item_idx);
            if (is_item_normal(item_idx)) {
                unassigned_normal_items.insert(item_idx);
            } else {
                unassigned_diag_items.insert(item_idx);
            }
        }

        num_diag_bids_submitted = 0;
        num_normal_bids_submitted = 0;
        num_diag_assignments = 0;
        num_normal_assignments = 0;

        all_assigned_round = 0;
        all_assigned_round_found = false;
        num_rounds_non_cumulative = 0;
#endif
#endif

    } // flush_assignment


    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::set_epsilon(Real new_val) {
        assert(new_val > 0.0);
        epsilon = new_val;
        oracle.set_epsilon(new_val);
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::run_auction_phases(const int max_num_phases, const Real _initial_epsilon) {
        set_epsilon(_initial_epsilon);
        assert(oracle.get_epsilon() > 0);
        for (int phase_num = 0; phase_num < max_num_phases; ++phase_num) {
            flush_assignment();
            run_auction_phase();

#ifdef LOG_AUCTION
            console_logger->info(
                    "Phase {0} done, current_result = {1}, eps = {2}, error = {7}, num_rounds = {3}, num_assignments = {4},  num_bids_submitted = {5}, # unassigned = {6}",
                    num_phase,
                    partial_cost,
                    get_epsilon(),
                    format_int<>(num_rounds),
                    format_int<>(num_normal_assignments + num_diag_assignments),
                    format_int<>(num_normal_bids_submitted + num_diag_bids_submitted),
                    unassigned_bidders.size(),
                    get_relative_error(num_phase == 1)
            );

//        console_logger->info("num_rounds (non-cumulative)= {0}, num_diag_assignments = {1}, num_normal_assignments = {2}, num_diag_bids_submitted = {3},  num_normal_bids_submitted = {4}",
//                              format_int<>(num_rounds_non_cumulative),
//                              format_int<>(num_diag_assignments),
//                              format_int<>(num_normal_assignments),
//                              format_int<>(num_diag_bids_submitted),
//                              format_int<>(num_normal_bids_submitted)
//                             );

            console_logger->info(
                    "num_parallel_bids / num_total_bids = {0} / {1} = {2}, num_parallel_assignments / num_total_aassignments = {3} / {4} = {5}",
                    format_int<>(num_parallel_bids),
                    format_int<>(num_total_bids),
                    static_cast<double>(num_parallel_bids) / static_cast<double>(num_total_bids),
                    format_int<>(num_parallel_assignments),
                    format_int<>(num_total_assignments),
                    static_cast<double>(num_parallel_assignments) / static_cast<double>(num_total_assignments)
            );

            console_logger->info(
                    "num_parallel_diag_bids / num_total_diag_bids = {0} / {1} = {2}, num_parallel_normal_bids / num_total_normal_bids = {3} / {4} = {5}",
                    format_int<>(num_parallel_diag_bids),
                    format_int<>(num_total_diag_bids),
                    static_cast<double>(num_parallel_diag_bids) / static_cast<double>(num_total_diag_bids),
                    format_int<>(num_parallel_normal_bids),
                    format_int<>(num_total_normal_bids),
                    static_cast<double>(num_parallel_normal_bids) / static_cast<double>(num_total_normal_bids)
            );

//        console_logger->info("num_rounds before all biders assigned = {0}, num_rounds (non-cumulative)= {1}, fraction = {2}",
//                              format_int<>(all_assigned_round),
//                              format_int<>(num_rounds_non_cumulative),
//                              static_cast<double>(all_assigned_round) / static_cast<double>(num_rounds_non_cumulative)
//                             );

            for (size_t item_idx = 0; item_idx < num_items; ++item_idx) {
                price_stat_logger->info("{0} {1} {2} {3} {4}",
                                        phase_num,
                                        item_idx,
                                        items[item_idx][0],
                                        items[item_idx][1],
                                        price_change_cnt_vec.back()[item_idx]
                );
            }
#endif


            if (is_done())
                break;
            else
                decrease_epsilon();

        }
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::decrease_epsilon() {
        set_epsilon(get_epsilon() / epsilon_common_ratio);
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::run_auction()
    {
        if (num_bidders == 1) {
            assign_item_to_bidder(0, 0);
            wasserstein_cost = get_item_bidder_cost(0,0);
            is_distance_computed = true;
            return;
        }
        R init_eps = (initial_epsilon > 0.0) ? initial_epsilon : oracle.max_val_ / 4.0;
        run_auction_phases(max_num_phases, init_eps);
        is_distance_computed = true;
        wasserstein_cost = partial_cost;
        if (not is_done()) {
#ifndef FOR_R_TDA
            std::cerr << "Maximum iteration number exceeded, exiting. Current result is: ";
            std::cerr << get_wasserstein_distance() << std::endl;
#endif
            throw std::runtime_error("Maximum iteration number exceeded");
        }
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::add_unassigned_bidder(const size_t bidder_idx)
    {
        unassigned_bidders.insert(bidder_idx);

#ifndef WASSERSTEIN_PURE_GEOM
        const auto &bidder = bidders[bidder_idx];
        unassigned_bidders_persistence += get_cost_to_diagonal(bidder);

        if (is_bidder_diagonal(bidder_idx)) {
            unassigned_diag_bidders.insert(bidder_idx);
        } else {
            unassigned_normal_bidders.insert(bidder_idx);
        }
#ifdef ORDERED_BY_PERSISTENCE
        if (is_bidder_normal(bidder_idx)) {
            unassigned_normal_bidders_by_persistence.insert(std::make_pair(bidder.persistence_lp(1.0), bidder_idx));
        }
#endif

#endif
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::remove_unassigned_bidder(const size_t bidder_idx)
    {
        unassigned_bidders.erase(bidder_idx);
#ifndef WASSERSTEIN_PURE_GEOM
        const auto &bidder = bidders[bidder_idx];
        unassigned_bidders_persistence -= get_cost_to_diagonal(bidder);

#ifdef ORDERED_BY_PERSISTENCE
        if (is_bidder_normal(bidder_idx)) {
            unassigned_normal_bidders_by_persistence.erase(std::make_pair(bidder.persistence_lp(1.0), bidder_idx));
        }
#endif

        if (is_bidder_diagonal(bidder_idx)) {
            unassigned_diag_bidders.erase(bidder_idx);
        } else {
            unassigned_normal_bidders.erase(bidder_idx);
        }


#ifdef LOG_AUCTION
        never_assigned_bidders.erase(bidder_idx);
        if (never_assigned_bidders.empty() and not all_assigned_round_found) {
            all_assigned_round = num_rounds_non_cumulative;
            all_assigned_round_found = true;
        }
#endif
#endif
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::remove_unassigned_item(const size_t item_idx) {
#ifndef WASSERSTEIN_PURE_GEOM
        unassigned_items_persistence -= get_cost_to_diagonal(items[item_idx]);

#ifdef LOG_AUCTION
        unassigned_items.erase(item_idx);

        if (is_item_normal(item_idx)) {
            unassigned_normal_items.erase(item_idx);
        } else {
            unassigned_diag_items.erase(item_idx);
        }
#endif
#endif
    }

    template<class R, class AO, class PC>
    template<class Range>
    void AuctionRunnerJac<R, AO, PC>::run_bidding_step(const Range &active_bidders)
    {
#ifdef LOG_AUCTION
        is_step_parallel = false;
        size_t diag_bids_submitted = 0;
        size_t normal_bids_submitted = 0;
#endif

        clear_bid_table();
        size_t bids_submitted = 0;
        for (const auto bidder_idx : active_bidders) {

            ++bids_submitted;

            submit_bid(bidder_idx, oracle.get_optimal_bid(bidder_idx));

#ifdef LOG_AUCTION
            if (is_bidder_diagonal(bidder_idx)) {
                diag_bids_submitted++;
            } else {
                normal_bids_submitted++;
            }

            if (bids_submitted >= parallel_threshold) {
                is_step_parallel = true;
            }

            if (bids_submitted >= max_bids_per_round) {
                break;
            }
            if (diag_first and not unassigned_diag_bidders.empty() and
                diag_bids_submitted >= oracle.get_heap_top_size()) {
                continue;
            }
#endif
        }

#ifdef LOG_AUCTION
        num_total_diag_bids += diag_bids_submitted;
        num_total_normal_bids += normal_bids_submitted;
        if (is_step_parallel) {
            num_parallel_bids += bids_submitted;
            num_parallel_diag_bids += diag_bids_submitted;
            num_parallel_normal_bids += normal_bids_submitted;
        }
#endif
    }

    template<class R, class AO, class PC>
    bool AuctionRunnerJac<R, AO, PC>::is_done() const
    {
        return get_relative_error() <= delta;
    }

    template<class R, class AO, class PC>
    bool AuctionRunnerJac<R, AO, PC>::continue_auction_phase() const
    {
#ifdef WASSERSTEIN_PURE_GEOM
        return not unassigned_bidders.empty();
#else
        return not unassigned_bidders.empty() and not is_done();
#endif
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::run_auction_phase()
    {
        num_phase++;
        //console_logger->debug("Entered run_auction_phase");

        do {
            num_rounds++;
#ifdef LOG_AUCTION
            num_diag_stole_from_diag = 0;
            num_normal_assignments_non_cumulative = 0;
            num_diag_assignments_non_cumulative = 0;
            num_rounds_non_cumulative++;
#endif

            // bidding
#ifdef ORDERED_BY_PERSISTENCE
            if (not unassigned_diag_bidders.empty()) {
                run_bidding_step(unassigned_diag_bidders);
            } else {
                std::vector<size_t> active_bidders;
                active_bidders.reserve(batch_size);
                for (auto iter = unassigned_normal_bidders_by_persistence.begin(); iter != unassigned_normal_bidders_by_persistence.end(); ++iter) {
                    active_bidders.push_back(iter->second);
                    if (active_bidders.size() >= batch_size) {
                        break;
                    }
                }
                run_bidding_step(active_bidders);
            }
#elif defined WASSERSTEIN_PURE_GEOM
            run_bidding_step(unassigned_bidders);
#else
            if (diag_first and not unassigned_diag_bidders.empty()) {
                run_bidding_step(unassigned_diag_bidders);
            } else {
                run_bidding_step(unassigned_bidders);
            }
#endif

            // assignment
            for (auto item_idx : items_with_bids) {
                assign_to_best_bidder(item_idx);
            }
#ifdef LOG_AUCTION
            plot_logger->info("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14}",
                              num_phase,
                              num_rounds,
                              unassigned_bidders.size(),
                              get_gamma(),
                              partial_cost,
                              oracle.get_epsilon(),
                              unassigned_normal_bidders.size(),
                              unassigned_diag_bidders.size(),
                              unassigned_normal_items.size(),
                              unassigned_diag_items.size(),
                              num_normal_assignments_non_cumulative,
                              num_diag_assignments_non_cumulative,
                              oracle.get_heap_top_size(),
                              get_relative_error(false),
                              num_diag_stole_from_diag
            );
#endif
            //sanity_check();
        } while (continue_auction_phase());
    }

    template<class R, class AO, class PC>
    typename AuctionRunnerJac<R, AO, PC>::Real
    AuctionRunnerJac<R, AO, PC>::get_wasserstein_distance()
    {
        assert(is_distance_computed);
        return std::pow(wasserstein_cost, 1.0 / wasserstein_power);
    }

    template<class R, class AO, class PC>
    typename AuctionRunnerJac<R, AO, PC>::Real
    AuctionRunnerJac<R, AO, PC>::get_wasserstein_cost()
    {
        assert(is_distance_computed);
        return wasserstein_cost;
    }


    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::sanity_check()
    {
#ifdef DEBUG_AUCTION
        if (bidders_to_items.size() != num_bidders) {
            std::cerr << "Wrong size of bidders_to_items, must be " << num_bidders << ", is " << bidders_to_items.size() << std::endl;
            throw "Wrong size of bidders_to_items";
        }

        if (items_to_bidders.size() != num_bidders) {
            std::cerr << "Wrong size of items_to_bidders, must be " << num_bidders << ", is " << items_to_bidders.size() << std::endl;
            throw "Wrong size of items_to_bidders";
        }

        for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
            if ( bidders_to_items[bidder_idx] >= 0) {

                if ( std::count(bidders_to_items.begin(),
                            bidders_to_items.end(),
                            bidders_to_items[bidder_idx]) > 1 ) {
                    std::cerr << "Good " << bidders_to_items[bidder_idx];
                    std::cerr << " appears in bidders_to_items more than once" << std::endl;
                    throw "Duplicate in bidders_to_items";
                }

                if (items_to_bidders.at(bidders_to_items[bidder_idx]) != static_cast<int>(bidder_idx)) {
                    std::cerr << "Inconsitency: bidder_idx = " << bidder_idx;
                    std::cerr << ", item_idx in bidders_to_items = ";
                    std::cerr << bidders_to_items[bidder_idx];
                    std::cerr << ", bidder_idx in items_to_bidders = ";
                    std::cerr << items_to_bidders[bidders_to_items[bidder_idx]] << std::endl;
                    throw "inconsistent mapping";
                }
            }
        }

        for(IdxType item_idx = 0; item_idx < static_cast<IdxType>(num_bidders); ++item_idx) {
            if ( items_to_bidders[item_idx] >= 0) {

                // check for uniqueness
                if ( std::count(items_to_bidders.begin(),
                            items_to_bidders.end(),
                            items_to_bidders[item_idx]) > 1 ) {
                    std::cerr << "Bidder " << items_to_bidders[item_idx];
                    std::cerr << " appears in items_to_bidders more than once" << std::endl;
                    throw "Duplicate in items_to_bidders";
                }
                // check for consistency
                if (bidders_to_items.at(items_to_bidders[item_idx]) != static_cast<int>(item_idx)) {
                    std::cerr << "Inconsitency: item_idx = " << item_idx;
                    std::cerr << ", bidder_idx in items_to_bidders = ";
                    std::cerr << items_to_bidders[item_idx];
                    std::cerr << ", item_idx in bidders_to_items= ";
                    std::cerr << bidders_to_items[items_to_bidders[item_idx]] << std::endl;
                    throw "inconsistent mapping";
                }
            }
        }
#endif
    }

    template<class R, class AO, class PC>
    void AuctionRunnerJac<R, AO, PC>::print_matching() {
#ifdef DEBUG_AUCTION
        sanity_check();
        for(size_t bidder_idx = 0; bidder_idx < bidders_to_items.size(); ++bidder_idx) {
            if (bidders_to_items[bidder_idx] >= 0) {
                auto pA = bidders[bidder_idx];
                auto pB = items[bidders_to_items[bidder_idx]];
                std::cout <<  pA << " <-> " << pB << "+" << pow(dist_lp(pA, pB, internal_p, dimension), wasserstein_power) << std::endl;
            } else {
                assert(false);
            }
        }
#endif
    }

} // ws
} // hera

#endif
