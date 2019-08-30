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

#ifndef AUCTION_RUNNER_FR_HPP
#define AUCTION_RUNNER_FR_HPP

#include <cassert>
#include <algorithm>
#include <functional>
#include <iterator>

#include "def_debug_ws.h"

#include "auction_runner_fr.h"

#ifdef FOR_R_TDA
#include "Rcpp.h"
#undef DEBUG_FR_AUCTION
#endif


namespace hera {
namespace ws {


// *****************************
// AuctionRunnerFR
// *****************************

template<class R, class AO>
AuctionRunnerFR<R, AO>::AuctionRunnerFR(const std::vector<DgmPoint>& A,
                                          const std::vector<DgmPoint>& B,
                                          const Real q,
                                          const Real _delta,
                                          const Real _internal_p,
                                          const Real _initial_epsilon,
                                          const Real _eps_factor,
                                          const int _max_num_phases,
                                          const Real _gamma_threshold,
                                          const size_t _max_bids_per_round,
                                          const std::string& _log_filename_prefix
                                         ) :
    bidders(A),
    items(B),
    num_bidders(A.size()),
    num_items(A.size()),
    items_to_bidders(A.size(), k_invalid_index),
    bidders_to_items(A.size(), k_invalid_index),
    wasserstein_power(q),
    delta(_delta),
    internal_p(_internal_p),
    initial_epsilon(_initial_epsilon),
    epsilon_common_ratio(_eps_factor == 0.0 ? 5.0 : _eps_factor),
    max_num_phases(_max_num_phases),
    forward_bid_table(A.size(), std::make_pair(k_invalid_index, k_lowest_bid_value) ),
    reverse_bid_table(B.size(), std::make_pair(k_invalid_index, k_lowest_bid_value) ),
    forward_oracle(bidders, items, q, _internal_p),
    reverse_oracle(items, bidders, q, _internal_p),
    max_bids_per_round(_max_bids_per_round),
    total_items_persistence(std::accumulate(items.begin(),
                                            items.end(),
                                            R(0.0),
                                            [_internal_p, q](const Real& ps, const DgmPoint& item)
                                                { return ps + std::pow(item.persistence_lp(_internal_p), q); }
                                           )),
    total_bidders_persistence(std::accumulate(bidders.begin(),
                                              bidders.end(),
                                              R(0.0),
                                              [_internal_p, q](const Real& ps, const DgmPoint& bidder)
                                                  { return ps + std::pow(bidder.persistence_lp(_internal_p), q); }
                                             )),
    partial_cost(0.0),
    unassigned_bidders_persistence(total_bidders_persistence),
    unassigned_items_persistence(total_items_persistence),
    gamma_threshold(_gamma_threshold),
    log_filename_prefix(_log_filename_prefix)
{
    assert(A.size() == B.size());
    for(const auto& p : bidders) {
        if (p.is_normal()) {
            num_normal_bidders++;
            num_diag_items++;
        } else {
            num_normal_items++;
            num_diag_bidders++;
        }
    }

#ifdef ORDERED_BY_PERSISTENCE
    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        unassigned_bidders_by_persistence.insert(std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx));
    }

    for(size_t item_idx = 0; item_idx < num_items; ++item_idx) {
        unassigned_items_by_persistence.insert(std::make_pair(items[item_idx].persistence_lp(1.0), item_idx));
    }
#endif

    // for experiments
    unassigned_threshold = bidders.size() / 200;
    unassigned_threshold = 0;

#ifdef ORDERED_BY_PERSISTENCE
    batch_size = 5000;
#endif

    console_logger = spdlog::get("console");
    if (not console_logger) {
        console_logger = spdlog::stdout_logger_st("console");
    }
    console_logger->set_pattern("[%H:%M:%S.%e] %v");
    console_logger->info("Forward-reverse runnder, max_num_phases = {0}, max_bids_per_round = {1}, gamma_threshold = {2}, unassigned_threshold = {3}",
                          max_num_phases,
                          max_bids_per_round,
                          gamma_threshold,
                          unassigned_threshold);


//    check_epsilon_css();
#ifdef LOG_AUCTION
    parallel_threshold = bidders.size() / 100;
    forward_plot_logger_file_name = log_filename_prefix + "_forward_plot.txt";
    forward_plot_logger = spdlog::get(forward_plot_logger_name);
    if (not forward_plot_logger) {
        forward_plot_logger = spdlog::basic_logger_st(forward_plot_logger_name, forward_plot_logger_file_name);
    }
    forward_plot_logger->info("New forward plot starts here, diagram size = {0}, gamma_threshold = {1}, epsilon_common_ratio = {2}",
                              bidders.size(),
                              gamma_threshold,
                              epsilon_common_ratio);
    forward_plot_logger->set_pattern("%v");

    reverse_plot_logger_file_name = log_filename_prefix + "_reverse_plot.txt";
    reverse_plot_logger = spdlog::get(reverse_plot_logger_name);
    if (not reverse_plot_logger) {
        reverse_plot_logger = spdlog::basic_logger_st(reverse_plot_logger_name, reverse_plot_logger_file_name);
    }
    reverse_plot_logger->info("New reverse plot starts here, diagram size = {0}, gamma_threshold = {1}, epsilon_common_ratio = {2}",
                              bidders.size(),
                              gamma_threshold,
                              epsilon_common_ratio);
    reverse_plot_logger->set_pattern("%v");



    forward_price_stat_logger_file_name = log_filename_prefix + "_forward_price_change_stat";
    forward_price_stat_logger = spdlog::get(forward_price_stat_logger_name);
    if (not forward_price_stat_logger) {
        forward_price_stat_logger = spdlog::basic_logger_st(forward_price_stat_logger_name,
                                                            forward_price_stat_logger_file_name);
    }
    forward_price_stat_logger->info("New forward price statistics starts here, diagram size = {0}, gamma_threshold = {1}, epsilon_common_ratio = {2}",
                                     bidders.size(),
                                     gamma_threshold,
                                     epsilon_common_ratio);
    forward_price_stat_logger->set_pattern("%v");

    reverse_price_stat_logger_file_name = log_filename_prefix + "_reverse_price_change_stat";
    reverse_price_stat_logger = spdlog::get(reverse_price_stat_logger_name);
    if (not reverse_price_stat_logger) {
        reverse_price_stat_logger = spdlog::basic_logger_st(reverse_price_stat_logger_name,
                                                            reverse_price_stat_logger_file_name);
    }
    reverse_price_stat_logger->info("New reverse price statistics starts here, diagram size = {0}, gamma_threshold = {1}, epsilon_common_ratio = {2}",
                                     bidders.size(),
                                     gamma_threshold,
                                     epsilon_common_ratio);
    reverse_price_stat_logger->set_pattern("%v");
#endif
}

template<class R, class AO>
typename AuctionRunnerFR<R, AO>::Real
AuctionRunnerFR<R, AO>::get_cost_to_diagonal(const DgmPoint& pt) const
{
    if (1.0 == wasserstein_power) {
        return pt.persistence_lp(internal_p);
    } else {
        return std::pow(pt.persistence_lp(internal_p), wasserstein_power);
    }
}


template<class R, class AO>
typename AuctionRunnerFR<R, AO>::Real
AuctionRunnerFR<R, AO>::get_gamma() const
{
    if (1.0 == wasserstein_power) {
        return unassigned_items_persistence + unassigned_bidders_persistence;
    } else {
        return std::pow(unassigned_items_persistence + unassigned_bidders_persistence,
                        1.0 / wasserstein_power);
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::reset_phase_stat()
{
    num_rounds_non_cumulative = 0;
#ifdef LOG_AUCTION
    num_parallelizable_rounds = 0;
    num_parallelizable_forward_rounds = 0;
    num_parallelizable_reverse_rounds = 0;
    num_forward_rounds_non_cumulative = 0;
    num_reverse_rounds_non_cumulative = 0;
#endif
}


template<class R, class AO>
void AuctionRunnerFR<R, AO>::reset_round_stat()
{
    num_forward_bids_submitted = 0;
    num_reverse_bids_submitted = 0;
#ifdef LOG_AUCTION
    num_normal_forward_bids_submitted = 0;
    num_diag_forward_bids_submitted = 0;

    num_forward_diag_to_diag_assignments = 0;
    num_forward_diag_to_normal_assignments = 0;
    num_forward_normal_to_diag_assignments = 0;
    num_forward_normal_to_normal_assignments = 0;

    num_forward_diag_from_diag_thefts = 0;
    num_forward_diag_from_normal_thefts = 0;
    num_forward_normal_from_diag_thefts = 0;
    num_forward_normal_from_normal_thefts = 0;

    // reverse rounds
    num_normal_reverse_bids_submitted = 0;
    num_diag_reverse_bids_submitted = 0;

    num_reverse_diag_to_diag_assignments = 0;
    num_reverse_diag_to_normal_assignments = 0;
    num_reverse_normal_to_diag_assignments = 0;
    num_reverse_normal_to_normal_assignments = 0;

    num_reverse_diag_from_diag_thefts = 0;
    num_reverse_diag_from_normal_thefts = 0;
    num_reverse_normal_from_diag_thefts = 0;
    num_reverse_normal_from_normal_thefts = 0;
#endif
}


template<class R, class AO>
void AuctionRunnerFR<R, AO>::assign_forward(IdxType item_idx, IdxType bidder_idx)
{
    console_logger->debug("Enter assign_forward, item_idx = {0}, bidder_idx = {1}", item_idx, bidder_idx);
    sanity_check();
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

    if (unassigned_bidders.size() > parallel_threshold) {
        num_parallel_assignments++;
    }
    num_total_assignments++;


    int it_d = is_item_diagonal(item_idx);
    int b_d = is_bidder_diagonal(bidder_idx);
    // 2 - None
    int old_d = ( k_invalid_index == old_item_owner ) ? 2 : is_bidder_diagonal(old_item_owner);
    int key = 100 * old_d + 10 * b_d + it_d;
    switch(key) {
        case 211 : num_forward_diag_to_diag_assignments++;
                   break;
        case 210 : num_forward_diag_to_normal_assignments++;
                   break;
        case 201 : num_forward_normal_to_diag_assignments++;
                   break;
        case 200 : num_forward_normal_to_normal_assignments++;
                   break;

        case 111 : num_forward_diag_to_diag_assignments++;
                   num_forward_diag_from_diag_thefts++;
                   break;
        case 110 : num_forward_diag_to_normal_assignments++;
                   num_forward_diag_from_diag_thefts++;
                   break;
                   break;
        case 101 : num_forward_normal_to_diag_assignments++;
                   num_forward_normal_from_diag_thefts++;
                   break;
                   break;
        case 100 : num_forward_normal_to_normal_assignments++;
                   num_forward_normal_from_diag_thefts++;
                   break;

        case 11  : num_forward_diag_to_diag_assignments++;
                   num_forward_diag_from_normal_thefts++;
                   break;
        case 10  : num_forward_diag_to_normal_assignments++;
                   num_forward_diag_from_normal_thefts++;
                   break;
                   break;
        case 1   : num_forward_normal_to_diag_assignments++;
                   num_forward_normal_from_normal_thefts++;
                   break;
                   break;
        case 0   : num_forward_normal_to_normal_assignments++;
                   num_forward_normal_from_normal_thefts++;
                   break;
        default  : std::cerr << "key = " << key << std::endl;
                   throw std::runtime_error("Bug in logging, wrong key");
                   break;
    }
#endif

    sanity_check();
    console_logger->debug("Exit assign_forward, item_idx = {0}, bidder_idx = {1}", item_idx, bidder_idx);
}


template<class R, class AO>
void AuctionRunnerFR<R, AO>::assign_reverse(IdxType item_idx, IdxType bidder_idx)
{
    console_logger->debug("Enter assign_reverse, item_idx = {0}, bidder_idx = {1}", item_idx, bidder_idx);
    // only unassigned items submit bids in reverse phase
    assert(items_to_bidders[item_idx] == k_invalid_index);

    IdxType old_bidder_owner = bidders_to_items[bidder_idx];

    // set new owner
    bidders_to_items[bidder_idx] = item_idx;
    items_to_bidders[item_idx] = bidder_idx;

    // remove bidder and item from the sets of unassigned bidders/items
    remove_unassigned_item(item_idx);

    if (k_invalid_index != old_bidder_owner) {
        // old owner of item becomes unassigned
        items_to_bidders[old_bidder_owner] = k_invalid_index;
        add_unassigned_item(old_bidder_owner);
        // existing edge was removed, decrease partial_cost
        partial_cost -= get_item_bidder_cost(old_bidder_owner, bidder_idx);
    } else {
        // item was unassigned before
        remove_unassigned_bidder(bidder_idx);
    }

    // new edge was added to matching, increase partial cost
    partial_cost += get_item_bidder_cost(item_idx, bidder_idx);

#ifdef LOG_AUCTION
    if (unassigned_items.size() > parallel_threshold) {
        num_parallel_assignments++;
    }
    num_total_assignments++;

    int it_d = is_item_diagonal(item_idx);
    int b_d = is_bidder_diagonal(bidder_idx);
    // 2 - None
    int old_d = (k_invalid_index == old_bidder_owner) ? 2 : is_item_diagonal(old_bidder_owner);
    int key = 100 * old_d + 10 * it_d + b_d;
    switch(key) {
        case 211 : num_reverse_diag_to_diag_assignments++;
                   break;
        case 210 : num_reverse_diag_to_normal_assignments++;
                   break;
        case 201 : num_reverse_normal_to_diag_assignments++;
                   break;
        case 200 : num_reverse_normal_to_normal_assignments++;
                   break;

        case 111 : num_reverse_diag_to_diag_assignments++;
                   num_reverse_diag_from_diag_thefts++;
                   break;
        case 110 : num_reverse_diag_to_normal_assignments++;
                   num_reverse_diag_from_diag_thefts++;
                   break;
                   break;
        case 101 : num_reverse_normal_to_diag_assignments++;
                   num_reverse_normal_from_diag_thefts++;
                   break;
                   break;
        case 100 : num_reverse_normal_to_normal_assignments++;
                   num_reverse_normal_from_diag_thefts++;
                   break;

        case 11  : num_reverse_diag_to_diag_assignments++;
                   num_reverse_diag_from_normal_thefts++;
                   break;
        case 10  : num_reverse_diag_to_normal_assignments++;
                   num_reverse_diag_from_normal_thefts++;
                   break;
                   break;
        case 1   : num_reverse_normal_to_diag_assignments++;
                   num_reverse_normal_from_normal_thefts++;
                   break;
                   break;
        case 0   : num_reverse_normal_to_normal_assignments++;
                   num_reverse_normal_from_normal_thefts++;
                   break;
        default  : std::cerr << "key = " << key << std::endl;
                   throw std::runtime_error("Bug in logging, wrong key");
                   break;
    }

#endif
    console_logger->debug("Exit assign_reverse, item_idx = {0}, bidder_idx = {1}", item_idx, bidder_idx);
}

template<class R, class AO>
typename AuctionRunnerFR<R, AO>::Real
AuctionRunnerFR<R, AO>::get_item_bidder_cost(const size_t item_idx, const size_t bidder_idx) const
{
    if (wasserstein_power == 1.0) {
        return dist_lp(bidders[bidder_idx], items[item_idx], internal_p);
    } else {
        return std::pow(dist_lp(bidders[bidder_idx], items[item_idx], internal_p),
                            wasserstein_power);
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::assign_to_best_bidder(IdxType item_idx)
{
    console_logger->debug("Enter assign_to_best_bidder, item_idx = {0}", item_idx);
    assert( item_idx >= 0 and item_idx < static_cast<IdxType>(num_items) );
    assert( forward_bid_table[item_idx].first != k_invalid_index);

    auto best_bidder_idx = forward_bid_table[item_idx].first;
    auto best_bid_value = forward_bid_table[item_idx].second;
    assign_forward(item_idx, best_bidder_idx);
    forward_oracle.sanity_check();
    forward_oracle.set_price(item_idx,  best_bid_value, true);
    forward_oracle.sanity_check();
    auto new_bidder_price = -get_item_bidder_cost(item_idx, best_bidder_idx) - best_bid_value;
    reverse_oracle.set_price(best_bidder_idx, new_bidder_price, false);
    check_epsilon_css();
#ifdef LOG_AUCTION
    forward_price_change_cnt_vec.back()[item_idx]++;
    reverse_price_change_cnt_vec.back()[best_bidder_idx]++;
#endif
    console_logger->debug("Exit assign_to_best_bidder, item_idx = {0}", item_idx);
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::assign_to_best_item(IdxType bidder_idx)
{
    console_logger->debug("Enter assign_to_best_item, bidder_idx = {0}", bidder_idx);
    check_epsilon_css();
    assert( bidder_idx >= 0 and bidder_idx < static_cast<IdxType>(num_bidders) );
    assert( reverse_bid_table[bidder_idx].first != k_invalid_index);
    auto best_item_idx = reverse_bid_table[bidder_idx].first;
    auto best_bid_value = reverse_bid_table[bidder_idx].second;
    // both assign_forward and assign_reverse take item index first, bidder index second!
    assign_reverse(best_item_idx, bidder_idx);
    reverse_oracle.sanity_check();
    reverse_oracle.set_price(bidder_idx,  best_bid_value, true);
    reverse_oracle.sanity_check();
    auto new_item_price = -get_item_bidder_cost(best_item_idx, bidder_idx) - best_bid_value;
    forward_oracle.set_price(best_item_idx, new_item_price, false);
#ifdef LOG_AUCTION
    forward_price_change_cnt_vec.back()[best_item_idx]++;
    reverse_price_change_cnt_vec.back()[bidder_idx]++;
#endif
    check_epsilon_css();
    console_logger->debug("Exit assign_to_best_item, bidder_idx = {0}", bidder_idx);
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::clear_forward_bid_table()
{
    auto item_iter = items_with_bids.begin();
    while(item_iter != items_with_bids.end()) {
        auto item_with_bid_idx = *item_iter;
        forward_bid_table[item_with_bid_idx].first = k_invalid_index;
        forward_bid_table[item_with_bid_idx].second = k_lowest_bid_value;
        item_iter = items_with_bids.erase(item_iter);
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::clear_reverse_bid_table()
{
    auto bidder_iter = bidders_with_bids.begin();
    while(bidder_iter != bidders_with_bids.end()) {
        auto bidder_with_bid_idx = *bidder_iter;
        reverse_bid_table[bidder_with_bid_idx].first = k_invalid_index;
        reverse_bid_table[bidder_with_bid_idx].second = k_lowest_bid_value;
        bidder_iter = bidders_with_bids.erase(bidder_iter);
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::submit_forward_bid(IdxType bidder_idx, const IdxValPairR& bid)
{
    IdxType best_item_idx = bid.first;
    Real bid_value = bid.second;
    assert( best_item_idx >= 0 );

    auto value_in_bid_table = forward_bid_table[best_item_idx].second;
    bool new_bid_wins = (value_in_bid_table < bid_value);
    // if we have tie, lower persistence wins
//    if (value_in_bid_table == bid_value) {
//
//        assert(forward_bid_table.at(best_item_idx).first != k_invalid_index);
//        assert(&bidders.at(forward_bid_table.at(best_item_idx).first));
//
//        auto bidder_in_bid_table = bidders[forward_bid_table[best_item_idx].first];
//        new_bid_wins = bidders[best_item_idx].persistence_lp(internal_p) < bidder_in_bid_table.persistence_lp(internal_p);
//    }

    if (new_bid_wins) {
        forward_bid_table[best_item_idx].first = bidder_idx;
        forward_bid_table[best_item_idx].second = bid_value;
    }

    items_with_bids.insert(best_item_idx);

#ifdef LOG_AUCTION

    if (unassigned_bidders.size() > parallel_threshold) {
        num_parallel_bids++;
    }
    num_total_bids++;


    if (is_bidder_diagonal(bidder_idx)) {
        num_diag_forward_bids_submitted++;
    } else {
        num_normal_forward_bids_submitted++;
    }
#endif
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::submit_reverse_bid(IdxType item_idx, const IdxValPairR& bid)
{
    assert( items.at(item_idx).is_diagonal() or items.at(item_idx).is_normal() );
    IdxType best_bidder_idx = bid.first;
    assert( bidders.at(best_bidder_idx).is_diagonal() or bidders.at(best_bidder_idx).is_normal() );
    Real bid_value = bid.second;
    assert(bid_value > k_lowest_bid_value);
    auto value_in_bid_table = reverse_bid_table[best_bidder_idx].second;
    bool new_bid_wins = (value_in_bid_table < bid_value);
    // if we have tie, lower persistence wins
//    if (value_in_bid_table == bid_value) {
//        assert(reverse_bid_table[best_bidder_idx].first != k_invalid_index);
//        auto bidder_in_bid_table = bidders[reverse_bid_table[best_bidder_idx].first];
//        new_bid_wins = bidders[best_bidder_idx].persistence_lp(internal_p) < bidder_in_bid_table.persistence_lp(internal_p);
//    }
    if (new_bid_wins) {
        reverse_bid_table[best_bidder_idx].first = item_idx;
        reverse_bid_table[best_bidder_idx].second = bid_value;
    }
    bidders_with_bids.insert(best_bidder_idx);

#ifdef LOG_AUCTION

    if (unassigned_items.size() > parallel_threshold) {
        num_parallel_bids++;
    }
    num_total_bids++;

    if (is_item_diagonal(item_idx)) {
        num_diag_reverse_bids_submitted++;
    } else {
        num_normal_reverse_bids_submitted++;
    }
#endif
}


template<class R, class AO>
void AuctionRunnerFR<R, AO>::print_debug()
{
#ifdef DEBUG_FR_AUCTION
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
    std::cout << "Bidder prices: " << std::endl;
    for(const auto price : forward_oracle.get_prices()) {
        std::cout << price << std::endl;
    }
    std::cout << "**********************" << std::endl;
#endif
}


template<class R, class AO>
typename AuctionRunnerFR<R,AO>::Real
AuctionRunnerFR<R, AO>::get_relative_error(const bool debug_output) const
{
    Real result;
    Real gamma = get_gamma();
    // cost minus n epsilon
    Real reduced_cost = partial_cost - num_bidders * get_epsilon();
    if ( reduced_cost < 0) {
#ifdef LOG_AUCTION
        if (debug_output) {
            console_logger->debug("Epsilon too large, reduced_cost = {0}", reduced_cost);
        }
#endif
        result = k_max_relative_error;
    } else {
        Real denominator = std::pow(reduced_cost, 1.0 / wasserstein_power) - gamma;
        if (denominator <= 0) {
#ifdef LOG_AUCTION
            if (debug_output) {
                console_logger->debug("Epsilon too large, reduced_cost = {0}, denominator = {1}, gamma = {2}", reduced_cost, denominator, gamma);
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
                console_logger->debug("Reduced_cost = {0}, denominator = {1}, numerator {2}, error = {3},  gamma = {4}",
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

template<class R, class AO>
void AuctionRunnerFR<R, AO>::flush_assignment()
{
    console_logger->debug("Enter flush_assignment");
    for(auto& b2i : bidders_to_items) {
        b2i = k_invalid_index;
    }
    for(auto& i2b : items_to_bidders) {
        i2b = k_invalid_index;
    }

    // all bidders and items become unassigned
    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        unassigned_bidders.insert(bidder_idx);
    }

    // all items and items become unassigned
    for(size_t item_idx = 0; item_idx < num_items; ++item_idx) {
        unassigned_items.insert(item_idx);
    }


    //forward_oracle.adjust_prices();
    //reverse_oracle.adjust_prices();

    partial_cost = 0.0;
    unassigned_bidders_persistence = total_bidders_persistence;
    unassigned_items_persistence = total_items_persistence;

#ifdef ORDERED_BY_PERSISTENCE
    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        unassigned_bidders_by_persistence.insert(std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx));
    }

    for(size_t item_idx = 0; item_idx < num_items; ++item_idx) {
        unassigned_items_by_persistence.insert(std::make_pair(items[item_idx].persistence_lp(1.0), item_idx));
    }
#endif

#ifdef LOG_AUCTION

    reset_phase_stat();

    forward_price_change_cnt_vec.push_back(std::vector<size_t>(num_items, 0));
    reverse_price_change_cnt_vec.push_back(std::vector<size_t>(num_bidders, 0));

    // all bidders and items become unassigned
    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        if (is_bidder_normal(bidder_idx)) {
            unassigned_normal_bidders.insert(bidder_idx);
        } else {
            unassigned_diag_bidders.insert(bidder_idx);
        }
    }

    never_assigned_bidders = unassigned_bidders;

    for(size_t item_idx = 0; item_idx < items.size(); ++item_idx) {
        if (is_item_normal(item_idx)) {
            unassigned_normal_items.insert(item_idx);
        } else {
            unassigned_diag_items.insert(item_idx);
        }
    }

    never_assigned_items = unassigned_items;
#endif
    check_epsilon_css();
    console_logger->debug("Exit flush_assignment");
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::set_epsilon(Real new_val)
{
    assert(new_val > 0.0);
    epsilon = new_val;
    forward_oracle.set_epsilon(new_val);
    reverse_oracle.set_epsilon(new_val);
}


template<class R, class AO>
bool AuctionRunnerFR<R, AO>::continue_forward(const size_t original_unassigned_bidders, const size_t min_forward_matching_increment)
{
//    if (unassigned_threshold == 0) {
//        return not unassigned_bidders.empty() and get_relative_error(false) > delta;
//    }
    //return unassigned_bidders.size() > unassigned_threshold and
           //static_cast<int>(unassigned_bidders.size()) >= static_cast<int>(original_unassigned_bidders) - static_cast<int>(min_forward_matching_increment);
    return unassigned_bidders.size() > unassigned_threshold and
           static_cast<int>(unassigned_bidders.size()) >= static_cast<int>(original_unassigned_bidders) - static_cast<int>(min_forward_matching_increment) and
           get_relative_error() >= delta;
//    return not unassigned_bidders.empty() and
//           static_cast<int>(unassigned_bidders.size()) >= static_cast<int>(original_unassigned_bidders) - static_cast<int>(min_forward_matching_increment) and
//           get_relative_error() >= delta;
}


template<class R, class AO>
bool AuctionRunnerFR<R, AO>::continue_reverse(const size_t original_unassigned_items, const size_t min_reverse_matching_increment)
{
    //return unassigned_items.size() > unassigned_threshold and
           //static_cast<int>(unassigned_items.size()) >= static_cast<int>(original_unassigned_items) - static_cast<int>(min_reverse_matching_increment);
    return unassigned_items.size() > unassigned_threshold and
           static_cast<int>(unassigned_items.size()) >= static_cast<int>(original_unassigned_items) - static_cast<int>(min_reverse_matching_increment) and
           get_relative_error() >= delta;
//    return not unassigned_items.empty() and
//           static_cast<int>(unassigned_items.size()) >= static_cast<int>(original_unassigned_items) - static_cast<int>(min_reverse_matching_increment) and
//           get_relative_error() >= delta;
}


template<class R, class AO>
bool AuctionRunnerFR<R, AO>::continue_phase()
{
    //return not unassigned_bidders.empty();
    return unassigned_bidders.size() > unassigned_threshold and get_relative_error() >= delta;
//    return not never_assigned_bidders.empty() or
//           not never_assigned_items.empty() or
//           unassigned_bidders.size() > unassigned_threshold and get_relative_error() >= delta;
}



template<class R, class AO>
void AuctionRunnerFR<R, AO>::run_auction_phase()
{
    num_phase++;
    while(continue_phase()) {
        forward_oracle.recompute_top_diag_items(true);
        forward_oracle.sanity_check();
        console_logger->debug("forward_oracle recompute_top_diag_items done");
        run_forward_auction_phase();
        reverse_oracle.recompute_top_diag_items(true);
        console_logger->debug("reverse_oracle recompute_top_diag_items done");
        reverse_oracle.sanity_check();
        run_reverse_auction_phase();
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::run_auction_phases(const int max_num_phases, const Real _initial_epsilon)
{
    set_epsilon(_initial_epsilon);
    assert( forward_oracle.get_epsilon() > 0 );
    assert( reverse_oracle.get_epsilon() > 0 );
    for(int phase_num = 0; phase_num < max_num_phases; ++phase_num) {
        flush_assignment();
        console_logger->info("Phase {0} started: eps = {1}",
                              num_phase,
                              get_epsilon());

        run_auction_phase();
        Real current_result = partial_cost;
#ifdef LOG_AUCTION
        console_logger->info("Phase {0} done: current_result = {1}, eps = {2}, unassigned_threshold = {3}, unassigned = {4}, error = {5}, gamma = {6}",
                              num_phase,
                              partial_cost,
                              get_epsilon(),
                              format_int<>(unassigned_threshold),
                              unassigned_bidders.size(),
                              get_relative_error(false),
                              get_gamma());

        console_logger->info("Phase {0} done: num_rounds / num_parallelizable_rounds = {1} / {2} = {3}, cumulative rounds = {4}",
                              num_phase,
                              format_int(num_rounds_non_cumulative),
                              format_int(num_parallelizable_rounds),
                              static_cast<double>(num_parallelizable_rounds) / static_cast<double>(num_rounds_non_cumulative),
                              format_int(num_rounds)
                              );

        console_logger->info("parallelizable_forward_rounds / num_forward_rounds = {0} / {1} = {2}",
                         format_int<>(num_parallelizable_forward_rounds),
                         format_int<>(num_forward_rounds_non_cumulative),
                         static_cast<double>(num_parallelizable_forward_rounds) / static_cast<double>(num_forward_rounds_non_cumulative)
                        );

        num_parallelizable_forward_rounds = 0;
        num_forward_rounds_non_cumulative = 0;

        console_logger->info("parallelizable_reverse_rounds / num_reverse_rounds = {0} / {1} = {2}",
                         format_int<>(num_parallelizable_reverse_rounds),
                         format_int<>(num_reverse_rounds_non_cumulative),
                         static_cast<double>(num_parallelizable_reverse_rounds) / static_cast<double>(num_reverse_rounds_non_cumulative)
                        );

        num_parallelizable_reverse_rounds = 0;
        num_reverse_rounds_non_cumulative = 0;

        console_logger->info("num_parallel_bids / num_total_bids = {0} / {1} = {2}, num_parallel_assignments / num_total_assignments = {3} / {4} = {5}",
                         format_int<>(num_parallel_bids),
                         format_int<>(num_total_bids),
                         static_cast<double>(num_parallel_bids) / static_cast<double>(num_total_bids),
                         format_int<>(num_parallel_assignments),
                         format_int<>(num_total_assignments),
                         static_cast<double>(num_parallel_assignments) / static_cast<double>(num_total_assignments)
                        );

        auto forward_min_max_price = forward_oracle.get_minmax_price();
        auto reverse_min_max_price = reverse_oracle.get_minmax_price();

        console_logger->info("forward min price = {0}, max price = {1}; reverse min price = {2}, reverse max price = {3}",
                             forward_min_max_price.first,
                             forward_min_max_price.second,
                             reverse_min_max_price.first,
                             reverse_min_max_price.second
                             );

        for(size_t item_idx = 0; item_idx < num_items; ++item_idx) {
            forward_price_stat_logger->info("{0} {1} {2} {3} {4}",
                                            phase_num,
                                            item_idx,
                                            items[item_idx].getRealX(),
                                            items[item_idx].getRealY(),
                                            forward_price_change_cnt_vec.back()[item_idx]
                                           );
        }

        for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
            reverse_price_stat_logger->info("{0} {1} {2} {3} {4}",
                                            phase_num,
                                            bidder_idx,
                                            bidders[bidder_idx].getRealX(),
                                            bidders[bidder_idx].getRealY(),
                                            reverse_price_change_cnt_vec.back()[bidder_idx]
                                           );
        }
#endif

        if (get_relative_error(true) <= delta) {
            break;
        }
        // decrease epsilon for the next iteration
        decrease_epsilon();

        unassigned_threshold = std::floor( static_cast<double>(unassigned_threshold) / 1.1 );

        if (phase_can_be_final()) {
            unassigned_threshold = 0;
#ifdef LOG_AUCTION
            console_logger->info("Unassigned threshold set to zero!");
#endif
        }
    }
}

template<class R, class AO>
bool AuctionRunnerFR<R, AO>::phase_can_be_final() const
{
    Real estimated_error;
    // cost minus n epsilon
    Real reduced_cost = partial_cost - num_bidders * get_epsilon();
    if (reduced_cost <= 0.0) {
        return false;
    } else {
        Real denominator = std::pow(reduced_cost, 1.0 / wasserstein_power);
        if (denominator <= 0) {
            return false;
        } else {
            Real numerator = std::pow(partial_cost, 1.0 / wasserstein_power) -
                             std::pow(reduced_cost, 1.0 / wasserstein_power);

            estimated_error = numerator / denominator;
            return estimated_error <= delta;
        }
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::run_auction()
{
    double init_eps = ( initial_epsilon > 0.0 ) ? initial_epsilon : std::min(forward_oracle.max_val_, reverse_oracle.max_val_) / 4.0 ;
    assert(init_eps > 0.0);
    run_auction_phases(max_num_phases, init_eps);
    is_distance_computed = true;
    wasserstein_cost = partial_cost;
    if (get_relative_error() > delta) {
#ifndef FOR_R_TDA
            std::cerr << "Maximum iteration number exceeded, exiting. Current result is: ";
            std::cerr << get_wasserstein_distance() << std::endl;
#endif
            throw std::runtime_error("Maximum iteration number exceeded");
    }
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::add_unassigned_bidder(const size_t bidder_idx)
{
    const DgmPoint& bidder = bidders[bidder_idx];
    unassigned_bidders.insert(bidder_idx);
    unassigned_bidders_persistence += get_cost_to_diagonal(bidder);

#ifdef ORDERED_BY_PERSISTENCE
    unassigned_bidders_by_persistence.insert(std::make_pair(bidder.persistence_lp(1.0), bidder_idx));
#endif

#ifdef LOG_AUCTION
    if (is_bidder_diagonal(bidder_idx)) {
        unassigned_diag_bidders.insert(bidder_idx);
    } else {
        unassigned_normal_bidders.insert(bidder_idx);
    }
#endif
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::add_unassigned_item(const size_t item_idx)
{
    const DgmPoint& item = items[item_idx];
    unassigned_items.insert(item_idx);
    unassigned_items_persistence += get_cost_to_diagonal(item);

#ifdef ORDERED_BY_PERSISTENCE
    unassigned_items_by_persistence.insert(std::make_pair(item.persistence_lp(1.0), item_idx));
#endif

#ifdef LOG_AUCTION
    if (is_item_diagonal(item_idx)) {
        unassigned_diag_items.insert(item_idx);
    } else {
        unassigned_normal_items.insert(item_idx);
    }
#endif
}


template<class R, class AO>
void AuctionRunnerFR<R, AO>::remove_unassigned_bidder(const size_t bidder_idx)
{
    unassigned_bidders_persistence -= get_cost_to_diagonal(bidders[bidder_idx]);

    unassigned_bidders.erase(bidder_idx);
    never_assigned_bidders.erase(bidder_idx);

#ifdef ORDERED_BY_PERSISTENCE
    unassigned_bidders_by_persistence.erase(std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx));
#endif

#ifdef LOG_AUCTION
    if (is_bidder_diagonal(bidder_idx)) {
        unassigned_diag_bidders.erase(bidder_idx);
    } else {
        unassigned_normal_bidders.erase(bidder_idx);
    }
    if (never_assigned_bidders.empty() and not all_assigned_round_found) {
        all_assigned_round = num_rounds_non_cumulative;
        all_assigned_round_found = true;
    }
#endif
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::remove_unassigned_item(const size_t item_idx)
{
    console_logger->debug("Enter remove_unassigned_item, unassigned_items.size = {0}", unassigned_items.size());
    unassigned_items_persistence -= get_cost_to_diagonal(items[item_idx]);

    never_assigned_items.erase(item_idx);
    unassigned_items.erase(item_idx);

#ifdef ORDERED_BY_PERSISTENCE
    unassigned_items_by_persistence.erase(std::make_pair(items[item_idx].persistence_lp(1.0), item_idx));
#endif

#ifdef LOG_AUCTION
    if (is_item_normal(item_idx)) {
        unassigned_normal_items.erase(item_idx);
    } else {
        unassigned_diag_items.erase(item_idx);
    }
    if (never_assigned_items.empty() and not all_assigned_round_found) {
        all_assigned_round = num_rounds_non_cumulative;
        all_assigned_round_found = true;
    }
#endif
    console_logger->debug("Exit remove_unassigned_item, unassigned_items.size = {0}", unassigned_items.size());
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::decrease_epsilon()
{
    auto eps_diff = 1.01 * get_epsilon() * (epsilon_common_ratio - 1.0 ) / epsilon_common_ratio;
    reverse_oracle.adjust_prices( -eps_diff );
    set_epsilon( get_epsilon() / epsilon_common_ratio );
    cumulative_epsilon_factor *= epsilon_common_ratio;
}



template<class R, class AO>
void AuctionRunnerFR<R, AO>::run_reverse_auction_phase()
{
    console_logger->debug("Enter run_reverse_auction_phase");
    size_t original_unassigned_items = unassigned_items.size();
//    const size_t min_reverse_matching_increment = std::max( static_cast<size_t>(1), static_cast<size_t>(original_unassigned_items / 10));
    size_t min_reverse_matching_increment = 1;

    while (continue_reverse(original_unassigned_items, min_reverse_matching_increment)) {
        num_rounds++;
        num_rounds_non_cumulative++;
        console_logger->debug("started round = {0}, reverse, unassigned = {1}", num_rounds, unassigned_items.size());

        check_epsilon_css();
#ifdef LOG_AUCTION
        if (unassigned_items.size() >= parallel_threshold) {
            ++num_parallelizable_reverse_rounds;
            ++num_parallelizable_rounds;
        }
        num_reverse_rounds++;
        num_reverse_rounds_non_cumulative++;
#endif

        reset_round_stat();
        // bidding
#ifdef ORDERED_BY_PERSISTENCE
        std::vector<size_t> active_items;
        active_items.reserve(batch_size);
        for(auto iter = unassigned_items_by_persistence.begin();
                iter != unassigned_items_by_persistence.end(); ++iter) {
            active_items.push_back(iter->second);
            if (active_items.size() >= batch_size) {
                break;
            }
        }
        run_reverse_bidding_step(active_items);
#else
        //if (not never_assigned_items.empty())
            //run_reverse_bidding_step(never_assigned_items);
        //else
            //run_reverse_bidding_step(unassigned_items);
        run_reverse_bidding_step(unassigned_items);
#endif

        // assignment phase
        for(auto bidder_idx : bidders_with_bids ) {
            assign_to_best_item(bidder_idx);
        }

        check_epsilon_css();

        console_logger->debug("ended round = {0}, reverse, unassigned = {1}", num_rounds, unassigned_items.size());

#ifdef LOG_AUCTION

        reverse_plot_logger->info("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20} {21} {22}",
                                  num_phase,
                                  num_rounds,
                                  num_reverse_rounds,
                                  unassigned_bidders.size(),
                                  get_gamma(),
                                  partial_cost,
                                  reverse_oracle.get_epsilon(),
                                  num_normal_reverse_bids_submitted,
                                  num_diag_reverse_bids_submitted,
                                  num_reverse_diag_to_diag_assignments,
                                  num_reverse_diag_to_normal_assignments,
                                  num_reverse_normal_to_diag_assignments,
                                  num_reverse_normal_to_normal_assignments,
                                  num_reverse_diag_from_diag_thefts,
                                  num_reverse_diag_from_normal_thefts,
                                  num_reverse_normal_from_diag_thefts,
                                  num_reverse_normal_from_normal_thefts,
                                  unassigned_normal_bidders.size(),
                                  unassigned_diag_bidders.size(),
                                  unassigned_normal_items.size(),
                                  unassigned_diag_items.size(),
                                  reverse_oracle.get_heap_top_size(),
                                  get_relative_error(false)
                                 );
        sanity_check();
#endif
    }
}

template<class R, class AO>
template<class Range>
void AuctionRunnerFR<R, AO>::run_forward_bidding_step(const Range& active_bidders)
{
    clear_forward_bid_table();
    for(const auto bidder_idx : active_bidders) {
        console_logger->debug("current bidder (forward): {0}, persistence = {1}", bidders[bidder_idx], bidders[bidder_idx].persistence_lp(1.0));
        submit_forward_bid(bidder_idx, forward_oracle.get_optimal_bid(bidder_idx));
        if (++num_forward_bids_submitted >= max_bids_per_round) {
            break;
        }
    }
}

template<class R, class AO>
template<class Range>
void AuctionRunnerFR<R, AO>::run_reverse_bidding_step(const Range& active_items)
{
    clear_reverse_bid_table();

    assert(bidders_with_bids.empty());
    assert(std::all_of(reverse_bid_table.begin(), reverse_bid_table.end(),
                [ki = k_invalid_index, kl = k_lowest_bid_value](const IdxValPairR& b) { return static_cast<size_t>(b.first) == ki and b.second == kl; }));

    for(const auto item_idx : active_items) {
        console_logger->debug("current bidder (reverse): {0}, persistence = {1}", items[item_idx], items[item_idx].persistence_lp(1.0));
        submit_reverse_bid(item_idx, reverse_oracle.get_optimal_bid(item_idx));
        if (++num_reverse_bids_submitted >= max_bids_per_round) {
            break;
        }
    }
}


template<class R, class AO>
void AuctionRunnerFR<R, AO>::run_forward_auction_phase()
{
    const size_t original_unassigned_bidders = unassigned_bidders.size();
//    const size_t min_forward_matching_increment = std::max( static_cast<size_t>(1), static_cast<size_t>(original_unassigned_bidders / 10));
    const size_t min_forward_matching_increment = 1;
    while (continue_forward(original_unassigned_bidders, min_forward_matching_increment)) {
        console_logger->debug("started round = {0}, forward, unassigned = {1}", num_rounds, unassigned_bidders.size());
        check_epsilon_css();
        num_rounds++;
#ifdef LOG_AUCTION
        if (unassigned_bidders.size() >= parallel_threshold) {
            ++num_parallelizable_forward_rounds;
            ++num_parallelizable_rounds;
        }
        num_forward_rounds++;
        num_forward_rounds_non_cumulative++;
#endif

        reset_round_stat();
        // bidding step
#ifdef ORDERED_BY_PERSISTENCE
        std::vector<size_t> active_bidders;
        active_bidders.reserve(batch_size);
        for(auto iter = unassigned_bidders_by_persistence.begin();
                iter != unassigned_bidders_by_persistence.end(); ++iter) {
            active_bidders.push_back(iter->second);
            if (active_bidders.size() >= batch_size) {
                break;
            }
        }
        run_forward_bidding_step(active_bidders);
#else

        //if (not never_assigned_bidders.empty())
            //run_forward_bidding_step(never_assigned_bidders);
        //else
            //run_forward_bidding_step(unassigned_bidders);
        run_forward_bidding_step(unassigned_bidders);
#endif

        // assignment step
        for(auto item_idx : items_with_bids ) {
            assign_to_best_bidder(item_idx);
        }

        console_logger->debug("ended round = {0}, forward, unassigned = {1}", num_rounds, unassigned_bidders.size());
        check_epsilon_css();

#ifdef LOG_AUCTION
        forward_plot_logger->info("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20} {21} {22}",
                                  num_phase,
                                  num_rounds,
                                  num_forward_rounds,
                                  unassigned_bidders.size(),
                                  get_gamma(),
                                  partial_cost,
                                  forward_oracle.get_epsilon(),
                                  num_normal_forward_bids_submitted,
                                  num_diag_forward_bids_submitted,
                                  num_forward_diag_to_diag_assignments,
                                  num_forward_diag_to_normal_assignments,
                                  num_forward_normal_to_diag_assignments,
                                  num_forward_normal_to_normal_assignments,
                                  num_forward_diag_from_diag_thefts,
                                  num_forward_diag_from_normal_thefts,
                                  num_forward_normal_from_diag_thefts,
                                  num_forward_normal_from_normal_thefts,
                                  unassigned_normal_bidders.size(),
                                  unassigned_diag_bidders.size(),
                                  unassigned_normal_items.size(),
                                  unassigned_diag_items.size(),
                                  forward_oracle.get_heap_top_size(),
                                  get_relative_error(false)
                                 );
#endif
    } ;

}
template<class R, class AO>
void AuctionRunnerFR<R, AO>::assign_diag_to_diag()
{
    size_t n_diag_to_diag = std::min(num_diag_bidders, num_diag_items);
    if (n_diag_to_diag < 2)
        return;
    for(size_t i = 0; i < n_diag_to_diag; ++i) {
    }
}

template<class R, class AO>
typename AuctionRunnerFR<R, AO>::Real
AuctionRunnerFR<R, AO>::get_wasserstein_distance()
{
    assert(is_distance_computed);
    return std::pow(wasserstein_cost, 1.0 / wasserstein_power);
}

template<class R, class AO>
typename AuctionRunnerFR<R, AO>::Real
AuctionRunnerFR<R, AO>::get_wasserstein_cost()
{
    assert(is_distance_computed);
    return wasserstein_cost;
}



template<class R, class AO>
void AuctionRunnerFR<R, AO>::sanity_check()
{
#ifdef DEBUG_FR_AUCTION
    assert(partial_cost >= 0);


    assert(num_diag_items == num_normal_bidders);
    assert(num_diag_bidders == num_normal_items);
    assert(num_diag_bidders + num_normal_bidders == num_bidders);
    assert(num_diag_items + num_normal_items == num_items);
    assert(num_items == num_bidders);


    for(size_t b = 0; b < num_bidders; ++b) {
        assert( is_bidder_diagonal(b) == bidders.at(b).is_diagonal() );
        assert( is_bidder_normal(b) == bidders.at(b).is_normal() );
    }

    for(size_t i = 0; i < num_items; ++i) {
        assert( is_item_diagonal(i) == items.at(i).is_diagonal() );
        assert( is_item_normal(i) == items.at(i).is_normal() );
    }

    // check matching consistency
    assert(bidders_to_items.size() == num_bidders);
    assert(items_to_bidders.size() == num_bidders);

    assert(std::count(bidders_to_items.begin(), bidders_to_items.end(), k_invalid_index)  == std::count(items_to_bidders.begin(), items_to_bidders.end(), k_invalid_index));

    Real true_partial_cost = 0.0;

    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        if (bidders_to_items[bidder_idx] != k_invalid_index) {
            assert(items_to_bidders.at(bidders_to_items[bidder_idx]) == static_cast<int>(bidder_idx));
            true_partial_cost += get_item_bidder_cost(bidders_to_items[bidder_idx], bidder_idx);
        }
    }

    assert(fabs(partial_cost - true_partial_cost) < 0.00001);

    for(size_t item_idx = 0; item_idx < num_items; ++item_idx) {
        if (items_to_bidders[item_idx] != k_invalid_index) {
            assert(bidders_to_items.at(items_to_bidders[item_idx]) == static_cast<int>(item_idx));
        }
    }

#ifdef ORDERED_BY_PERSISTENCE
    assert(unassigned_bidders.size() == unassigned_bidders_by_persistence.size());
    if (unassigned_items.size() != unassigned_items_by_persistence.size()) {
        console_logger->error("unassigned_items.size() = {0}, unassigned_items_by_persistence.size() = {1}",  unassigned_items.size(),unassigned_items_by_persistence.size());
        console_logger->error("unassigned_items = {0}, unassigned_items_by_persistence = {1}",  format_container_to_log(unassigned_items),format_pair_container_to_log(unassigned_items_by_persistence));
    }
    assert(unassigned_items.size() == unassigned_items_by_persistence.size());

    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        if (bidders_to_items[bidder_idx] == k_invalid_index) {
            assert(unassigned_bidders.count(bidder_idx) == 1);
            assert(unassigned_bidders_by_persistence.count(std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx)) == 1);
        } else {
            assert(unassigned_bidders.count(bidder_idx) == 0);
            assert(unassigned_bidders_by_persistence.count(std::make_pair(bidders[bidder_idx].persistence_lp(1.0), bidder_idx)) == 0);
        }
    }

    for(size_t item_idx = 0; item_idx < num_items; ++item_idx) {
        if (items_to_bidders[item_idx] == k_invalid_index) {
            assert(unassigned_items.count(item_idx) == 1);
            assert(unassigned_items_by_persistence.count(std::make_pair(items[item_idx].persistence_lp(1.0), item_idx)) == 1);
        } else {
            assert(unassigned_items.count(item_idx) == 0);
            assert(unassigned_items_by_persistence.count(std::make_pair(items[item_idx].persistence_lp(1.0), item_idx)) == 0);
        }
    }
#endif


#endif
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::check_epsilon_css()
{
#ifdef DEBUG_FR_AUCTION
    sanity_check();

    std::vector<double> b_prices = reverse_oracle.get_prices();
    std::vector<double> i_prices = forward_oracle.get_prices();
    double eps = forward_oracle.get_epsilon();

    for(size_t b = 0; b < num_bidders; ++b) {
        for(size_t i = 0; i < num_items; ++i) {
            if(((is_bidder_normal(b) and is_item_diagonal(i)) or (is_bidder_diagonal(b) and is_item_normal(i))) and b != i)
                continue;
            if (b_prices[b] + i_prices[i] + eps < -get_item_bidder_cost(i, b) - 0.000001) {
                console_logger->debug("b = {0}, i = {1}, eps = {2}, b_price = {3}, i_price[i] = {4}, cost = {5}, b_price + i_price + eps = {6}",
                                       b,
                                       i,
                                       eps,
                                       b_prices[b],
                                       i_prices[i],
                                       get_item_bidder_cost(i, b),
                                       b_prices[b] + i_prices[i] + eps
                                      );
            }
            assert(b_prices[b] + i_prices[i] + eps >= -get_item_bidder_cost(i, b) - 0.000001);
        }
    }

    for(size_t b = 0; b < num_bidders; ++b) {
        auto i = bidders_to_items[b];
        if (i != k_invalid_index) {
            assert( fabs(b_prices[b] + i_prices[i] + get_item_bidder_cost(i, b)) < 0.000001 );
        }
    }
#endif
}

template<class R, class AO>
void AuctionRunnerFR<R, AO>::print_matching()
{
#ifdef DEBUG_FR_AUCTION
    sanity_check();
    for(size_t bidder_idx = 0; bidder_idx < bidders_to_items.size(); ++bidder_idx) {
        if (bidders_to_items[bidder_idx] >= 0) {
            auto pA = bidders[bidder_idx];
            auto pB = items[bidders_to_items[bidder_idx]];
            std::cout <<  pA << " <-> " << pB << "+" << pow(dist_lp(pA, pB, internal_p), wasserstein_power) << std::endl;
        } else {
            assert(false);
        }
    }
#endif
}

} // ws
} // hera

#endif
