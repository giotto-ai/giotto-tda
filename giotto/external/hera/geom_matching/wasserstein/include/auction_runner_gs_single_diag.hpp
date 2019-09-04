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


#include <assert.h>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <iterator>
#include <chrono>

#include "def_debug_ws.h"

#define PRINT_DETAILED_TIMING

#ifdef FOR_R_TDA
#include "Rcpp.h"
#undef DEBUG_AUCTION
#endif


namespace hera {
namespace ws {

// *****************************
// AuctionRunnerGaussSeidelSingleDiag
// *****************************

template<class R, class AO>
std::ostream& operator<<(std::ostream& s, const AuctionRunnerGaussSeidelSingleDiag<R, AO>& ar)
{
    s << "--------------------------------------------------\n";
    s << "AuctionRunnerGaussSeidelSingleDiag, current assignment, bidders_to_items:" << std::endl;
    for(size_t idx = 0; idx < ar.bidders_to_items.size(); ++idx) {
        s << idx << " <--> " << ar.bidders_to_items[idx] << std::endl;
    }
    s << "--------------------------------------------------\n";
    s << "AuctionRunnerGaussSeidelSingleDiag, current assignment, items_to_bidders:" << std::endl;
    for(size_t idx = 0; idx < ar.items_to_bidders.size(); ++idx) {
        s << idx << " <--> " << ar.items_to_bidders[idx] << std::endl;
    }
    s << "--------------------------------------------------\n";
    s << "AuctionRunnerGaussSeidelSingleDiag, prices:" << std::endl;
    for(size_t item_idx = 0; item_idx < ar.num_items; ++item_idx) {
        s << item_idx << ": " << ar.oracle->get_price(item_idx) << std::endl;
    }
    s << "--------------------------------------------------\n";
    s << "AuctionRunnerGaussSeidelSingleDiag, oracle :" << *(ar.oracle) << std::endl;
    s << "--------------------------------------------------\n";
    return s;
}


template<class R, class AO>
AuctionRunnerGaussSeidelSingleDiag<R, AO>::AuctionRunnerGaussSeidelSingleDiag(const DgmPointVec& A,
                                        const DgmPointVec& B,
                                        const Real q,
                                        const Real _delta,
                                        const Real _internal_p,
                                        const Real _initial_epsilon,
                                        const Real _eps_factor,
                                        const int _max_iter_num) :
    bidders(A),
    items(B),
    num_bidders(A.size()),
    num_items(B.size()),
    items_to_bidders(B.size(), k_invalid_index),
    bidders_to_items(A.size(), k_invalid_index),
    wasserstein_power(q),
    delta(_delta),
    internal_p(_internal_p),
    initial_epsilon(_initial_epsilon),
    epsilon_common_ratio(_eps_factor == 0.0 ? 5.0 : _eps_factor),
    max_iter_num(_max_iter_num)
#ifdef LOG_AUCTION
    , total_items_persistence(std::accumulate(items.begin(),
                                            items.end(),
                                            R(0.0),
                                            [_internal_p, q](const Real& ps, const DgmPoint& item)
                                                { return ps + std::pow(item.persistence_lp(_internal_p), q); }
                                           ))

    , total_bidders_persistence(std::accumulate(bidders.begin(),
                                              bidders.end(),
                                              R(0.0),
                                              [_internal_p, q](const Real& ps, const DgmPoint& bidder)
                                                  { return ps + std::pow(bidder.persistence_lp(_internal_p), q); }
                                             ))
    , partial_cost(0.0)
    , unassigned_bidders_persistence(0.0)
    , unassigned_items_persistence(0.0)
#endif

{
    assert(initial_epsilon >= 0.0 );
    assert(epsilon_common_ratio >= 0.0 );
    assert(A.size() == B.size());
    oracle = std::unique_ptr<AuctionOracle>(new AuctionOracle(bidders, items, wasserstein_power, internal_p));

    for(num_normal_bidders = 0; num_normal_bidders < num_bidders; ++num_normal_bidders) {
        if (bidders[num_normal_bidders].is_diagonal())
            break;
    }

    num_diag_bidders = num_bidders - num_normal_bidders;
    num_diag_items = num_normal_bidders;
    num_normal_items = num_items - num_diag_items;

    for(size_t i = num_normal_bidders; i < num_bidders; ++i) {
        assert(bidders[i].is_diagonal());
    }

#ifdef LOG_AUCTION

    unassigned_items_persistence = total_items_persistence;
    unassigned_bidders_persistence = total_bidders_persistence;

    if (not spdlog::get("plot_logger")) {
        auto log = spdlog::basic_logger_st("plot_logger", "plot_logger.txt");
        log->info("New plot starts here");
        log->set_pattern("%v");
    }
#endif

}

#ifdef LOG_AUCTION
template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::enable_logging(const char* log_filename, const size_t _max_unassigned_to_log)
{
    log_auction = true;
    max_unassigned_to_log = _max_unassigned_to_log;

    auto log = spdlog::basic_logger_st(logger_name, log_filename);
    log->set_pattern("%v");
}
#endif

template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::process_diagonal_bid(const DiagonalBidR& bid)
{

    //std::cout << "Enter process_diagonal_bid, bid = " << bid << std::endl;

    // increase price of already assigned normal items
    for(size_t k = 0; k < bid.assigned_normal_items.size(); ++k) {
        size_t assigned_normal_item_idx = bid.assigned_normal_items[k];
        Real new_price = bid.assigned_normal_items_bid_values[k];
        bool item_is_diagonal = false;
        bool bidder_is_diagonal = true;

        // TODO: SPECIAL PROCEDURE HEER`
        oracle->set_price(assigned_normal_item_idx, new_price, item_is_diagonal, bidder_is_diagonal, OwnerType::k_diagonal);
    }

    // set common diag-diag price
    // if diag_assigned_to_diag_slice_ is empty, it will be
    // numeric_limits<Real>::max()

    oracle->diag_to_diag_price_ = bid.diag_to_diag_value;

    int unassigned_diag_idx  = 0;
    auto unassigned_diag_item_iter = oracle->diag_unassigned_slice_.begin();
    auto bid_vec_idx = 0;
    for(const auto diag_bidder_idx : unassigned_diag_bidders) {
        if (unassigned_diag_idx < bid.num_from_unassigned_diag) {
            // take diagonal point from unassigned slice

            //std::cout << "assigning to diag_bidder_idx = " << diag_bidder_idx << std::endl;
            assert(unassigned_diag_item_iter != oracle->diag_unassigned_slice_.end());

            auto item_idx = *unassigned_diag_item_iter;

            ++unassigned_diag_idx;
            ++unassigned_diag_item_iter;
            assign_item_to_bidder(item_idx, diag_bidder_idx, k_invalid_index, true, true, false);
        } else {
            // take point from best_item_indices
            size_t item_idx = bid.best_item_indices[bid_vec_idx];
            Real new_price = bid.bid_values[bid_vec_idx];
            bid_vec_idx++;

            auto old_owner_idx = items_to_bidders[item_idx];
            bool item_is_diagonal = is_item_diagonal(item_idx);

            assign_item_to_bidder(item_idx, diag_bidder_idx, old_owner_idx, item_is_diagonal, true, true, new_price);
        }
    }

    // all bids of diagonal bidders are satisfied
    unassigned_diag_bidders.clear();

    if (oracle->diag_unassigned_slice_.empty()) {
        oracle->diag_unassigned_price_ = std::numeric_limits<Real>::max();
    }

    //std::cout << "Exit process_diagonal_bid\n" << *this;
}

template<class R, class AO>
bool AuctionRunnerGaussSeidelSingleDiag<R, AO>::is_bidder_diagonal(const size_t bidder_idx) const
{
    return bidder_idx >= num_normal_bidders;
}

template<class R, class AO>
bool AuctionRunnerGaussSeidelSingleDiag<R, AO>::is_bidder_normal(const size_t bidder_idx) const
{
    return bidder_idx < num_normal_bidders;
}

template<class R, class AO>
bool AuctionRunnerGaussSeidelSingleDiag<R, AO>::is_item_diagonal(const size_t item_idx) const
{
    return item_idx < num_diag_items;
}

template<class R, class AO>
bool AuctionRunnerGaussSeidelSingleDiag<R, AO>::is_item_normal(const size_t item_idx) const
{
    return item_idx >= num_diag_items;
}

template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::assign_item_to_bidder(const IdxType item_idx,
                                                   const IdxType bidder_idx,
                                                   const IdxType old_owner_idx,
                                                   const bool item_is_diagonal,
                                                   const bool bidder_is_diagonal,
                                                   const bool call_set_price,
                                                   const R new_price)
{
    //std::cout << "Enter assign_item_to_bidder, " << std::boolalpha ;
    //std::cout << "item_idx = " << item_idx << ", bidder_idx = " << bidder_idx << ", old_owner_idx = " << old_owner_idx << ", item_is_diagonal = " << item_is_diagonal << ", bidder_is_diagonal = " << bidder_is_diagonal << std::endl;
    //std::cout << "################################################################################" << std::endl;
    //std::cout << *this << std::endl;
    //std::cout << *(this->oracle) << std::endl;
    //std::cout << "################################################################################" << std::endl;
    num_rounds++;

    // for readability
    const bool item_is_normal = not item_is_diagonal;
    const bool bidder_is_normal = not bidder_is_diagonal;

    // only unassigned bidders should submit bids and get items
    assert(bidders_to_items[bidder_idx] == k_invalid_index);


    // update matching information
    bidders_to_items[bidder_idx] = item_idx;
    items_to_bidders[item_idx] = bidder_idx;


    // remove bidder from the list of unassigned bidders
    // for diagonal bidders we don't need to: in Gauss-Seidel they are all
    // processed at once, so the set unassigned_diag_bidders will be cleared
    if (bidder_is_normal) {
        unassigned_normal_bidders.erase(bidder_idx);
    }

    OwnerType old_owner_type = get_owner_type(old_owner_idx);

    if (old_owner_type != OwnerType::k_none) {
        bidders_to_items[old_owner_idx] = k_invalid_index;
    }

    switch(old_owner_type)
    {
        case OwnerType::k_normal   : unassigned_normal_bidders.insert(old_owner_idx);
                                     break;
        case OwnerType::k_diagonal : unassigned_diag_bidders.insert(old_owner_idx);
                                     break;
        case OwnerType::k_none     : break;
    }


    // update normal_items_assigned_to_diag_

    if (old_owner_type == OwnerType::k_diagonal and item_is_normal and bidder_is_normal) {
        // normal item was stolen from diagonal, erase
        assert( oracle->normal_items_assigned_to_diag_.count(item_idx) == 1 );
        oracle->normal_items_assigned_to_diag_.erase(item_idx);
    } else if (bidder_is_diagonal and item_is_normal and old_owner_type != OwnerType::k_diagonal) {
        // diagonal bidder got a new normal item, insert
        assert(oracle->normal_items_assigned_to_diag_.count(item_idx) == 0);
        oracle->normal_items_assigned_to_diag_.insert(item_idx);
    }


    // update diag_assigned_to_diag_slice_
    if (item_is_diagonal and bidder_is_normal and old_owner_type == OwnerType::k_diagonal) {
        assert( oracle->diag_assigned_to_diag_slice_.count(item_idx) == 1);
        oracle->diag_assigned_to_diag_slice_.erase(item_idx);
    } else if (item_is_diagonal and bidder_is_diagonal) {
        assert( old_owner_type != OwnerType::k_diagonal ); // diagonal does not steal from itself
        assert( oracle->diag_assigned_to_diag_slice_.count(item_idx) == 0);
        oracle->diag_assigned_to_diag_slice_.insert(item_idx);
    }

    // update diag_unassigned_slice_
    if (item_is_diagonal and old_owner_type == OwnerType::k_none) {
        oracle->diag_unassigned_slice_.erase(item_idx);
    }

    if ( not (not call_set_price or new_price != std::numeric_limits<R>::max())) {
        std::cout << "In the middle of assign_item_to_bidder, " << std::boolalpha ;
        std::cout << "item_idx = " << item_idx << ", bidder_idx = " << bidder_idx << ", old_owner_idx = " << old_owner_idx << ", item_is_diagonal = " << item_is_diagonal << ", bidder_is_diagonal = " << bidder_is_diagonal << std::endl;
        std::cout << "################################################################################" << std::endl;
        std::cout << *this << std::endl;
        std::cout << "################################################################################" << std::endl;
    }
    assert(not call_set_price or new_price != std::numeric_limits<R>::max());
    if (call_set_price) {
        oracle->set_price(item_idx, new_price, item_is_diagonal, bidder_is_diagonal, old_owner_type);
    }

    //std::cout << "Exit assign_item_to_bidder, state\n" << *this << std::endl;

#ifdef LOG_AUCTION

    partial_cost += get_item_bidder_cost(item_idx, bidder_idx, true);
    partial_cost -= get_item_bidder_cost(item_idx, old_owner_idx, true);

    unassigned_items.erase(item_idx);

    unassigned_bidders_persistence -= std::pow(bidders[bidder_idx].persistence_lp(internal_p), wasserstein_power);

    if (old_owner_type != OwnerType::k_none) {
        // item has been assigned to some other bidder,
        // and he became unassigned
        unassigned_bidders_persistence += std::pow(bidders[old_owner_idx].persistence_lp(internal_p), wasserstein_power);
    } else {
        // item was unassigned before
        unassigned_items_persistence -= std::pow(items[item_idx].persistence_lp(internal_p), wasserstein_power);
    }

    auto plot_logger = spdlog::get("plot_logger");
    plot_logger->info("{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10}",
                      num_phase,
                      num_rounds,
                      unassigned_normal_bidders.size(),
                      unassigned_diag_bidders.size(),
                      unassigned_items_persistence,
                      unassigned_bidders_persistence,
                      unassigned_items_persistence + unassigned_bidders_persistence,
                      partial_cost,
                      total_bidders_persistence,
                      total_items_persistence,
                      oracle->get_epsilon()
                      );


    if (log_auction and unassigned_normal_bidders.size() + unassigned_diag_bidders.size() <= max_unassigned_to_log) {
        auto logger = spdlog::get(logger_name);
        if (logger) {
            auto item = items[item_idx];
            auto bidder = bidders[bidder_idx];
            logger->info("{0} # ({1}, {2}) # ({3}, {4}) # {5} # {6} # {7} # {8}",
                         num_rounds,
                         item.getRealX(),
                         item.getRealY(),
                         bidder.getRealX(),
                         bidder.getRealY(),
                         format_point_set_to_log(unassigned_diag_bidders, bidders),
                         format_point_set_to_log(unassigned_normal_bidders, bidders),
                         format_point_set_to_log(unassigned_items, items),
                         oracle->get_epsilon());
        }
    }
#endif
}



template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::flush_assignment()
{
    for(auto& b2i : bidders_to_items) {
        b2i = k_invalid_index;
    }
    for(auto& i2b : items_to_bidders) {
        i2b = k_invalid_index;
    }

    // we must flush assignment only after we got perfect matching
    assert(unassigned_normal_bidders.empty() and unassigned_diag_bidders.empty());
    // all bidders become unassigned
    for(size_t bidder_idx = 0; bidder_idx < num_normal_bidders; ++bidder_idx) {
        unassigned_normal_bidders.insert(bidder_idx);
    }
    for(size_t bidder_idx = num_normal_bidders; bidder_idx < num_bidders; ++bidder_idx) {
        unassigned_diag_bidders.insert(bidder_idx);
    }
    assert(unassigned_normal_bidders.size() + unassigned_diag_bidders.size() == bidders.size());
    assert(unassigned_normal_bidders.size() == num_normal_bidders);
    assert(unassigned_diag_bidders.size() == num_diag_bidders);

    oracle->flush_assignment();
    oracle->adjust_prices();

#ifdef LOG_AUCTION
    partial_cost = 0.0;
    unassigned_bidders_persistence = total_bidders_persistence;
    unassigned_items_persistence = total_items_persistence;

    for(size_t item_idx = 0; item_idx < items.size(); ++item_idx) {
        unassigned_items.insert(item_idx);
    }
#endif

}


template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::run_auction_phases(const int max_num_phases, const Real _initial_epsilon)
{
    relative_error = std::numeric_limits<Real>::max();
    // choose some initial epsilon
    oracle->set_epsilon(_initial_epsilon);
    assert( oracle->get_epsilon() > 0 );
    for(int phase_num = 0; phase_num < max_num_phases; ++phase_num) {
        flush_assignment();
        run_auction_phase();
        phase_num++;
        //std::cout << "Iteration " << phase_num << " completed. " << std::endl;
        // result is d^q
        Real current_result = getDistanceToQthPowerInternal();
        Real denominator = current_result - num_bidders * oracle->get_epsilon();
        current_result = pow(current_result, 1.0 / wasserstein_power);
        //std::cout << "Current result is " << current_result << std::endl;
        if ( denominator <= 0 ) {
            //std::cout << "Epsilon is too big." << std::endl;
        } else {
            denominator = pow(denominator, 1.0 / wasserstein_power);
            Real numerator = current_result - denominator;
            relative_error = numerator / denominator;
            //std::cout << " numerator: " << numerator << " denominator: " << denominator << std::endl;
            //std::cout << " error bound: " << numerator / denominator << std::endl;
            // if relative error is greater than delta, continue
            if (relative_error <= delta) {
                break;
            }
        }
        // decrease epsilon for the next iteration
        oracle->set_epsilon( oracle->get_epsilon() / epsilon_common_ratio );
    }
    //print_matching();
}


template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::run_auction()
{
    double init_eps = ( initial_epsilon > 0.0 ) ? initial_epsilon : oracle->max_val_ / 4.0 ;
    run_auction_phases(max_iter_num, init_eps);
    if (relative_error > delta) {
#ifndef FOR_R_TDA
            std::cerr << "Maximum iteration number exceeded, exiting. Current result is: ";
            std::cerr << pow(wasserstein_cost, 1.0/wasserstein_power) << std::endl;
#endif
            throw std::runtime_error("Maximum iteration number exceeded");
    }
}

template<class R, class AO>
OwnerType AuctionRunnerGaussSeidelSingleDiag<R, AO>::get_owner_type(size_t bidder_idx) const
{
    if (bidder_idx == k_invalid_index) {
        return OwnerType::k_none;
    } else if (is_bidder_diagonal(bidder_idx)) {
        return OwnerType::k_diagonal;
    } else {
        return OwnerType::k_normal;
    }
}

template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::run_auction_phase()
{
    num_phase++;
    //std::cout << "Entered run_auction_phase" << std::endl;
    do {

        if (not unassigned_diag_bidders.empty()) {
            // process all unassigned diagonal bidders
            // easy for Gauss-Seidel: every bidder alwasy gets all he wants
            //
            sanity_check();
            //std::cout << "Current state " << __LINE__  << *this << std::endl;
            process_diagonal_bid(oracle->get_optimal_bids_for_diagonal( unassigned_diag_bidders.size() ));
            sanity_check();
        } else {
            sanity_check();
            // process normal unassigned bidder
            size_t bidder_idx = *(unassigned_normal_bidders.begin());
            auto optimal_bid = oracle->get_optimal_bid(bidder_idx);
            auto optimal_item_idx = optimal_bid.first;
            auto bid_value = optimal_bid.second;
            bool item_is_diagonal = is_item_diagonal(optimal_item_idx);
            size_t old_owner_idx = items_to_bidders[optimal_item_idx];

            //OwnerType old_owner_type = get_owner_type(old_owner_idx);
            //std::cout << "bidder_idx = " << bidder_idx << ", item_idx = " << optimal_item_idx <<  ", old_owner_type = " << old_owner_type << std::endl;

            assign_item_to_bidder(optimal_item_idx, bidder_idx, old_owner_idx, item_is_diagonal, false, true, bid_value);
            sanity_check();
       }

#ifdef FOR_R_TDA
        if ( num_rounds % 10000 == 0 ) {
            Rcpp::check_user_interrupt();
        }
#endif
    } while (not (unassigned_diag_bidders.empty() and unassigned_normal_bidders.empty()));
    //std::cout << "run_auction_phase finished" << std::endl;

#ifdef DEBUG_AUCTION
    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        if ( bidders_to_items[bidder_idx] < 0 or bidders_to_items[bidder_idx] >= (IdxType)num_bidders) {
            std::cerr << "After auction terminated bidder " << bidder_idx;
            std::cerr << " has no items assigned" << std::endl;
            throw std::runtime_error("Auction did not give a perfect matching");
        }
    }
#endif

}

template<class R, class AO>
R AuctionRunnerGaussSeidelSingleDiag<R, AO>::get_item_bidder_cost(size_t item_idx, size_t bidder_idx, const bool tolerate_invalid_idx) const
{
    if (item_idx != k_invalid_index and bidder_idx != k_invalid_index) {
        // skew edges are replaced by edges to projection
        if (is_bidder_diagonal(bidder_idx) and is_item_normal(item_idx)) {
            bidder_idx = item_idx;
        } else if (is_bidder_normal(bidder_idx) and is_item_diagonal(item_idx)) {
            item_idx = bidder_idx;
        }
        return std::pow(dist_lp(bidders[bidder_idx], items[item_idx], internal_p),
                        wasserstein_power);
    } else {
        if (tolerate_invalid_idx)
            return R(0.0);
        else
            throw std::runtime_error("Invalid idx in get_item_bidder_cost, item_idx = " + std::to_string(item_idx) + ", bidder_idx = " + std::to_string(bidder_idx));
    }
}

template<class R, class AO>
R AuctionRunnerGaussSeidelSingleDiag<R, AO>::getDistanceToQthPowerInternal()
{
    sanity_check();
    Real result = 0.0;
    for(size_t bIdx = 0; bIdx < num_bidders; ++bIdx) {
        result += get_item_bidder_cost(bidders_to_items[bIdx], bIdx);
    }
    wasserstein_cost = result;
    return result;
}

template<class R, class AO>
R AuctionRunnerGaussSeidelSingleDiag<R, AO>::get_wasserstein_distance()
{
    return pow(get_wasserstein_cost(), 1.0/wasserstein_power);
}

template<class R, class AO>
R AuctionRunnerGaussSeidelSingleDiag<R, AO>::get_wasserstein_cost()
{
    run_auction();
    return wasserstein_cost;
}



// Debug routines


template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::print_debug()
{
#ifdef DEBUG_AUCTION
    std::cout << "**********************" << std::endl;
    std::cout << "Current assignment:" << std::endl;
    for(size_t idx = 0; idx < bidders_to_items.size(); ++idx) {
        std::cout << idx << " <--> " << bidders_to_items[idx] << std::endl;
    }
    std::cout << "Weights: " << std::endl;
    //for(size_t i = 0; i < num_bidders; ++i) {
        //for(size_t j = 0; j < num_items; ++j) {
            //std::cout << oracle->weight_matrix[i][j] << " ";
        //}
        //std::cout << std::endl;
    //}
    std::cout << "Prices: " << std::endl;
    for(const auto price : oracle->get_prices()) {
        std::cout << price << std::endl;
    }
    std::cout << "**********************" << std::endl;
#endif
}


template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::sanity_check()
{
#ifdef DEBUG_AUCTION
    if (bidders_to_items.size() != num_bidders) {
        std::cerr << "Wrong size of bidders_to_items, must be " << num_bidders << ", is " << bidders_to_items.size() << std::endl;
        throw std::runtime_error("Wrong size of bidders_to_items");
    }

    if (items_to_bidders.size() != num_bidders) {
        std::cerr << "Wrong size of items_to_bidders, must be " << num_bidders << ", is " << items_to_bidders.size() << std::endl;
        throw std::runtime_error("Wrong size of items_to_bidders");
    }

    for(size_t bidder_idx = 0; bidder_idx < num_bidders; ++bidder_idx) {
        assert( bidders_to_items[bidder_idx] == k_invalid_index or ( bidders_to_items[bidder_idx] < static_cast<IdxType>(num_items) and bidders_to_items[bidder_idx] >= 0));

        if ( bidders_to_items[bidder_idx] != k_invalid_index) {

            if ( std::count(bidders_to_items.begin(),
                        bidders_to_items.end(),
                        bidders_to_items[bidder_idx]) > 1 ) {
                std::cerr << "Item " << bidders_to_items[bidder_idx];
                std::cerr << " appears in bidders_to_items more than once" << std::endl;
                throw std::runtime_error("Duplicate in bidders_to_items");
            }

            if (items_to_bidders.at(bidders_to_items[bidder_idx]) != static_cast<int>(bidder_idx)) {
                std::cerr << "Inconsitency: bidder_idx = " << bidder_idx;
                std::cerr << ", item_idx in bidders_to_items = ";
                std::cerr << bidders_to_items[bidder_idx];
                std::cerr << ", bidder_idx in items_to_bidders = ";
                std::cerr << items_to_bidders[bidders_to_items[bidder_idx]] << std::endl;
                throw std::runtime_error("inconsistent mapping");
            }
        }
    }

    for(size_t item_idx = 0; item_idx < num_diag_items; ++item_idx) {
        auto owner = items_to_bidders.at(item_idx);
        if ( owner == k_invalid_index) {
            assert((oracle->diag_unassigned_slice_.count(item_idx) == 1 and
                    oracle->diag_items_heap__iters_[item_idx] == oracle->diag_items_heap_.end() and
                    oracle->all_items_heap__iters_[item_idx] == oracle->all_items_heap_.end())
                   or
                    (oracle->diag_unassigned_slice_.count(item_idx) == 0 and
                     oracle->diag_items_heap__iters_[item_idx] != oracle->diag_items_heap_.end() and
                     oracle->all_items_heap__iters_[item_idx] != oracle->all_items_heap_.end()));
             assert(oracle->diag_assigned_to_diag_slice_.count(item_idx) == 0);
       } else {
            if (is_bidder_diagonal(owner)) {
                assert(oracle->diag_unassigned_slice_.count(item_idx) == 0);
                assert(oracle->diag_assigned_to_diag_slice_.count(item_idx) == 1);
                assert(oracle->diag_items_heap__iters_[item_idx] == oracle->diag_items_heap_.end());
                assert(oracle->all_items_heap__iters_[item_idx] == oracle->all_items_heap_.end());
            } else {
                assert(oracle->diag_unassigned_slice_.count(item_idx) == 0);
                assert(oracle->diag_assigned_to_diag_slice_.count(item_idx) == 0);
                assert(oracle->diag_items_heap__iters_[item_idx] != oracle->diag_items_heap_.end());
                assert(oracle->all_items_heap__iters_[item_idx] != oracle->all_items_heap_.end());
            }
        }
    }

    for(IdxType item_idx = 0; item_idx < static_cast<IdxType>(num_bidders); ++item_idx) {
        assert( items_to_bidders[item_idx] == k_invalid_index or ( items_to_bidders[item_idx] < static_cast<IdxType>(num_items) and items_to_bidders[item_idx] >= 0));
        if ( items_to_bidders.at(item_idx) != k_invalid_index) {

            // check for uniqueness
            if ( std::count(items_to_bidders.begin(),
                        items_to_bidders.end(),
                        items_to_bidders[item_idx]) > 1 ) {
                std::cerr << "Bidder " << items_to_bidders[item_idx];
                std::cerr << " appears in items_to_bidders more than once" << std::endl;
                throw std::runtime_error("Duplicate in items_to_bidders");
            }
            // check for consistency
            if (bidders_to_items.at(items_to_bidders.at(item_idx)) != static_cast<int>(item_idx)) {
                std::cerr << "Inconsitency: item_idx = " << item_idx;
                std::cerr << ", bidder_idx in items_to_bidders = ";
                std::cerr << items_to_bidders[item_idx];
                std::cerr << ", item_idx in bidders_to_items= ";
                std::cerr << bidders_to_items[items_to_bidders[item_idx]] << std::endl;
                throw std::runtime_error("inconsistent mapping");
            }
        }
    }

    oracle->sanity_check();
#endif
}

template<class R, class AO>
void AuctionRunnerGaussSeidelSingleDiag<R, AO>::print_matching()
{
#ifdef DEBUG_AUCTION
    sanity_check();
    for(size_t bIdx = 0; bIdx < bidders_to_items.size(); ++bIdx) {
        if (bidders_to_items[bIdx] != k_invalid_index) {
            auto pA = bidders[bIdx];
            auto pB = items[bidders_to_items[bIdx]];
            std::cout <<  pA << " <-> " << pB << "+" << pow(dist_lp(pA, pB, internal_p), wasserstein_power) << std::endl;
        } else {
            assert(false);
        }
    }
#endif
}

} // ws
} // hera
