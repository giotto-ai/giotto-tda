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

#include <assert.h>
#include <algorithm>
#include <functional>
#include <iterator>

#include "def_debug_ws.h"
#include "auction_oracle.h"


#ifdef FOR_R_TDA
#undef DEBUG_AUCTION
#endif

namespace ws {

// *****************************
// AuctionOracleLazyHeapRestricted
// *****************************


template<class Real>
AuctionOracleLazyHeapRestricted<Real>::AuctionOracleLazyHeapRestricted(const std::vector<DiagramPoint<Real>>& _bidders,
                                                                       const std::vector<DiagramPoint<Real>>& _items,
                                                                       Real _wasserstein_power,
                                                                       Real _internal_p) :
    AuctionOracleAbstract<Real>(_bidders, _items, _wasserstein_power, _internal_p),
    max_val(0.0),
    bidders_update_moments(_bidders.size(), 0),
    update_counter(0),
    heap_handles_indices(_items.size(), k_invalid_index),
    best_diagonal_items_computed(false)
{
    weight_matrix.reserve(_bidders.size());
    //const Real max_dist_upper_bound = 3 * getFurthestDistance3Approx(b, g);
    //weight_adj_const = pow(max_dist_upper_bound, wasserstein_power);
    // init weight matrix
    for(const auto& point_A : _bidders) {
        std::vector<Real> weight_vec;
        weight_vec.clear();
        weight_vec.reserve(_bidders.size());
        for(const auto& point_B : _items) {
            Real val = pow(dist_lp(point_A, point_B, _internal_p), _wasserstein_power);
            weight_vec.push_back( val );
            if ( val > max_val )
                max_val = val;
        }
        weight_matrix.push_back(weight_vec);
    }
    fill_in_losses_heap();
    for(size_t item_idx = 0; item_idx < _items.size(); ++item_idx) {
        update_list.push_back(std::make_pair(static_cast<IdxType>(item_idx), 0));
    }
    for(auto update_list_iter = update_list.begin(); update_list_iter != update_list.end(); ++update_list_iter) {
        items_iterators.push_back(update_list_iter);
    }

    size_t handle_idx {0};
    for(size_t item_idx = 0; item_idx < _items.size(); ++item_idx) {
        if (_items[item_idx].is_diagonal() ) {
            heap_handles_indices[item_idx] = handle_idx++;
            diag_heap_handles.push_back(diag_items_heap.push(std::make_pair(item_idx, 0)));
        }
    }
}


template<class Real>
void AuctionOracleLazyHeapRestricted<Real>::update_queue_for_bidder(IdxType bidder_idx)
{
    assert(0 <= bidder_idx and bidder_idx < static_cast<int>(this->bidders.size()));
    assert(bidder_idx < static_cast<int>(bidders_update_moments.size()));
    assert(losses_heap[bidder_idx] != nullptr );

    int bidder_last_update_time = bidders_update_moments[bidder_idx];
    auto iter = update_list.begin();
    while (iter != update_list.end() and iter->second >= bidder_last_update_time) {
        IdxType item_idx = iter->first;
        size_t handle_idx = items_indices_for_heap_handles[bidder_idx][item_idx];
        if (handle_idx  < this->items.size() ) {
            IdxValPair<Real> new_val { item_idx, weight_matrix[bidder_idx][item_idx] + this->prices[item_idx]};
            // to-do: change indexing of losses_heap_handles
            losses_heap[bidder_idx]->decrease(losses_heap_handles[bidder_idx][handle_idx], new_val);
        }
        iter++;
    }
    bidders_update_moments[bidder_idx] = update_counter;
}


template<class Real>
void AuctionOracleLazyHeapRestricted<Real>::fill_in_losses_heap()
{
    using LossesHeapR           = typename ws::LossesHeap<Real>;
    using LossesHeapRHandleVec  = typename std::vector<typename ws::LossesHeap<Real>::handle_type>;

    for(size_t bidder_idx = 0; bidder_idx < this->bidders.size(); ++bidder_idx) {
        DiagramPoint<Real> bidder { this->bidders[bidder_idx] };
        // no heap for diagonal bidders
        if ( bidder.is_diagonal() ) {
            losses_heap.push_back( nullptr );
            losses_heap_handles.push_back(LossesHeapRHandleVec());
            items_indices_for_heap_handles.push_back( std::vector<size_t>() );
            continue;
        } else {
            losses_heap.push_back( new LossesHeapR() );

            assert( losses_heap.at(bidder_idx) != nullptr );

            items_indices_for_heap_handles.push_back( std::vector<size_t>(this->items.size(), k_invalid_index) );
            LossesHeapRHandleVec handles_vec;
            losses_heap_handles.push_back(handles_vec);
            size_t handle_idx { 0 };
            for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
                assert( items_indices_for_heap_handles.at(bidder_idx).at(item_idx) > 0 );
                DiagramPoint<Real> item { this->items[item_idx] };
                if ( item.is_normal() ) {
                    // item can be assigned to bidder, store in heap
                    IdxValPair<Real> vp { item_idx, weight_matrix[bidder_idx][item_idx] + this->prices[item_idx] };
                    losses_heap_handles[bidder_idx].push_back(  losses_heap[bidder_idx]->push(vp) );
                    // keep corresponding index in items_indices_for_heap_handles
                    items_indices_for_heap_handles[bidder_idx][item_idx] = handle_idx++;
                }
            }
        }
    }
}


template<class Real>
AuctionOracleLazyHeapRestricted<Real>::~AuctionOracleLazyHeapRestricted()
{
    for(auto h : losses_heap) {
        delete h;
    }
}


template<class Real>
void AuctionOracleLazyHeapRestricted<Real>::set_price(IdxType item_idx, Real new_price)
{
    assert( this->prices.at(item_idx) < new_price );
#ifdef DEBUG_AUCTION
    std::cout << "price incremented by " <<  this->prices.at(item_idx) - new_price << std::endl;
#endif
    this->prices[item_idx] = new_price;
    if (this->items[item_idx].is_normal() ) {
        // lazy: record the moment we updated the price of the items,
        // do not update queues.
        // 1. move the items with updated price to the front of the update_list,
        update_list.splice(update_list.begin(), update_list, items_iterators[item_idx]);
        // 2. record the moment we updated the price and increase the counter
        update_list.front().second = update_counter++;
    } else {
        // diagonal items are stored in one heap
        diag_items_heap.decrease(diag_heap_handles[heap_handles_indices[item_idx]], std::make_pair(item_idx, new_price));
        best_diagonal_items_computed = false;
    }
}

// subtract min. price from all prices
template<class Real>
void AuctionOracleLazyHeapRestricted<Real>::adjust_prices()
{
}


template<class Real>
DebugOptimalBid<Real> AuctionOracleLazyHeapRestricted<Real>::get_optimal_bid_debug(IdxType bidder_idx)
{
    DebugOptimalBid<Real> result;
    assert(bidder_idx >=0 and bidder_idx < static_cast<IdxType>(this->bidders.size()) );

    auto bidder = this->bidders[bidder_idx];
    std::vector<IdxValPair<Real>> cand_items;
    // corresponding point is always considered as a candidate

    size_t proj_item_idx = bidder_idx;
    assert( 0 <= proj_item_idx and proj_item_idx < this->items.size() );
    auto proj_item = this->items[proj_item_idx];
    assert(proj_item.type != bidder.type);
    //assert(proj_item.proj_id == bidder.id);
    //assert(proj_item.id == bidder.proj_id);
    // todo: store precomputed distance?
    Real proj_item_value = this->get_value_for_bidder(bidder_idx, proj_item_idx);
    cand_items.push_back( std::make_pair(proj_item_idx, proj_item_value) );

    if (bidder.is_normal()) {
        assert(losses_heap.at(bidder_idx) != nullptr);
        assert(losses_heap[bidder_idx]->size() >= 2);
        update_queue_for_bidder(bidder_idx);
        auto pHeap = losses_heap[bidder_idx];
        assert( pHeap != nullptr );
        auto top_iter = pHeap->ordered_begin();
        cand_items.push_back( *top_iter );
        ++top_iter; // now points to the second-best items
        cand_items.push_back( *top_iter );
        std::sort(cand_items.begin(), cand_items.end(), CompPairsBySecondStruct<Real>());
        assert(cand_items[1].second >= cand_items[0].second);
    } else {
        // for diagonal bidder the only normal point has already been added
        // the other 2 candidates are diagonal items only, get from the heap
        // with prices
        assert(diag_items_heap.size() > 1);
        auto top_diag_iter = diag_items_heap.ordered_begin();
        auto topDiag1 = *top_diag_iter++;
        auto topDiag2 = *top_diag_iter;
        cand_items.push_back(topDiag1);
        cand_items.push_back(topDiag2);
        std::sort(cand_items.begin(), cand_items.end(), CompPairsBySecondStruct<Real>());
        assert(cand_items.size() == 3);
        assert(cand_items[2].second >= cand_items[1].second);
        assert(cand_items[1].second >= cand_items[0].second);
    }

    result.best_item_idx = cand_items[0].first;
    result.second_best_item_idx = cand_items[1].first;
    result.best_item_value = cand_items[0].second;
    result.second_best_item_value = cand_items[1].second;

    // checking code

    //DebugOptimalBid<Real> debug_my_result(result);
    //DebugOptimalBid<Real> debug_naive_result;
    //debug_naive_result.best_item_value = 1e20;
    //debug_naive_result.second_best_item_value = 1e20;
    //Real curr_item_value;
    //for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        //if ( this->bidders[bidder_idx].type != this->items[item_idx].type and
                //this->bidders[bidder_idx].proj_id != this->items[item_idx].id)
            //continue;

        //curr_item_value = pow(dist_lp(this->bidders[bidder_idx], this->items[item_idx]), wasserstein_power) + this->prices[item_idx];
        //if (curr_item_value < debug_naive_result.best_item_value) {
            //debug_naive_result.best_item_value = curr_item_value;
            //debug_naive_result.best_item_idx  = item_idx;
        //}
    //}

    //for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        //if (item_idx == debug_naive_result.best_item_idx) {
            //continue;
        //}
        //if ( this->bidders[bidder_idx].type != this->items[item_idx].type and
                //this->bidders[bidder_idx].proj_id != this->items[item_idx].id)
            //continue;

        //curr_item_value = pow(dist_lp(this->bidders[bidder_idx], this->items[item_idx]), wasserstein_power) + this->prices[item_idx];
        //if (curr_item_value < debug_naive_result.second_best_item_value) {
            //debug_naive_result.second_best_item_value = curr_item_value;
            //debug_naive_result.second_best_item_idx = item_idx;
        //}
    //}

    //if ( fabs( debug_my_result.best_item_value - debug_naive_result.best_item_value ) > 1e-6 or
            //fabs( debug_naive_result.second_best_item_value - debug_my_result.second_best_item_value) > 1e-6 ) {
        //std::cerr << "bidder_idx = " << bidder_idx << "; ";
        //std::cerr << this->bidders[bidder_idx] << std::endl;
        //for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
            //std::cout << item_idx << ": " << this->items[item_idx] << "; price = " << this->prices[item_idx] << std::endl;
        //}
        //std::cerr << "debug_my_result: " << debug_my_result << std::endl;
        //std::cerr << "debug_naive_result: " << debug_naive_result << std::endl;
        //auto pHeap = losses_heap[bidder_idx];
        //assert( pHeap != nullptr );
        //for(auto top_iter = pHeap->ordered_begin(); top_iter != pHeap->ordered_end(); ++top_iter) {
            //std::cerr << "in heap: " << top_iter->first << ": " << top_iter->second << "; real value = " << dist_lp(bidder, this->items[top_iter->first]) + this->prices[top_iter->first] << std::endl;
        //}
        //for(auto ci : cand_items) {
            //std::cout << "ci.idx = " << ci.first << ", value = " << ci.second << std::endl;
        //}

        ////std::cerr << "two_best_items: " << two_best_items[0].d << " " << two_best_items[1].d << std::endl;
        //assert(false);
    //}


    //std::cout << "get_optimal_bid: bidder_idx = " << bidder_idx << "; best_item_idx = " << best_item_idx << "; best_item_value = " << best_item_value << "; best_items_price = " << this->prices[best_item_idx] << "; second_best_item_idx = " << top_iter->first << "; second_best_value = " << second_best_item_value << "; second_best_price = " << this->prices[top_iter->first] <<  "; bid = " << this->prices[best_item_idx] + ( best_item_value - second_best_item_value ) + epsilon << "; epsilon = " << epsilon << std::endl;
    //std::cout << "get_optimal_bid: bidder_idx = " << bidder_idx << "; best_item_idx = " << best_item_idx << "; best_items_dist= " << (weight_adj_const -  best_item_value) << "; best_items_price = " << this->prices[best_item_idx] << "; second_best_item_idx = " << top_iter->first << "; second_best_dist= " << (weight_adj_const - second_best_item_value) << "; second_best_price = " << this->prices[top_iter->first] <<  "; bid = " << this->prices[best_item_idx] + ( best_item_value - second_best_item_value ) + epsilon << "; epsilon = " << epsilon << std::endl;

    return result;
}


template<class Real>
IdxValPair<Real> AuctionOracleLazyHeapRestricted<Real>::get_optimal_bid(const IdxType bidder_idx)
{
    IdxType best_item_idx;
    //IdxType second_best_item_idx;
    Real best_item_value;
    Real second_best_item_value;

    auto& bidder = this->bidders[bidder_idx];
    IdxType proj_item_idx = bidder_idx;
    assert( 0 <= proj_item_idx and proj_item_idx < this->items.size() );
    auto proj_item = this->items[proj_item_idx];
    assert(proj_item.type != bidder.type);
    //assert(proj_item.proj_id == bidder.id);
    //assert(proj_item.id == bidder.proj_id);
    // todo: store precomputed distance?
    Real proj_item_value = this->get_value_for_bidder(bidder_idx, proj_item_idx);

    if (bidder.is_diagonal()) {
        // for diagonal bidder the only normal point has already been added
        // the other 2 candidates are diagonal items only, get from the heap
        // with prices
        assert(diag_items_heap.size() > 1);
        if (!best_diagonal_items_computed) {
            auto top_diag_iter = diag_items_heap.ordered_begin();
            best_diagonal_item_idx = top_diag_iter->first;
            best_diagonal_item_value = top_diag_iter->second;
            top_diag_iter++;
            second_best_diagonal_item_idx = top_diag_iter->first;
            second_best_diagonal_item_value = top_diag_iter->second;
            best_diagonal_items_computed = true;
        }

        if ( proj_item_value < best_diagonal_item_value) {
            best_item_idx = proj_item_idx;
            best_item_value = proj_item_value;
            second_best_item_value = best_diagonal_item_value;
            //second_best_item_idx = best_diagonal_item_idx;
        } else if (proj_item_value < second_best_diagonal_item_value) {
            best_item_idx = best_diagonal_item_idx;
            best_item_value = best_diagonal_item_value;
            second_best_item_value = proj_item_value;
            //second_best_item_idx = proj_item_idx;
        } else {
            best_item_idx = best_diagonal_item_idx;
            best_item_value = best_diagonal_item_value;
            second_best_item_value = second_best_diagonal_item_value;
            //second_best_item_idx = second_best_diagonal_item_idx;
        }
    } else {
        // for normal bidder get 2 best items among non-diagonal (=normal) points
        // from the corresponding heap
        assert(diag_items_heap.size() > 1);
        update_queue_for_bidder(bidder_idx);
        auto top_norm_iter = losses_heap[bidder_idx]->ordered_begin();
        IdxType best_normal_item_idx { top_norm_iter->first };
        Real best_normal_item_value { top_norm_iter->second };
        top_norm_iter++;
        Real second_best_normal_item_value { top_norm_iter->second };
        //IdxType second_best_normal_item_idx { top_norm_iter->first };

        if ( proj_item_value < best_normal_item_value) {
            best_item_idx = proj_item_idx;
            best_item_value = proj_item_value;
            second_best_item_value = best_normal_item_value;
            //second_best_item_idx = best_normal_item_idx;
        } else if (proj_item_value < second_best_normal_item_value) {
            best_item_idx = best_normal_item_idx;
            best_item_value = best_normal_item_value;
            second_best_item_value = proj_item_value;
            //second_best_item_idx = proj_item_idx;
        } else {
            best_item_idx = best_normal_item_idx;
            best_item_value = best_normal_item_value;
            second_best_item_value = second_best_normal_item_value;
            //second_best_item_idx = second_best_normal_item_idx;
        }
    }

    IdxValPair<Real> result;

    assert( second_best_item_value >= best_item_value );

    result.first = best_item_idx;
    result.second = ( second_best_item_value - best_item_value ) + this->prices[best_item_idx] + this->epsilon;


    // checking code

    //DebugOptimalBid<Real> debug_my_result;
    //debug_my_result.best_item_idx = best_item_idx;
    //debug_my_result.best_item_value = best_item_value;
    //debug_my_result.second_best_item_idx = second_best_item_idx;
    //debug_my_result.second_best_item_value = second_best_item_value;
    //DebugOptimalBid<Real> debug_naive_result;
    //debug_naive_result.best_item_value = 1e20;
    //debug_naive_result.second_best_item_value = 1e20;
    //Real curr_item_value;
    //for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        //if ( this->bidders[bidder_idx].type != this->items[item_idx].type and
                //this->bidders[bidder_idx].proj_id != this->items[item_idx].id)
            //continue;

        //curr_item_value = this->get_value_for_bidder(bidder_idx, item_idx);
        //if (curr_item_value < debug_naive_result.best_item_value) {
            //debug_naive_result.best_item_value = curr_item_value;
            //debug_naive_result.best_item_idx  = item_idx;
        //}
    //}

    //for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
        //if (item_idx == debug_naive_result.best_item_idx) {
            //continue;
        //}
        //if ( this->bidders[bidder_idx].type != this->items[item_idx].type and
                //this->bidders[bidder_idx].proj_id != this->items[item_idx].id)
            //continue;

        //curr_item_value = this->get_value_for_bidder(bidder_idx, item_idx);
        //if (curr_item_value < debug_naive_result.second_best_item_value) {
            //debug_naive_result.second_best_item_value = curr_item_value;
            //debug_naive_result.second_best_item_idx = item_idx;
        //}
    //}
    ////std::cout << "got naive result" << std::endl;

    //if ( fabs( debug_my_result.best_item_value - debug_naive_result.best_item_value ) > 1e-6 or
            //fabs( debug_naive_result.second_best_item_value - debug_my_result.second_best_item_value) > 1e-6 ) {
        //std::cerr << "bidder_idx = " << bidder_idx << "; ";
        //std::cerr << this->bidders[bidder_idx] << std::endl;
        //for(size_t item_idx = 0; item_idx < this->items.size(); ++item_idx) {
            //std::cout << item_idx << ": " << this->items[item_idx] << "; price = " << this->prices[item_idx] << std::endl;
        //}
        //std::cerr << "debug_my_result: " << debug_my_result << std::endl;
        //std::cerr << "debug_naive_result: " << debug_naive_result << std::endl;
        //auto pHeap = losses_heap[bidder_idx];
        //if ( pHeap != nullptr ) {
            //for(auto top_iter = pHeap->ordered_begin(); top_iter != pHeap->ordered_end(); ++top_iter) {
                //std::cerr << "in heap: " << top_iter->first << ": " << top_iter->second << "; real value = " << dist_lp(bidder, this->items[top_iter->first]) + this->prices[top_iter->first] << std::endl;
            //}
        //}
        ////for(auto ci : cand_items) {
            ////std::cout << "ci.idx = " << ci.first << ", value = " << ci.second << std::endl;
        ////}

        ////std::cerr << "two_best_items: " << two_best_items[0].d << " " << two_best_items[1].d << std::endl;
        //assert(false);
    // }
    //std::cout << "get_optimal_bid: bidder_idx = " << bidder_idx << "; best_item_idx = " << best_item_idx << "; best_item_value = " << best_item_value << "; best_items_price = " << this->prices[best_item_idx] << "; second_best_item_idx = " << top_iter->first << "; second_best_value = " << second_best_item_value << "; second_best_price = " << this->prices[top_iter->first] <<  "; bid = " << this->prices[best_item_idx] + ( best_item_value - second_best_item_value ) + epsilon << "; epsilon = " << epsilon << std::endl;
    //std::cout << "get_optimal_bid: bidder_idx = " << bidder_idx << "; best_item_idx = " << best_item_idx << "; best_items_dist= " << (weight_adj_const -  best_item_value) << "; best_items_price = " << this->prices[best_item_idx] << "; second_best_item_idx = " << top_iter->first << "; second_best_dist= " << (weight_adj_const - second_best_item_value) << "; second_best_price = " << this->prices[top_iter->first] <<  "; bid = " << this->prices[best_item_idx] + ( best_item_value - second_best_item_value ) + epsilon << "; epsilon = " << epsilon << std::endl;

    return result;
}

} // end of namespace ws
