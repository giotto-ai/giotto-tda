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

#ifndef HERA_WASSERSTEIN_H
#define HERA_WASSERSTEIN_H

#include <vector>
#include <map>
#include <math.h>

#include "def_debug_ws.h"
#include "basic_defs_ws.h"
#include "diagram_reader.h"
#include "auction_runner_gs.h"
#include "auction_runner_gs_single_diag.h"
#include "auction_runner_jac.h"
#include "auction_runner_fr.h"


namespace hera
{

template<class PairContainer_, class PointType_ = typename std::remove_reference< decltype(*std::declval<PairContainer_>().begin())>::type >
struct DiagramTraits
{
    using Container = PairContainer_;
    using PointType = PointType_;
    using RealType  = typename std::remove_reference< decltype(std::declval<PointType>()[0]) >::type;

    static RealType get_x(const PointType& p)       { return p[0]; }
    static RealType get_y(const PointType& p)       { return p[1]; }
};

template<class PairContainer_>
struct DiagramTraits<PairContainer_, std::pair<double, double>>
{
    using PointType = std::pair<double, double>;
    using RealType  = double;
    using Container = std::vector<PointType>;

    static RealType get_x(const PointType& p)       { return p.first; }
    static RealType get_y(const PointType& p)       { return p.second; }
};


namespace ws
{

    // compare as multisets
    template<class PairContainer>
    inline bool are_equal(const PairContainer& dgm1, const PairContainer& dgm2)
    {
        if (dgm1.size() != dgm2.size()) {
            return false;
        }

        using Traits = typename hera::DiagramTraits<PairContainer>;
        using PointType = typename Traits::PointType;

        std::map<PointType, int> m1, m2;

        for(const auto& pair1 : dgm1) {
            m1[pair1]++;
        }

        for(const auto& pair2 : dgm2) {
            m2[pair2]++;
        }

        return m1 == m2;
    }

    // to handle points with one coordinate = infinity
    template<class RealType>
    inline RealType get_one_dimensional_cost(std::vector<RealType>& set_A,
            std::vector<RealType>& set_B,
            const RealType wasserstein_power)
    {
        if (set_A.size() != set_B.size()) {
            return std::numeric_limits<RealType>::infinity();
        }
        std::sort(set_A.begin(), set_A.end());
        std::sort(set_B.begin(), set_B.end());
        RealType result = 0.0;
        for(size_t i = 0; i < set_A.size(); ++i) {
            result += std::pow(std::fabs(set_A[i] - set_B[i]), wasserstein_power);
        }
        return result;
    }


    template<class RealType>
    struct SplitProblemInput
    {
        std::vector<DiagramPoint<RealType>> A_1;
        std::vector<DiagramPoint<RealType>> B_1;
        std::vector<DiagramPoint<RealType>> A_2;
        std::vector<DiagramPoint<RealType>> B_2;

        std::unordered_map<size_t, size_t> A_1_indices;
        std::unordered_map<size_t, size_t> A_2_indices;
        std::unordered_map<size_t, size_t> B_1_indices;
        std::unordered_map<size_t, size_t> B_2_indices;

        RealType mid_coord { 0.0 };
        RealType strip_width { 0.0 };

        void init_vectors(size_t n)
        {

            A_1_indices.clear();
            A_2_indices.clear();
            B_1_indices.clear();
            B_2_indices.clear();

            A_1.clear();
            A_2.clear();
            B_1.clear();
            B_2.clear();

            A_1.reserve(n / 2);
            B_1.reserve(n / 2);
            A_2.reserve(n / 2);
            B_2.reserve(n / 2);
        }

        void init(const std::vector<DiagramPoint<RealType>>& A,
                  const std::vector<DiagramPoint<RealType>>& B)
        {
            using DiagramPointR = DiagramPoint<RealType>;

            init_vectors(A.size());

            RealType min_sum = std::numeric_limits<RealType>::max();
            RealType max_sum = -std::numeric_limits<RealType>::max();
            for(const auto& p_A : A) {
                RealType s = p_A[0] + p_A[1];
                if (s > max_sum)
                    max_sum = s;
                if (s < min_sum)
                    min_sum = s;
                mid_coord += s;
            }

            mid_coord /= A.size();

            strip_width = 0.25 * (max_sum - min_sum);

            auto first_diag_iter = std::upper_bound(A.begin(), A.end(), 0, [](const int& a, const DiagramPointR& p) { return a < (int)(p.is_diagonal()); });
            size_t num_normal_A_points = std::distance(A.begin(), first_diag_iter);

            // process all normal points in A,
            // projections follow normal points
            for(size_t i = 0; i < A.size(); ++i) {

                assert(i < num_normal_A_points and A.is_normal() or i >= num_normal_A_points and A.is_diagonal());
                assert(i < num_normal_A_points and B.is_diagonal() or i >= num_normal_A_points and B.is_normal());

                RealType s = i < num_normal_A_points ? A[i][0] + A[i][1] : B[i][0] + B[i][1];

                if (s < mid_coord + strip_width) {
                    // add normal point and its projection to the
                    // left half
                    A_1.push_back(A[i]);
                    B_1.push_back(B[i]);
                    A_1_indices[i] = A_1.size() - 1;
                    B_1_indices[i] = B_1.size() - 1;
                }

                if (s > mid_coord - strip_width) {
                    // to the right half
                    A_2.push_back(A[i]);
                    B_2.push_back(B[i]);
                    A_2_indices[i] = A_2.size() - 1;
                    B_2_indices[i] = B_2.size() - 1;
                }

            }
        } // end init

    };


    // CAUTION:
    // this function assumes that all coordinates are finite
    // points at infinity are processed in wasserstein_cost
    template<class RealType>
    inline RealType wasserstein_cost_vec(const std::vector<DiagramPoint<RealType>>& A,
                                  const std::vector<DiagramPoint<RealType>>& B,
                                  AuctionParams<RealType>& params,
                                  const std::string& _log_filename_prefix)
    {
        if (params.wasserstein_power < 1.0) {
            throw std::runtime_error("Bad q in Wasserstein " + std::to_string(params.wasserstein_power));
        }
        if (params.delta < 0.0) {
            throw std::runtime_error("Bad delta in Wasserstein " + std::to_string(params.delta));
        }
        if (params.initial_epsilon < 0.0) {
            throw std::runtime_error("Bad initial epsilon in Wasserstein" + std::to_string(params.initial_epsilon));
        }
        if (params.epsilon_common_ratio < 0.0) {
            throw std::runtime_error("Bad epsilon factor in Wasserstein " + std::to_string(params.epsilon_common_ratio));
        }

        if (A.empty() and B.empty())
            return 0.0;

        RealType result;

        // just use Gauss-Seidel
        AuctionRunnerGS<RealType> auction(A, B, params, _log_filename_prefix);
        auction.run_auction();
        result = auction.get_wasserstein_cost();
        params.final_relative_error = auction.get_relative_error();
        return result;
    }

} // ws



template<class PairContainer>
inline typename DiagramTraits<PairContainer>::RealType
wasserstein_cost(const PairContainer& A,
                const PairContainer& B,
                AuctionParams< typename DiagramTraits<PairContainer>::RealType >& params,
                const std::string& _log_filename_prefix = "")
{
    using Traits = DiagramTraits<PairContainer>;

    //using PointType = typename Traits::PointType;
    using RealType  = typename Traits::RealType;

    if (hera::ws::are_equal(A, B)) {
        return 0.0;
    }

    bool a_empty = true;
    bool b_empty = true;
    RealType total_cost_A = 0.0;
    RealType total_cost_B = 0.0;

    using DgmPoint = hera::ws::DiagramPoint<RealType>;

    std::vector<DgmPoint> dgm_A, dgm_B;
    // coordinates of points at infinity
    std::vector<RealType> x_plus_A, x_minus_A, y_plus_A, y_minus_A;
    std::vector<RealType> x_plus_B, x_minus_B, y_plus_B, y_minus_B;
    // loop over A, add projections of A-points to corresponding positions
    // in B-vector
    for(auto& pair_A : A) {
        a_empty = false;
        RealType x = Traits::get_x(pair_A);
        RealType y = Traits::get_y(pair_A);
        if ( x == std::numeric_limits<RealType>::infinity()) {
            y_plus_A.push_back(y);
        } else if (x == -std::numeric_limits<RealType>::infinity()) {
            y_minus_A.push_back(y);
        } else if (y == std::numeric_limits<RealType>::infinity()) {
            x_plus_A.push_back(x);
        } else if (y == -std::numeric_limits<RealType>::infinity()) {
            x_minus_A.push_back(x);
        } else {
            dgm_A.emplace_back(x, y,  DgmPoint::NORMAL);
            dgm_B.emplace_back(x, y,  DgmPoint::DIAG);
            total_cost_A += std::pow(dgm_A.back().persistence_lp(params.internal_p), params.wasserstein_power);
        }
    }
    // the same for B
    for(auto& pair_B : B) {
        b_empty = false;
        RealType x = Traits::get_x(pair_B);
        RealType y = Traits::get_y(pair_B);
        if (x == std::numeric_limits<RealType>::infinity()) {
            y_plus_B.push_back(y);
        } else if (x == -std::numeric_limits<RealType>::infinity()) {
            y_minus_B.push_back(y);
        } else if (y == std::numeric_limits<RealType>::infinity()) {
            x_plus_B.push_back(x);
        } else if (y == -std::numeric_limits<RealType>::infinity()) {
            x_minus_B.push_back(x);
        } else {
            dgm_A.emplace_back(x, y,  DgmPoint::DIAG);
            dgm_B.emplace_back(x, y,  DgmPoint::NORMAL);
            total_cost_B += std::pow(dgm_B.back().persistence_lp(params.internal_p), params.wasserstein_power);
        }
    }

    RealType infinity_cost = ws::get_one_dimensional_cost(x_plus_A, x_plus_B, params.wasserstein_power);
    infinity_cost += ws::get_one_dimensional_cost(x_minus_A, x_minus_B, params.wasserstein_power);
    infinity_cost += ws::get_one_dimensional_cost(y_plus_A, y_plus_B, params.wasserstein_power);
    infinity_cost += ws::get_one_dimensional_cost(y_minus_A, y_minus_B, params.wasserstein_power);

    if (a_empty)
        return total_cost_B + infinity_cost;

    if (b_empty)
        return total_cost_A + infinity_cost;


    if (infinity_cost == std::numeric_limits<RealType>::infinity()) {
        return infinity_cost;
    } else {
        return infinity_cost + wasserstein_cost_vec(dgm_A, dgm_B, params, _log_filename_prefix);
    }

}

template<class PairContainer>
inline typename DiagramTraits<PairContainer>::RealType
wasserstein_dist(const PairContainer& A,
                 const PairContainer& B,
                 AuctionParams<typename DiagramTraits<PairContainer>::RealType>& params,
                 const std::string& _log_filename_prefix = "")
{
    using Real = typename DiagramTraits<PairContainer>::RealType;
    return std::pow(hera::wasserstein_cost(A, B, params, _log_filename_prefix), Real(1.)/params.wasserstein_power);
}

} // end of namespace hera

#endif
