#ifndef WASSERSTEIN_PURE_GEOM_HPP
#define WASSERSTEIN_PURE_GEOM_HPP

#define WASSERSTEIN_PURE_GEOM


#include "diagram_reader.h"
#include "auction_oracle_kdtree_pure_geom.h"
#include "auction_runner_gs.h"
#include "auction_runner_jac.h"

namespace hera
{
namespace ws
{

    template <class Real>
    using DynamicTraits = typename hera::ws::dnn::DynamicPointTraits<Real>;

    template <class Real>
    using DynamicPoint = typename hera::ws::dnn::DynamicPointTraits<Real>::PointType;

    template <class Real>
    using DynamicPointVector = typename hera::ws::dnn::DynamicPointVector<Real>;

    template <class Real>
    using AuctionRunnerGSR = typename hera::ws::AuctionRunnerGS<Real, hera::ws::AuctionOracleKDTreePureGeom<Real>, hera::ws::dnn::DynamicPointVector<Real>>;

    template <class Real>
    using AuctionRunnerJacR = typename hera::ws::AuctionRunnerJac<Real, hera::ws::AuctionOracleKDTreePureGeom<Real>, hera::ws::dnn::DynamicPointVector<Real>>;


inline double wasserstein_cost(const DynamicPointVector<double>& set_A, const DynamicPointVector<double>& set_B, const AuctionParams<double>& params)
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

    if (set_A.size() != set_B.size()) {
        throw std::runtime_error("Different cardinalities of point clouds: " + std::to_string(set_A.size()) + " != " +  std::to_string(set_B.size()));
    }

    DynamicTraits<double> traits(params.dim);

    DynamicPointVector<double> set_A_copy(set_A);
    DynamicPointVector<double> set_B_copy(set_B);

    // set point id to the index in vector
    for(size_t i = 0; i < set_A.size(); ++i) {
        traits.id(set_A_copy[i]) = i;
        traits.id(set_B_copy[i]) = i;
    }

    if (params.max_bids_per_round == 1) {
        hera::ws::AuctionRunnerGSR<double> auction(set_A_copy, set_B_copy, params);
        auction.run_auction();
        return auction.get_wasserstein_cost();
    } else {
        hera::ws::AuctionRunnerJacR<double> auction(set_A_copy, set_B_copy, params);
        auction.run_auction();
        return auction.get_wasserstein_cost();
    }
}

inline double wasserstein_dist(const DynamicPointVector<double>& set_A, const DynamicPointVector<double>& set_B, const AuctionParams<double>& params)
{
    return std::pow(wasserstein_cost(set_A, set_B, params), 1.0 / params.wasserstein_power);
}

} // ws
} // hera


#endif
