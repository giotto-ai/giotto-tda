#include "catch/catch.hpp"

#include <sstream>
#include <iostream>


#undef LOG_AUCTION

#include "wasserstein_pure_geom.hpp"
#include "tests_reader.h"

using namespace hera_test;

TEST_CASE("simple point clouds", "wasserstein_dist_pure_geom")
{
//    int n_points = 3;
//    int dim = 3;
//    using Traits = hera::ws::dnn::DynamicPointTraits<double>;
//    hera::ws::dnn::DynamicPointTraits<double> traits(dim);
//    hera::ws::dnn::DynamicPointVector<double> dgm_a = traits.container(n_points);
//    hera::ws::dnn::DynamicPointVector<double> dgm_b = traits.container(n_points);
//
//    dgm_a[0][0] = 0.0;
//    dgm_a[0][1] = 0.0;
//    dgm_a[0][2] = 0.0;
//
//    dgm_a[1][0] = 1.0;
//    dgm_a[1][1] = 0.0;
//    dgm_a[1][2] = 0.0;
//
//    dgm_a[2][0] = 0.0;
//    dgm_a[2][1] = 1.0;
//    dgm_a[2][2] = 1.0;
//
//    dgm_b[0][0] = 0.0;
//    dgm_b[0][1] = 0.1;
//    dgm_b[0][2] = 0.1;
//
//    dgm_b[1][0] = 1.1;
//    dgm_b[1][1] = 0.0;
//    dgm_b[1][2] = 0.0;
//
//    dgm_b[2][0] = 0.0;
//    dgm_b[2][1] = 1.1;
//    dgm_b[2][2] = 0.9;

    const int dim = 3;
    using Traits = hera::ws::dnn::DynamicPointTraits<double>;
    hera::ws::dnn::DynamicPointTraits<double> traits(dim);
    hera::AuctionParams<double> params;
    params.dim = dim;
    params.wasserstein_power = 1.0;
    params.delta = 0.01;
    params.internal_p = hera::get_infinity<double>();
    params.initial_epsilon = 0.0;
    params.epsilon_common_ratio = 0.0;
    params.max_num_phases = 30;
    params.gamma_threshold = 0.0;
    params.max_bids_per_round = 0;  // use Jacobi


    SECTION("trivial: two single-point diagrams-1") {

        int n_points = 1;
        hera::ws::dnn::DynamicPointVector<double> dgm_a = traits.container(n_points);
        hera::ws::dnn::DynamicPointVector<double> dgm_b = traits.container(n_points);

        dgm_a[0][0] = 0.0;
        dgm_a[0][1] = 0.0;
        dgm_a[0][2] = 0.0;

        dgm_b[0][0] = 1.0;
        dgm_b[0][1] = 1.0;
        dgm_b[0][2] = 1.0;

        std::vector<size_t> max_bids { 1, 10, 0 };
        std::vector<int> internal_ps{ 1, 2, static_cast<int>(hera::get_infinity()) };
        std::vector<double> wasserstein_powers { 1, 2, 3 };

        for(auto internal_p : internal_ps) {
            // there is only one point, so the answer does not depend wasserstein power
            double correct_answer;
            switch (internal_p) {
                case 1 :
                    correct_answer = 3.0;
                    break;
                case 2 :
                    correct_answer = sqrt(3.0);
                    break;
                case static_cast<int>(hera::get_infinity()) :
                    correct_answer = 1.0;
                    break;
                default :
                    throw std::runtime_error("Correct answer not specified in test case");
            }

            for (auto max_bid : max_bids) {
                for (auto wasserstein_power : wasserstein_powers) {
                    params.max_bids_per_round = max_bid;
                    params.internal_p = internal_p;
                    params.wasserstein_power = wasserstein_power;
                    double d1 = hera::ws::wasserstein_dist(dgm_a, dgm_b, params);
                    double d2 = hera::ws::wasserstein_dist(dgm_b, dgm_a, params);
                    REQUIRE(fabs(d1 - d2) <= 0.00000000001);
                    REQUIRE(fabs(d1 - correct_answer) <= 0.00000000001);
                }
            }
        }
    }
}

