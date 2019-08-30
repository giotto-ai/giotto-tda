#define LOG_AUCTION
#include "catch/catch.hpp"

#include <sstream>
#include <iostream>


#undef LOG_AUCTION

#include "wasserstein.h"
#include "tests_reader.h"

using namespace hera_test;

using PairVector = std::vector<std::pair<double, double>>;


TEST_CASE("simple cases", "wasserstein_dist")
{
    PairVector diagram_A, diagram_B;
    hera::AuctionParams<double> params;
    params.wasserstein_power = 1.0;
    params.delta = 0.01;
    params.internal_p = hera::get_infinity<double>();
    params.initial_epsilon = 0.0;
    params.epsilon_common_ratio = 0.0;
    params.max_num_phases = 30;
    params.gamma_threshold = 0.0;
    params.max_bids_per_round = 0;  // use Jacobi

    SECTION("trivial: two empty diagrams") {
        REQUIRE(  0.0 == hera::wasserstein_dist<>(diagram_A, diagram_B, params));
    }

    SECTION("trivial: one empty diagram, one single-point diagram") {

        diagram_A.emplace_back(1.0, 2.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        REQUIRE(  fabs(d1 - 0.5) <= 0.00000000001 );

        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d2 - 0.5) <= 0.00000000001 );

        params.internal_p = 2.0;
        double corr_answer = 1.0 / std::sqrt(2.0);
        double d3 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d3 - corr_answer) <= 0.00000000001 );

    }

    SECTION("trivial: two single-point diagrams-1") {

        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(13.0, 19.0);  // (3, 3)

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d1 - d2) <= 0.00000000001 );
        REQUIRE(  fabs(d1 - 3.0) <= 0.00000000001 );

        params.wasserstein_power = 2.0;
        double d3 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d4 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d3 - d4) <= 0.00000000001 );
        REQUIRE(  fabs(d4 - 9.0) <= 0.00000000001 );

        params.wasserstein_power = 1.0;
        params.internal_p = 1.0;
        double d5 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d6 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d5 - d6) <= 0.00000000001 );
        REQUIRE(  fabs(d5 - 4.0) <= 0.00000000001 );

        params.internal_p = 2.0;
        double d7 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d8 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d7 - d8) <= 0.00000000001 );
        REQUIRE(  fabs(d7 - std::sqrt(10.0)) <= 0.00000000001 );

    }

    SECTION("trivial: two single-point diagrams-2") {

        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(130.0, 138.0);  // (4, 4)

        double d1 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d1 - d2) <= 0.00000000001 );
        REQUIRE(  fabs(d1 - 9.0) <= 0.00000000001 );

        params.wasserstein_power = 2.0;
        double d3 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d4 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d3 - d4) <= 0.00000000001 );
        REQUIRE(  fabs(d4 - 41.0) <= 0.00000000001 ); // 5^2 + 4^2

        params.wasserstein_power = 1.0;
        params.internal_p = 1.0;
        double d5 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d6 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d5 - d6) <= 0.00000000001 );
        REQUIRE(  fabs(d5 - 18.0) <= 0.00000000001 ); // 5 + 5 + 4 + 4

        params.internal_p = 2.0;
        double d7 = hera::wasserstein_cost<>(diagram_A, diagram_B, params);
        double d8 = hera::wasserstein_cost<>(diagram_B, diagram_A, params);
        REQUIRE(  fabs(d7 - d8) <= 0.00000000001 );
        REQUIRE(  fabs(d7 - 9 * std::sqrt(2.0)) <= 0.00000000001 ); // sqrt(5^2 + 5^2) + sqrt(4^2 + 4^2) = 9 sqrt(2)

    }

}



TEST_CASE("file cases", "wasserstein_dist")
{
    PairVector diagram_A, diagram_B;
    hera::AuctionParams<double> params;
    params.wasserstein_power = 1.0;
    params.delta = 0.01;
    params.internal_p = hera::get_infinity<double>();
    params.initial_epsilon = 0.0;
    params.epsilon_common_ratio = 0.0;
    params.max_num_phases = 30;
    params.gamma_threshold = 0.0;
    params.max_bids_per_round = 1;  // use Jacobi


    SECTION("from file:") {
        const char* file_name = "../tests/data/test_list.txt";
        std::ifstream f;
        f.open(file_name);
        std::vector<TestFromFileCase> test_params;
        std::string s;
        while (std::getline(f, s)) {
            test_params.emplace_back(s);
        }

        for(const auto& ts : test_params) {
            params.wasserstein_power = ts.q;
            params.internal_p = ts.internal_p;
            bool read_file_A = hera::read_diagram_point_set<double, PairVector>(ts.file_1, diagram_A);
            bool read_file_B = hera::read_diagram_point_set<double, PairVector>(ts.file_2, diagram_B);
            REQUIRE( read_file_A );
            REQUIRE( read_file_B );
            double hera_answer = hera::wasserstein_dist(diagram_A, diagram_B, params);
            REQUIRE( fabs(hera_answer - ts.answer) <= 0.01 * hera_answer );
            std::cout << ts << " PASSED " << std::endl;
        }
    }

    SECTION("from DIPHA file:") {
        const char* file_name = "../tests/data/test_list.txt";
        std::ifstream f;
        f.open(file_name);
        std::vector<TestFromFileCase> test_params;
        std::string s;
        while (std::getline(f, s)) {
            test_params.emplace_back(s);
        }

        for(const auto& ts : test_params) {
            params.wasserstein_power = ts.q;
            params.internal_p = ts.internal_p;
            bool read_file_A = hera::read_diagram_dipha<double, PairVector>(ts.file_1 + std::string(".pd.dipha"), 1, diagram_A);
            bool read_file_B = hera::read_diagram_dipha<double, PairVector>(ts.file_2 + std::string(".pd.dipha"), 1, diagram_B);
            REQUIRE( read_file_A );
            REQUIRE( read_file_B );
            double hera_answer = hera::wasserstein_dist(diagram_A, diagram_B, params);
            REQUIRE( fabs(hera_answer - ts.answer) <= 0.01 * hera_answer );
            std::cout << ts << " PASSED " << std::endl;
        }
    }
}



TEST_CASE("infinity points", "wasserstein_dist")
{
    PairVector diagram_A, diagram_B;
    hera::AuctionParams<double> params;
    params.wasserstein_power = 1.0;
    params.delta = 0.01;
    params.internal_p = hera::get_infinity<double>();
    params.initial_epsilon = 0.0;
    params.epsilon_common_ratio = 0.0;
    params.max_num_phases = 30;
    params.gamma_threshold = 0.0;
    params.max_bids_per_round = 0;  // use Jacobi

    // do not use Hera's infinity! it is -1
    double inf = std::numeric_limits<double>::infinity();

    SECTION("two points at infinity, no finite points") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double corr_answer = 1.0;
        REQUIRE(  fabs(d - corr_answer) <= 0.00000000001 );
    }

    SECTION("two points at infinity") {

        // edge cost 3.0
        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(13.0, 19.0);  // (3, 3)

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double corr_answer = 1.0 + 3.0;
        REQUIRE(  fabs(d - corr_answer) <= 0.00000000001 );
    }

    SECTION("three points at infinity, no finite points") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double corr_answer = inf;
        REQUIRE( d  == corr_answer );
    }

    SECTION("three points at infinity") {

        // edge cost 3.0
        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(13.0, 19.0);  // (3, 3)

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double corr_answer = inf;
        REQUIRE( d  == corr_answer );
    }


    SECTION("all four corners at infinity, no finite points, finite answer") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        // edge cost 1.0
        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        // edge cost 1.0
        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        // edge cost 1.0
        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        double d = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double corr_answer = 4.0;

        REQUIRE( d  == corr_answer );
    }

    SECTION("all four corners at infinity, no finite points, infinite answer-1") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        // edge cost 1.0
        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        // edge cost 1.0
        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        // edge cost 1.0
        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, no finite points, infinite answer-2") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        // edge cost 1.0
        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        // edge cost 1.0
        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        // edge cost 1.0
        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, no finite points, infinite answer-3") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        // edge cost 1.0
        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        // edge cost 1.0
        diagram_A.emplace_back(inf, 1.0);
        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        // edge cost 1.0
        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, no finite points, infinite answer-4") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        // edge cost 1.0
        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        // edge cost 1.0
        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        // edge cost 1.0
        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);
        diagram_B.emplace_back(-inf, 2.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, with finite points, infinite answer-1") {

        diagram_A.emplace_back(1.0, inf);
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        // finite edge
        diagram_A.emplace_back(10.0, 20.0);
        diagram_B.emplace_back(13.0, 19.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, with finite points, infinite answer-2") {

        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        // finite edge
        diagram_A.emplace_back(10.0, 20.0);
        diagram_B.emplace_back(13.0, 19.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, with finite points, infinite answer-3") {

        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        diagram_A.emplace_back(inf, 1.0);
        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);

        // finite edge
        diagram_A.emplace_back(10.0, 20.0);
        diagram_B.emplace_back(13.0, 19.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }

    SECTION("all four corners at infinity, no finite points, infinite answer-4") {

        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        diagram_A.emplace_back(1.0, -inf);
        diagram_B.emplace_back(2.0, -inf);

        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        diagram_A.emplace_back(-inf, 1.0);
        diagram_B.emplace_back(-inf, 2.0);
        diagram_B.emplace_back(-inf, 2.0);

        // finite edge
        diagram_A.emplace_back(10.0, 20.0);
        diagram_B.emplace_back(13.0, 19.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = inf;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );
    }


    SECTION("simple small example with finite answer") {
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        diagram_A.emplace_back(1.9, inf);
        diagram_B.emplace_back(1.1, inf);

        // 1.1 - 1.0 +  2.0 - 1.9 = 0.2

        diagram_A.emplace_back(inf, 1.0);
        diagram_B.emplace_back(inf, 2.0);

        diagram_A.emplace_back(inf, 1.9);
        diagram_B.emplace_back(inf, 1.1);


        // finite edge
        diagram_A.emplace_back(10.0, 20.0);
        diagram_B.emplace_back(13.0, 19.0);

        double d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        double d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        double corr_answer = 3.0 + 0.2 + 0.2;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );

        params.wasserstein_power = 2.0;

        d1 = hera::wasserstein_dist<>(diagram_A, diagram_B, params);
        d2 = hera::wasserstein_dist<>(diagram_B, diagram_A, params);
        corr_answer = std::sqrt(3.0 * 3.0 + 4 * 0.1 * 0.1);

        REQUIRE( fabs(d1 - corr_answer) < 0.000000000001 );
        REQUIRE( fabs(d2 - corr_answer) < 0.000000000001 );

     }

}

