#include "catch/catch.hpp"

#include <sstream>
#include <iostream>

#include "bottleneck.h"

using PairVector = std::vector<std::pair<double, double>>;
using PairVectorF = std::vector<std::pair<float, float>>;

std::vector<std::string> split_on_delim(const std::string& s, char delim)
{
    std::stringstream ss(s);
    std::string token;
    std::vector<std::string> tokens;
    while(std::getline(ss, token, delim)) {
        tokens.push_back(token);
    }
    return tokens;
}


// single row in a file with test cases
struct TestFromFileCase {

    std::string file_1;
    std::string file_2;
    double delta;
    double answer;
    std::vector<std::pair<int, int>> longest_edges;
    double internal_p { hera::get_infinity<double>() };

    TestFromFileCase(const std::string& s)
    {
        auto tokens = split_on_delim(s, ' ');
//        assert(tokens.size() > 5);
        assert(tokens.size() % 2 == 0);

        size_t token_idx = 0;
        file_1 = tokens.at(token_idx++);
        file_2 = tokens.at(token_idx++);
        delta = std::stod(tokens.at(token_idx++));
        answer = std::stod(tokens.at(token_idx++));
        while(token_idx < tokens.size() - 1) {
            int v1 = std::stoi(tokens[token_idx++]);
            int v2 = std::stoi(tokens[token_idx++]);
            longest_edges.emplace_back(v1, v2);
        }
    }
};

std::ostream& operator<<(std::ostream& out, const TestFromFileCase& s)
{
    out << "[" << s.file_1 << ", " << s.file_2 << ", norm = ";
    if (s.internal_p != hera::get_infinity()) {
        out << s.internal_p;
    } else {
        out << "infinity";
    }
    out << ", answer = " << s.answer << ", edges { ";
    for(auto e : s.longest_edges) {
        out << e.first << " <-> " << e.second << ", ";
    }
    out << "} ]";
    return out;
}


TEST_CASE("simple cases", "bottleneckDistApprox")
{
    PairVector diagram_A, diagram_B;
    double delta = 0.01;
    //double internal_p = hera::get_infinity<double>();

    SECTION("trivial: two empty diagrams") {
        REQUIRE(  0.0 == hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta));
    }

    SECTION("trivial: one empty diagram, one single-point diagram") {

        diagram_A.emplace_back(1.0, 2.0);

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        REQUIRE(  fabs(d1 - 0.5) <= 0.00000000001 );

        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
        REQUIRE(  fabs(d2 - 0.5) <= 0.00000000001 );
    }

    SECTION("trivial: two single-point diagrams-1") {

        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(13.0, 19.0);  // (3, 3)

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
        double correct_answer = 3.0;
        REQUIRE(  fabs(d1 - correct_answer) <= delta * correct_answer);
        REQUIRE(  fabs(d2 - correct_answer) <= delta * correct_answer);
    }

    SECTION("trivial: two single-point diagrams-2") {

        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(130.0, 138.0);  // (4, 4)

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
        double correct_answer = 5.0;
        REQUIRE(  fabs(d1 - correct_answer) <= delta * correct_answer );
        REQUIRE(  fabs(d2 - correct_answer) <= delta * correct_answer );

    }

}

TEST_CASE("float version", "check_template")
{
    PairVectorF diagram_A, diagram_B;
    float delta = 0.01;
    //float internal_p = hera::get_infinity<double>();

    SECTION("trivial: two empty diagrams") {
        REQUIRE(  0.0 == hera::bottleneckDistApprox<PairVectorF>(diagram_A, diagram_B, delta));
        REQUIRE(  0.0 == hera::bottleneckDistExact<PairVectorF>(diagram_A, diagram_B));
    }

    SECTION("trivial: two single-point diagrams-2") {

        diagram_A.emplace_back(10, 20);  // (5, 5)
        diagram_B.emplace_back(130, 138);  // (4, 4)

        float d1 = hera::bottleneckDistApprox<PairVectorF>(diagram_A, diagram_B, delta);
        float d2 = hera::bottleneckDistApprox<PairVectorF>(diagram_B, diagram_A, delta);
        float d3 = hera::bottleneckDistExact<PairVectorF>(diagram_B, diagram_A);
        float correct_answer = 5;
        REQUIRE(  fabs(d1 - correct_answer) <= delta * correct_answer );
        REQUIRE(  fabs(d2 - correct_answer) <= delta * correct_answer );

    }

}


TEST_CASE("infinity points", "bottleneckDistApprox")
{
    PairVector diagram_A, diagram_B;
    double delta = 0.01;

    // do not use Hera's infinity! it is -1
    double inf = std::numeric_limits<double>::infinity();

    SECTION("two points at infinity, no finite points") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double corr_answer = 1.0;
        REQUIRE(  fabs(d - corr_answer) <= delta * corr_answer);
    }

    SECTION("two points at infinity") {

        // edge cost 3.0
        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
        diagram_B.emplace_back(13.0, 19.0);  // (3, 3)

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double corr_answer = 3.0;
        REQUIRE(  fabs(d - corr_answer) <= delta * corr_answer);
    }

    SECTION("three points at infinity, no finite points") {

        // edge cost 1.0
        diagram_A.emplace_back(1.0, inf);
        diagram_B.emplace_back(2.0, inf);
        diagram_B.emplace_back(2.0, inf);

        double d = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
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

        double d = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
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

        double d = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double corr_answer = 1.0;

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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
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

        double d1 = hera::bottleneckDistExact<>(diagram_A, diagram_B);
        double d2 = hera::bottleneckDistExact<>(diagram_B, diagram_A);
        double corr_answer = 3.0;

        REQUIRE( d1 == corr_answer );
        REQUIRE( d2 == corr_answer );


     }

}

TEST_CASE("longest edge", "bottleneckDistApprox")
{
    PairVector diagram_A, diagram_B;
    hera::bt::MatchingEdge<double> longest_edge_1;
    hera::bt::MatchingEdge<double> longest_edge_2;
    double delta = 0.01;

    SECTION("trivial: two empty diagrams") {
        // should not fail
        REQUIRE(  0.0 == hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta, longest_edge_1, true));
    }

    SECTION("trivial: one empty diagram, one single-point diagram") {

        diagram_A.emplace_back(1.0, 2.0);

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta, longest_edge_1, true);
        REQUIRE(longest_edge_1.first.getRealX() == 1.0);
        REQUIRE(longest_edge_1.first.getRealY() == 2.0);

        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta, longest_edge_2, true);
        REQUIRE(longest_edge_2.second.getRealX() == 1.0);
        REQUIRE(longest_edge_2.second.getRealY() == 2.0);
    }

    SECTION("trivial: two single-point diagrams-1") {

        diagram_A.emplace_back(10.0, 20.0);
        diagram_B.emplace_back(11.0, 19.0);

        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta, longest_edge_1, true);

        REQUIRE(longest_edge_1.first.getRealX() == 10.0);
        REQUIRE(longest_edge_1.first.getRealY() == 20.0);

        REQUIRE(longest_edge_1.second.getRealX() == 11.0);
        REQUIRE(longest_edge_1.second.getRealY() == 19.0);

//        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta, longest_edge_2, true);
//        REQUIRE(longest_edge_2.second.getRealX() == 1.0);
//        REQUIRE(longest_edge_2.second.getRealY() == 2.0);
//        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
//        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
//        double correct_answer = 3.0;
//        REQUIRE(  fabs(d1 - correct_answer) <= delta * correct_answer);
//        REQUIRE(  fabs(d2 - correct_answer) <= delta * correct_answer);
    }
//
//    SECTION("trivial: two single-point diagrams-2") {
//
//        diagram_A.emplace_back(10.0, 20.0);  // (5, 5)
//        diagram_B.emplace_back(130.0, 138.0);  // (4, 4)
//
//        double d1 = hera::bottleneckDistApprox<>(diagram_A, diagram_B, delta);
//        double d2 = hera::bottleneckDistApprox<>(diagram_B, diagram_A, delta);
//        double correct_answer = 5.0;
//        REQUIRE(  fabs(d1 - correct_answer) <= delta * correct_answer );
//        REQUIRE(  fabs(d2 - correct_answer) <= delta * correct_answer );
//
//    }

}

TEST_CASE("file cases", "bottleneck_dist")
{
    PairVector diagram_A, diagram_B;
    hera::bt::MatchingEdge<double> longest_edge;

    const char* file_name = "../tests/data/test_list.txt";
    std::string dir_prefix = "../tests/data/";
    std::ifstream f;
    f.open(file_name);
    std::vector<TestFromFileCase> test_params;
    std::string s;
    while (std::getline(f, s)) {
        test_params.emplace_back(s);
        //std::cout << "read test params " << test_params.back() << std::endl;
    }

    SECTION("from file:") {

        for (const auto& ts : test_params) {
            bool read_file_A = hera::readDiagramPointSet(dir_prefix + ts.file_1, diagram_A);
            bool read_file_B = hera::readDiagramPointSet(dir_prefix + ts.file_2, diagram_B);
            REQUIRE(read_file_A);
            REQUIRE(read_file_B);

            double hera_answer = hera::bottleneckDistApprox(diagram_A, diagram_B, ts.delta, longest_edge, true);
            std::pair<int, int> hera_le { longest_edge.first.get_user_id(), longest_edge.second.get_user_id() };

            REQUIRE((hera_answer == ts.answer or fabs(hera_answer - ts.answer) <= ts.delta * hera_answer));
            REQUIRE((ts.longest_edges.empty() or
                     std::find(ts.longest_edges.begin(), ts.longest_edges.end(), hera_le) != ts.longest_edges.end()));

            double hera_answer_exact = hera::bottleneckDistExact(diagram_A, diagram_B, 14, longest_edge, true);
            std::pair<int, int> hera_le_exact { longest_edge.first.get_user_id(), longest_edge.second.get_user_id() };

            REQUIRE((hera_answer_exact == ts.answer or
                     fabs(hera_answer_exact - ts.answer) <= 0.0001 * ts.answer));

            REQUIRE((ts.longest_edges.empty() or
                     std::find(ts.longest_edges.begin(), ts.longest_edges.end(), hera_le_exact) !=
                     ts.longest_edges.end()));

            // check that longest_edge length matches the bottleneck distance

            double hera_le_cost;
            bool check_longest_edge_cost = true;
            if (longest_edge.first.get_user_id() >= 0 and longest_edge.second.get_user_id() < 0) {
                // longest edge: off-diagonal point of A connected to its diagonal projection
                hera_le_cost = longest_edge.first.get_persistence(ts.internal_p);
            } else if (longest_edge.first.get_user_id() < 0 and longest_edge.second.get_user_id() >= 0) {
                // longest edge: off-diagonal point of B connected to its diagonal projection
                hera_le_cost = longest_edge.second.get_persistence(ts.internal_p);
            } else if (longest_edge.first.get_user_id() >= 0 and longest_edge.second.get_user_id() >= 0) {
                // longest edge connects two off-diagonal points of A and B
                hera_le_cost = hera::bt::dist_l_inf_slow(longest_edge.first, longest_edge.second);
            } else {
                check_longest_edge_cost = false;
            }
//            if (check_longest_edge_cost and hera_le_cost != hera_answer_exact) {
//                std::cout << "PROBLEM HERE: " << ts << ", longest  edge " << longest_edge.first << " - "
//                          << longest_edge.second << ", hera_le_cost " << hera_le_cost << ", answwer "
//                          << hera_answer_exact << std::endl;
//            }
            REQUIRE( (not check_longest_edge_cost or fabs(hera_le_cost - hera_answer_exact) < 0.0001 * hera_answer_exact) );
            std::cout << ts << " PASSED " << std::endl;
        }
    }

}
