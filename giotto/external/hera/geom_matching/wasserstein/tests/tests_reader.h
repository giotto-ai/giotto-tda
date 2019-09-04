#ifndef WASSERSTEIN_TESTS_READER_H
#define WASSERSTEIN_TESTS_READER_H

#include <vector>
#include <string>
#include <ostream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>

#include "hera_infinity.h"

namespace  hera_test {
    inline std::vector<std::string> split_on_delim(const std::string& s, char delim)
    {
        std::stringstream ss(s);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, delim)) {
            tokens.push_back(token);
        }
        return tokens;
    }


    // single row in a file with test cases
    struct TestFromFileCase
    {

        std::string file_1;
        std::string file_2;
        double q;
        double internal_p;
        double answer;

        TestFromFileCase(std::string s)
        {
            auto tokens = split_on_delim(s, ' ');
            assert(tokens.size() == 5);

            file_1 = tokens.at(0);
            file_2 = tokens.at(1);
            q = std::stod(tokens.at(2));
            internal_p = std::stod(tokens.at(3));
            answer = std::stod(tokens.at(4));

            if (q < 1.0 or std::isinf(q) or
                (internal_p != hera::get_infinity<double>() and internal_p < 1.0)) {
                throw std::runtime_error("Bad line in test_list.txt");
            }
        }
    };

    inline std::ostream& operator<<(std::ostream& out, const TestFromFileCase& s)
    {
        out << "[" << s.file_1 << ", " << s.file_2 << ", q = " << s.q << ", norm = ";
        if (s.internal_p != hera::get_infinity()) {
            out << s.internal_p;
        } else {
            out << "infinity";
        }
        out << ", answer = " << s.answer << "]";
        return out;
    }
} // namespace hera_test
#endif //WASSERSTEIN_TESTS_READER_H
