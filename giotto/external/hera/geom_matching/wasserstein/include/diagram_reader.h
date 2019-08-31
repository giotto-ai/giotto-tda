/*

Copyright (c) 2015, M. Kerber, D. Morozov, A. Nigmetov
Copyright (c) 2018, G. Spreemann
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

#ifndef HERA_DIAGRAM_READER_H
#define HERA_DIAGRAM_READER_H

#ifndef FOR_R_TDA
#include <iostream>
#endif

#include <iomanip>
#include <locale>
#include <sstream>
#include <fstream>
#include <string>
#include <cctype>
#include <algorithm>
#include <map>
#include <limits>

#include "basic_defs_ws.hpp"

#ifdef WASSERSTEIN_PURE_GEOM
#include "dnn/geometry/euclidean-dynamic.h"
#endif

namespace hera {

// cannot choose stod, stof or stold based on RealType,
// lazy solution: partial specialization
template<class RealType = double>
inline RealType parse_real_from_str(const std::string& s);

template <>
inline double parse_real_from_str<double>(const std::string& s)
{
    return std::stod(s);
}


template <>
inline long double parse_real_from_str<long double>(const std::string& s)
{
    return std::stold(s);
}


template <>
inline float parse_real_from_str<float>(const std::string& s)
{
    return std::stof(s);
}


template<class RealType>
inline RealType parse_real_from_str(const std::string& s)
{
    static_assert(sizeof(RealType) != sizeof(RealType), "Must be specialized for each type you want to use, see above");
}

// fill in result with points from file fname
// return false if file can't be opened
// or error occurred while reading
// decPrecision is the maximal decimal precision in the input,
// it is zero if all coordinates in the input are integers
template<class RealType = double, class ContType_ = std::vector<std::pair<RealType, RealType>>>
inline bool read_diagram_point_set(const char* fname, ContType_& result, int& decPrecision)
{
    size_t lineNumber { 0 };
    result.clear();
    std::ifstream f(fname);
    if (!f.good()) {
#ifndef FOR_R_TDA
        std::cerr << "Cannot open file " << fname << std::endl;
#endif
        return false;
    }
    std::locale loc;
    std::string line;
    while(std::getline(f, line)) {
        lineNumber++;
        // process comments: remove everything after hash
        auto hashPos = line.find_first_of("#", 0);
        if( std::string::npos != hashPos) {
            line = std::string(line.begin(), line.begin() + hashPos);
        }
        if (line.empty()) {
            continue;
        }
         // trim whitespaces
        auto whiteSpaceFront = std::find_if_not(line.begin(),line.end(),isspace);
        auto whiteSpaceBack = std::find_if_not(line.rbegin(),line.rend(),isspace).base();
        if (whiteSpaceBack <= whiteSpaceFront) {
            // line consists of spaces only - move to the next line
            continue;
        }
        line = std::string(whiteSpaceFront,whiteSpaceBack);

        // transform line to lower case
        // to parse Infinity
        for(auto& c : line) {
            c = std::tolower(c, loc);
        }

        bool fracPart = false;
        int currDecPrecision = 0;
        for(auto c : line) {
            if (c == '.') {
                fracPart = true;
            } else if (fracPart) {
                if (isdigit(c)) {
                    currDecPrecision++;
                } else {
                    fracPart = false;
                    if (currDecPrecision > decPrecision)
                        decPrecision = currDecPrecision;
                    currDecPrecision = 0;
                }
            }
        }

        RealType x, y;
        std::string str_x, str_y;
        std::istringstream iss(line);
        try {
            iss >> str_x >> str_y;

            x = parse_real_from_str<RealType>(str_x);
            y = parse_real_from_str<RealType>(str_y);

            if (x != y) {
                result.push_back(std::make_pair(x, y));
            } else {
#ifndef FOR_R_TDA
                std::cerr << "Warning: point with 0 persistence ignored in " << fname << ":" << lineNumber << "\n";
#endif
            }
        }
        catch (const std::invalid_argument& e) {
#ifndef FOR_R_TDA
            std::cerr << "Error in file " << fname << ", line number " << lineNumber << ": cannot parse \"" << line << "\"" << std::endl;
#endif
            return false;
        }
        catch (const std::out_of_range&) {
#ifndef FOR_R_TDA
            std::cerr << "Error while reading file " << fname << ", line number " << lineNumber << ": value too large in \"" << line << "\"" << std::endl;
#endif
            return false;
        }
    }
    f.close();
    return true;
}


// wrappers
template<class RealType = double, class ContType_ = std::vector<std::pair<RealType, RealType>>>
inline bool read_diagram_point_set(const std::string& fname, ContType_& result, int& decPrecision)
{
    return read_diagram_point_set<RealType, ContType_>(fname.c_str(), result, decPrecision);
}

// these two functions are now just wrappers for the previous ones,
// in case someone needs them; decPrecision is ignored
template<class RealType = double, class ContType_ = std::vector<std::pair<RealType, RealType>>>
inline bool read_diagram_point_set(const char* fname, ContType_& result)
{
    int decPrecision;
    return read_diagram_point_set<RealType, ContType_>(fname, result, decPrecision);
}

template<class RealType = double, class ContType_ = std::vector<std::pair<RealType, RealType>>>
inline bool read_diagram_point_set(const std::string& fname, ContType_& result)
{
    int decPrecision;
    return read_diagram_point_set<RealType, ContType_>(fname.c_str(), result, decPrecision);
}

template<class RealType = double, class ContType_ = std::vector<std::pair<RealType, RealType> > >
inline bool read_diagram_dipha(const std::string& fname, unsigned int dim, ContType_& result)
{
    std::ifstream file;
    file.open(fname, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
#ifndef FOR_R_TDA
        std::cerr << "Could not open file " << fname << "." << std::endl;
#endif
        return false;
    }

    if (read_le<int64_t>(file) != DIPHA_MAGIC) {
#ifndef FOR_R_TDA
        std::cerr << "File " << fname << " is not a valid DIPHA file." << std::endl;
#endif
        file.close();
        return false;
    }

    if (read_le<int64_t>(file) != DIPHA_PERSISTENCE_DIAGRAM) {
#ifndef FOR_R_TDA
        std::cerr << "File " << fname << " is not a valid DIPHA persistence diagram file." << std::endl;
#endif
        file.close();
        return false;
    }

    result.clear();

    int n = read_le<int64_t>(file);

    for (int i = 0; i < n; ++i) {
        int tmp_d = read_le<int64_t>(file);
        double birth = read_le<double>(file);
        double death = read_le<double>(file);

        if (death < birth) {
#ifndef FOR_R_TDA
            std::cerr << "File " << fname << " is malformed." << std::endl;
#endif
            file.close();
            return false;
        }

        int d = 0;
        if (tmp_d < 0) {
            d = -tmp_d - 1;
            death = std::numeric_limits<double>::infinity();
        } else
            d = tmp_d;

        if ((unsigned int)d == dim) {
            if (death == birth) {
#ifndef FOR_R_TDA
                std::cerr << "Warning: point with 0 persistence ignored in " << fname << "." << std::endl;
#endif
            } else {
                result.push_back(std::make_pair(birth, death));
            }
        }
    }

    file.close();

    return true;
}


template<class RealType, class ContType>
inline void remove_duplicates(ContType& dgm_A, ContType& dgm_B)
{
    std::map<std::pair<RealType, RealType>, int> map_A, map_B;
    // copy points to maps
    for(const auto& ptA : dgm_A) {
        map_A[ptA]++;
    }
    for(const auto& ptB : dgm_B) {
        map_B[ptB]++;
    }
    // clear vectors
    dgm_A.clear();
    dgm_B.clear();
    // remove duplicates from maps
    // loop over the smaller one
    if (map_A.size() <= map_B.size()) {
        for(auto& point_multiplicity_pair : map_A) {
            auto iter_B = map_B.find(point_multiplicity_pair.first);
            if (iter_B != map_B.end()) {
                int duplicate_multiplicity = std::min(point_multiplicity_pair.second, iter_B->second);
                point_multiplicity_pair.second -= duplicate_multiplicity;
                iter_B->second -= duplicate_multiplicity;
            }
        }
    } else {
        for(auto& point_multiplicity_pair : map_B) {
            auto iter_A = map_A.find(point_multiplicity_pair.first);
            if (iter_A != map_A.end()) {
                int duplicate_multiplicity = std::min(point_multiplicity_pair.second, iter_A->second);
                point_multiplicity_pair.second -= duplicate_multiplicity;
                iter_A->second -= duplicate_multiplicity;
            }
        }
    }
    // copy points back to vectors
    for(const auto& pointMultiplicityPairA : map_A) {
        assert( pointMultiplicityPairA.second >= 0);
        for(int i = 0; i < pointMultiplicityPairA.second; ++i) {
            dgm_A.push_back(pointMultiplicityPairA.first);
        }
    }

    for(const auto& pointMultiplicityPairB : map_B) {
        assert( pointMultiplicityPairB.second >= 0);
        for(int i = 0; i < pointMultiplicityPairB.second; ++i) {
            dgm_B.push_back(pointMultiplicityPairB.first);
        }
    }
}


#ifdef WASSERSTEIN_PURE_GEOM

template<class Real>
inline int get_point_dimension(const std::string& line)
{
    Real x;
    int dim = 0;
    std::istringstream iss(line);
    while(iss >> x) {
        dim++;
    }
    return dim;
}


template<class RealType = double >
inline bool read_point_cloud(const char* fname, hera::ws::dnn::DynamicPointVector<RealType>& result, int& dimension, int& decPrecision)
{
    using DynamicPointTraitsR = typename hera::ws::dnn::DynamicPointTraits<RealType>;

    size_t lineNumber { 0 };
    result.clear();
    std::ifstream f(fname);
    if (!f.good()) {
#ifndef FOR_R_TDA
        std::cerr << "Cannot open file " << fname << std::endl;
#endif
        return false;
    }
    std::string line;
    DynamicPointTraitsR traits;
    bool dim_computed = false;
    int point_idx = 0;
    while(std::getline(f, line)) {
        lineNumber++;
        // process comments: remove everything after hash
        auto hashPos = line.find_first_of("#", 0);
        if( std::string::npos != hashPos) {
            line = std::string(line.begin(), line.begin() + hashPos);
        }
        if (line.empty()) {
            continue;
        }
         // trim whitespaces
        auto whiteSpaceFront = std::find_if_not(line.begin(),line.end(),isspace);
        auto whiteSpaceBack = std::find_if_not(line.rbegin(),line.rend(),isspace).base();
        if (whiteSpaceBack <= whiteSpaceFront) {
            // line consists of spaces only - move to the next line
            continue;
        }

        line = std::string(whiteSpaceFront,whiteSpaceBack);

        if (not dim_computed) {
            dimension = get_point_dimension<RealType>(line);
            traits = hera::ws::dnn::DynamicPointTraits<RealType>(dimension);
            result = traits.container();
            result.clear();
            dim_computed = true;
        }

        bool fracPart = false;
        int currDecPrecision = 0;
        for(auto c : line) {
            if (c == '.') {
                fracPart = true;
            } else if (fracPart) {
                if (isdigit(c)) {
                    currDecPrecision++;
                } else {
                    fracPart = false;
                    if (currDecPrecision > decPrecision)
                        decPrecision = currDecPrecision;
                    currDecPrecision = 0;
                }
            }
        }

        result.resize(result.size() + 1);
        RealType x;
        std::istringstream iss(line);
        for(int d = 0; d < dimension; ++d) {
            if (not(iss >> x)) {
#ifndef FOR_R_TDA
                std::cerr << "Error in file " << fname << ", line number " << lineNumber << ": cannot parse \"" << line << "\"" << std::endl;
#endif
                return false;
            }
            result[point_idx][d] = x;
        }
        point_idx++;
    }
    f.close();
    return true;
}

// wrappers
template<class RealType = double >
inline bool read_point_cloud(const char* fname, hera::ws::dnn::DynamicPointVector<RealType>& result, int& dimension)
{
    int dec_precision;
    return read_point_cloud<RealType>(fname, result, dimension, dec_precision);
}

template<class RealType = double >
inline bool read_point_cloud(std::string fname, hera::ws::dnn::DynamicPointVector<RealType>& result, int& dimension, int& dec_precision)
{
    return read_point_cloud<RealType>(fname.c_str(), result, dimension, dec_precision);
}

template<class RealType = double >
inline bool read_point_cloud(std::string fname, hera::ws::dnn::DynamicPointVector<RealType>& result, int& dimension)
{
    return read_point_cloud<RealType>(fname.c_str(), result, dimension);
}

#endif // WASSERSTEIN_PURE_GEOM

} // end namespace hera
#endif // HERA_DIAGRAM_READER_H
