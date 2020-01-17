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

#ifndef HERA_DIAGRAM_READER_H
#define HERA_DIAGRAM_READER_H

#ifndef FOR_R_TDA
#include <iostream>
#endif

#include <iomanip>
#include <sstream>
#include <string>
#include <cctype>

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

    template<class RealType_ = double, class ContType_ = std::vector<std::pair<RealType_, RealType_>>>
    inline bool readDiagramPointSet(const char* fname, ContType_& result, int& decPrecision)
    {
        using RealType = RealType_;

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
    template<class RealType_ = double, class ContType_ = std::vector<std::pair<RealType_, RealType_>>>
    inline bool readDiagramPointSet(const std::string& fname, ContType_& result, int& decPrecision)
    {
        return readDiagramPointSet<RealType_, ContType_>(fname.c_str(), result, decPrecision);
    }

    // these two functions are now just wrappers for the previous ones,
    // in case someone needs them; decPrecision is ignored
    template<class RealType_ = double, class ContType_ = std::vector<std::pair<RealType_, RealType_>>>
    inline bool readDiagramPointSet(const char* fname, ContType_& result)
    {
        int decPrecision;
        return readDiagramPointSet<RealType_, ContType_>(fname, result, decPrecision);
    }

    template<class RealType_ = double, class ContType_ = std::vector<std::pair<RealType_, RealType_>>>
    inline bool readDiagramPointSet(const std::string& fname, ContType_& result)
    {
        int decPrecision;
        return readDiagramPointSet<RealType_, ContType_>(fname.c_str(), result, decPrecision);
    }

} // end namespace hera
#endif // HERA_DIAGRAM_READER_H
