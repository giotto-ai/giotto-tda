#ifndef HERA_WS_DNN_GEOMETRY_EUCLIDEAN_FIXED_H
#define HERA_WS_DNN_GEOMETRY_EUCLIDEAN_FIXED_H

#include <boost/operators.hpp>
#include <boost/array.hpp>
#include <boost/range/value_type.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/base_object.hpp>

//#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

#include "../parallel/tbb.h"        // for dnn::vector<...>

namespace hera
{
namespace ws
{
namespace dnn
{
    // TODO: wrap in another namespace (e.g., euclidean)

    template<size_t D, typename Real = double>
    struct Point:
        boost::addable< Point<D,Real>,
        boost::subtractable< Point<D,Real>,
        boost::dividable2< Point<D, Real>, Real,
        boost::multipliable2< Point<D, Real>, Real > > > >,
        public boost::array<Real, D>
    {
        public:
            typedef         Real                                        Coordinate;
            typedef         Real                                        DistanceType;


        public:
                            Point(size_t id = 0): id_(id)               {}
            template<size_t DD>
                            Point(const Point<DD,Real>& p, size_t id = 0):
                                id_(id)                                 { *this = p; }

            static size_t   dimension()                                 { return D; }

            // Assign a point of different dimension
            template<size_t DD>
            Point&          operator=(const Point<DD,Real>& p)          { for (size_t i = 0; i < (D < DD ? D : DD); ++i) (*this)[i] = p[i]; if (DD < D) for (size_t i = DD; i < D; ++i) (*this)[i] = 0; return *this; }

            Point&          operator+=(const Point& p)                  { for (size_t i = 0; i < D; ++i) (*this)[i] += p[i]; return *this; }
            Point&          operator-=(const Point& p)                  { for (size_t i = 0; i < D; ++i) (*this)[i] -= p[i]; return *this; }
            Point&          operator/=(Real r)                          { for (size_t i = 0; i < D; ++i) (*this)[i] /= r;    return *this; }
            Point&          operator*=(Real r)                          { for (size_t i = 0; i < D; ++i) (*this)[i] *= r;    return *this; }

            Real            norm2() const                               { Real n = 0; for (size_t i = 0; i < D; ++i) n += (*this)[i] * (*this)[i]; return n; }
            Real            max_norm() const
            {
                Real res = std::fabs((*this)[0]);
                for (size_t i = 1; i < D; ++i)
                    if (std::fabs((*this)[i]) > res)
                        res = std::fabs((*this)[i]);
                return res;
            }

            Real            l1_norm() const
            {
                Real res = std::fabs((*this)[0]);
                for (size_t i = 1; i < D; ++i)
                    res += std::fabs((*this)[i]);
                return res;
            }

            Real            lp_norm(const Real p) const
            {
                assert( !std::isinf(p) );
                if ( p == 1.0 )
                    return l1_norm();
                Real res = std::pow(std::fabs((*this)[0]), p);
                for (size_t i = 1; i < D; ++i)
                    res += std::pow(std::fabs((*this)[i]), p);
                return std::pow(res, 1.0 / p);
            }

            // quick and dirty for now; make generic later
            //DistanceType    distance(const Point& other) const          { return sqrt(sq_distance(other)); }
            //DistanceType    sq_distance(const Point& other) const       { return (other - *this).norm2(); }

            DistanceType    distance(const Point& other) const          { return (other - *this).max_norm(); }
            DistanceType    p_distance(const Point& other, const double p) const          { return (other - *this).lp_norm(p); }

            size_t          id() const                                  { return id_; }
            size_t&         id()                                        { return id_; }

        private:
            friend  class   boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int version)     {  ar & boost::serialization::base_object< boost::array<Real,D> >(*this) & id_; }

        private:
            size_t          id_;
    };

    template<size_t D, typename Real>
    std::ostream&
    operator<<(std::ostream& out, const Point<D,Real>& p)
    { out << p[0]; for (size_t i = 1; i < D; ++i) out << " " << p[i]; return out; }


    template<class Point>
    struct PointTraits;                         // intentionally undefined; should be specialized for each type


    template<size_t D, typename Real>
    struct PointTraits< Point<D, Real> >        // specialization for dnn::Point
    {
        typedef         Point<D,Real>                                       PointType;
        typedef         const PointType*                                    PointHandle;
        typedef         std::vector<PointType>                              PointContainer;

        typedef         typename PointType::Coordinate                      Coordinate;
        typedef         typename PointType::DistanceType                    DistanceType;


        static DistanceType
            distance(const PointType& p1, const PointType& p2)              { if (hera::is_infinity(internal_p)) return p1.distance(p2); else return p1.p_distance(p2, internal_p); }

        static DistanceType
            distance(PointHandle p1, PointHandle p2)                        { return distance(*p1,*p2); }

        static size_t   dimension()                                         { return D; }
        static Real     coordinate(const PointType& p, size_t i)            { return p[i]; }
        static Real&    coordinate(PointType& p, size_t i)                  { return p[i]; }
        static Real     coordinate(PointHandle p, size_t i)                 { return coordinate(*p,i); }

        static size_t   id(const PointType& p)                              { return p.id(); }
        static size_t&  id(PointType& p)                                    { return p.id(); }
        static size_t   id(PointHandle p)                                   { return id(*p); }

        static PointHandle
                        handle(const PointType& p)                          { return &p; }
        static const PointType&
                        point(PointHandle ph)                               { return *ph; }

        void            swap(PointType& p1, PointType& p2) const            { return std::swap(p1, p2); }

        static PointContainer
                        container(size_t n = 0, const PointType& p = PointType())   { return PointContainer(n, p); }
        static typename PointContainer::iterator
                        iterator(PointContainer& c, PointHandle ph)         { return c.begin() + (ph - &c[0]); }
        static typename PointContainer::const_iterator
                        iterator(const PointContainer& c, PointHandle ph)   { return c.begin() + (ph - &c[0]); }

        // Internal_p determines which norm will be used in Wasserstein metric (not to
        // be confused with wassersteinPower parameter:
        // we raise \| p - q \|_{internal_p} to wassersteinPower.
        static Real     internal_p;

        private:

            friend  class   boost::serialization::access;

            template<class Archive>
            void serialize(Archive& ar, const unsigned int version)         {}

        };

    template<size_t D, typename Real>
    Real PointTraits< Point<D, Real> >::internal_p = hera::get_infinity<Real>();


    template<class PointContainer>
    void read_points(const std::string& filename, PointContainer& points)
    {
        typedef         typename boost::range_value<PointContainer>::type   Point;
        typedef         typename PointTraits<Point>::Coordinate             Coordinate;

        std::ifstream in(filename.c_str());
        std::string   line;
        while(std::getline(in, line))
        {
            if (line[0] == '#') continue;               // comment line in the file
            std::stringstream linestream(line);
            Coordinate x;
            points.push_back(Point());
            size_t i = 0;
            while (linestream >> x)
                points.back()[i++] = x;
        }
    }
} // dnn
} // ws
} // hera

#endif
