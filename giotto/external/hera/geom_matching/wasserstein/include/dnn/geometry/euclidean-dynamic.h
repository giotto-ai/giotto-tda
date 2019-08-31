#ifndef DNN_GEOMETRY_EUCLIDEAN_DYNAMIC_H
#define DNN_GEOMETRY_EUCLIDEAN_DYNAMIC_H

#include <vector>
#include <algorithm>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <cmath>

#include "hera_infinity.h"

namespace hera
{
namespace ws
{
namespace dnn
{

template<class Real_>
class DynamicPointVector
{
    public:
        using Real = Real_;
        struct PointType
        {
            void* p;

            Real& operator[](const int i)
            {
                return (static_cast<Real*>(p))[i];
            }

            const Real& operator[](const int i) const
            {
                return (static_cast<Real*>(p))[i];
            }

        };
        struct iterator;
        typedef             iterator                                    const_iterator;

    public:
                            DynamicPointVector(size_t point_capacity = 0):
                                point_capacity_(point_capacity)         {}


        PointType           operator[](size_t i) const                  { return {(void*) &storage_[i*point_capacity_]}; }
        inline void         push_back(PointType p);

        inline iterator         begin();
        inline iterator         end();
        inline const_iterator   begin() const;
        inline const_iterator   end() const;

        size_t              size() const                                { return storage_.size() / point_capacity_; }

        void                clear()                                     { storage_.clear(); }
        void                swap(DynamicPointVector& other)             { storage_.swap(other.storage_); std::swap(point_capacity_, other.point_capacity_); }
        void                reserve(size_t sz)                          { storage_.reserve(sz * point_capacity_); }
        void                resize(size_t sz)                           { storage_.resize(sz * point_capacity_); }

    private:
        size_t              point_capacity_;
        std::vector<char>   storage_;

    private:
        friend  class   boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)         { ar & point_capacity_ & storage_; }
};

template<typename Real>
struct DynamicPointTraits
{
    typedef         DynamicPointVector<Real>                            PointContainer;
    typedef         typename PointContainer::PointType                  PointType;
    struct PointHandle
    {
        void* p;
        bool        operator==(const PointHandle& other) const          { return p == other.p; }
        bool        operator!=(const PointHandle& other) const          { return !(*this == other); }
        bool        operator<(const PointHandle& other) const           { return p < other.p; }
        bool        operator>(const PointHandle& other) const           { return p > other.p; }
    };

    typedef         Real                                                Coordinate;
    typedef         Real                                                DistanceType;

                    DynamicPointTraits(unsigned dim = 0):
                        dim_(dim)                                       {}

    DistanceType    distance(PointType p1, PointType p2) const
        {
            Real result = 0.0;
            if (hera::is_infinity(internal_p)) {
                // max norm
                for (unsigned i = 0; i < dimension(); ++i)
                    result = std::max(result, fabs(coordinate(p1,i) - coordinate(p2,i)));
            } else if (internal_p == Real(1.0)) {
                // l1-norm
                for (unsigned i = 0; i < dimension(); ++i)
                    result += fabs(coordinate(p1,i) - coordinate(p2,i));
            } else if (internal_p == Real(2.0)) {
                result = sqrt(sq_distance(p1,p2));
            } else {
                assert(internal_p > 1.0);
                for (unsigned i = 0; i < dimension(); ++i)
                    result += std::pow(fabs(coordinate(p1,i) - coordinate(p2,i)), internal_p);
                result = std::pow(result, Real(1.0) / internal_p);
            }
            return result;
        }
    DistanceType    distance(PointHandle p1, PointHandle p2) const      { return distance(PointType({p1.p}), PointType({p2.p})); }
    DistanceType    sq_distance(PointType p1, PointType p2) const       { Real res = 0; for (unsigned i = 0; i < dimension(); ++i) { Real c1 = coordinate(p1,i), c2 = coordinate(p2,i); res += (c1 - c2)*(c1 - c2); } return res; }
    DistanceType    sq_distance(PointHandle p1, PointHandle p2) const   { return sq_distance(PointType({p1.p}), PointType({p2.p})); }
    unsigned        dimension() const                                   { return dim_; }
    Real&           coordinate(PointType p, unsigned i) const           { return ((Real*) p.p)[i]; }
    Real&           coordinate(PointHandle h, unsigned i) const         { return ((Real*) h.p)[i]; }

    // it's non-standard to return a reference, but we can rely on it for code that assumes this particular point type
    size_t&         id(PointType p) const                               { return *((size_t*) ((Real*) p.p + dimension())); }
    size_t&         id(PointHandle h) const                             { return *((size_t*) ((Real*) h.p + dimension())); }
    PointHandle     handle(PointType p) const                           { return {p.p}; }
    PointType       point(PointHandle h) const                          { return {h.p}; }

    void            swap(PointType p1, PointType p2) const              { std::swap_ranges((char*) p1.p, ((char*) p1.p) + capacity(), (char*) p2.p); }
    bool            cmp(PointType p1, PointType p2) const               { return std::lexicographical_compare((Real*) p1.p, ((Real*) p1.p) + dimension(), (Real*) p2.p, ((Real*) p2.p) + dimension()); }
    bool            eq(PointType p1, PointType p2) const                { return std::equal((Real*) p1.p, ((Real*) p1.p) + dimension(), (Real*) p2.p); }

    // non-standard, and possibly a weird name
    size_t          capacity() const                                    { return sizeof(Real)*dimension() + sizeof(size_t); }

    PointContainer  container(size_t n = 0) const                       { PointContainer c(capacity()); c.resize(n); return c; }
    PointContainer  container(size_t n, const PointType& p) const;

    typename PointContainer::iterator
                    iterator(PointContainer& c, PointHandle ph) const;
    typename PointContainer::const_iterator
                    iterator(const PointContainer& c, PointHandle ph) const;

    Real internal_p;

    private:
        unsigned    dim_;

    private:
        friend  class   boost::serialization::access;

        template<class Archive>
        void serialize(Archive& ar, const unsigned int version)         { ar & dim_; }
};

} // dnn

template<class Real>
struct dnn::DynamicPointVector<Real>::iterator:
    public boost::iterator_facade<iterator,
                                  PointType,
                                  std::random_access_iterator_tag,
                                  PointType,
                                  std::ptrdiff_t>
{
    typedef     boost::iterator_facade<iterator,
                                       PointType,
                                       std::random_access_iterator_tag,
                                       PointType,
                                       std::ptrdiff_t>              Parent;


    public:
        typedef     typename Parent::value_type                     value_type;
        typedef     typename Parent::difference_type                difference_type;
        typedef     typename Parent::reference                      reference;

                    iterator(size_t point_capacity = 0):
                        point_capacity_(point_capacity)             {}

                    iterator(void* p, size_t point_capacity):
                        p_(p), point_capacity_(point_capacity)      {}

    private:
        void        increment()                                     { p_ = ((char*) p_) + point_capacity_; }
        void        decrement()                                     { p_ = ((char*) p_) - point_capacity_; }
        void        advance(difference_type n)                      { p_ = ((char*) p_) + n*point_capacity_; }
        difference_type
                    distance_to(iterator other) const               { return (((char*) other.p_) - ((char*) p_))/(int) point_capacity_; }
        bool        equal(const iterator& other) const              { return p_ == other.p_; }
        reference   dereference() const                             { return {p_}; }

        friend class ::boost::iterator_core_access;

    private:
        void*       p_;
        size_t      point_capacity_;
};

template<class Real>
void dnn::DynamicPointVector<Real>::push_back(PointType p)
{
    if (storage_.capacity() < storage_.size() + point_capacity_)
        storage_.reserve(1.5*storage_.capacity());

    storage_.resize(storage_.size() + point_capacity_);

    std::copy((char*) p.p, (char*) p.p + point_capacity_, storage_.end() - point_capacity_);
}

template<class Real>
typename dnn::DynamicPointVector<Real>::iterator        dnn::DynamicPointVector<Real>::begin()          { return       iterator((void*) &*storage_.begin(), point_capacity_); }

template<class Real>
typename dnn::DynamicPointVector<Real>::iterator        dnn::DynamicPointVector<Real>::end()            { return       iterator((void*) &*storage_.end(),   point_capacity_); }

template<class Real>
typename dnn::DynamicPointVector<Real>::const_iterator  dnn::DynamicPointVector<Real>::begin() const    { return const_iterator((void*) &*storage_.begin(), point_capacity_); }

template<class Real>
typename dnn::DynamicPointVector<Real>::const_iterator  dnn::DynamicPointVector<Real>::end() const      { return const_iterator((void*) &*storage_.end(),   point_capacity_); }

template<typename R>
typename dnn::DynamicPointTraits<R>::PointContainer
dnn::DynamicPointTraits<R>::container(size_t n, const PointType& p) const
{
    PointContainer c = container(n);
    for (auto x : c)
        std::copy((char*) p.p, (char*) p.p + capacity(), (char*) x.p);
    return c;
}

template<typename R>
typename dnn::DynamicPointTraits<R>::PointContainer::iterator
dnn::DynamicPointTraits<R>::iterator(PointContainer& c, PointHandle ph) const
{ return typename PointContainer::iterator(ph.p, capacity()); }

template<typename R>
typename dnn::DynamicPointTraits<R>::PointContainer::const_iterator
dnn::DynamicPointTraits<R>::iterator(const PointContainer& c, PointHandle ph) const
{ return typename PointContainer::const_iterator(ph.p, capacity()); }

} // ws
} // hera

namespace std {
    template<>
    struct hash<typename hera::ws::dnn::DynamicPointTraits<double>::PointHandle>
    {
        using PointHandle = typename hera::ws::dnn::DynamicPointTraits<double>::PointHandle;
        size_t operator()(const PointHandle& ph) const
        {
            return std::hash<void*>()(ph.p);
        }
    };

    template<>
    struct hash<typename hera::ws::dnn::DynamicPointTraits<float>::PointHandle>
    {
        using PointHandle = typename hera::ws::dnn::DynamicPointTraits<float>::PointHandle;
        size_t operator()(const PointHandle& ph) const
        {
            return std::hash<void*>()(ph.p);
        }
    };


} // std


#endif
