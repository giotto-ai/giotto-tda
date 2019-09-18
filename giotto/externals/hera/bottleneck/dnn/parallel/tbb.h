#ifndef HERA_BT_PARALLEL_H
#define HERA_BT_PARALLEL_H

#ifndef FOR_R_TDA
#include <iostream>
#endif

#include <vector>

#include <boost/range.hpp>
#include <boost/bind.hpp>
#include <boost/foreach.hpp>

#ifdef TBB

#include <tbb/tbb.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/scalable_allocator.h>

#include <boost/serialization/split_free.hpp>
#include <boost/serialization/collections_load_imp.hpp>
#include <boost/serialization/collections_save_imp.hpp>

namespace hera {
namespace bt {
namespace dnn
{
    using tbb::mutex;
    using tbb::task_scheduler_init;
    using tbb::task_group;
    using tbb::task;

    template<class T>
    struct vector
    {
        typedef         tbb::concurrent_vector<T>               type;
    };

    template<class T>
    struct atomic
    {
        typedef         tbb::atomic<T>                          type;
        static T        compare_and_swap(type& v, T n, T o)     { return v.compare_and_swap(n,o); }
    };

    template<class Iterator, class F>
    void                do_foreach(Iterator begin, Iterator end, const F& f)            { tbb::parallel_do(begin, end, f); }

    template<class Range, class F>
    void                for_each_range_(const Range& r, const F& f)
    {
        for (typename Range::iterator cur = r.begin(); cur != r.end(); ++cur)
            f(*cur);
    }

    template<class F>
    void                for_each_range(size_t from, size_t to, const F& f)
    {
        //static tbb::affinity_partitioner ap;
        //tbb::parallel_for(c.range(), boost::bind(&for_each_range_<typename Container::range_type, F>, _1, f), ap);
        tbb::parallel_for(from, to, f);
    }

    template<class Container, class F>
    void                for_each_range(const Container& c, const F& f)
    {
        //static tbb::affinity_partitioner ap;
        //tbb::parallel_for(c.range(), boost::bind(&for_each_range_<typename Container::range_type, F>, _1, f), ap);
        tbb::parallel_for(c.range(), boost::bind(&for_each_range_<typename Container::const_range_type, F>, _1, f));
    }

    template<class Container, class F>
    void                for_each_range(Container& c, const F& f)
    {
        //static tbb::affinity_partitioner ap;
        //tbb::parallel_for(c.range(), boost::bind(&for_each_range_<typename Container::range_type, F>, _1, f), ap);
        tbb::parallel_for(c.range(), boost::bind(&for_each_range_<typename Container::range_type, F>, _1, f));
    }

    template<class ID, class NodePointer, class IDTraits, class Allocator>
    struct map_traits
    {
        typedef         tbb::concurrent_hash_map<ID, NodePointer, IDTraits, Allocator>              type;
        typedef         typename type::range_type                                                   range;
    };

    struct progress_timer
    {
                        progress_timer(): start(tbb::tick_count::now())                 {}
                        ~progress_timer()
                        {
#ifndef FOR_R_TDA
                            std::cout << (tbb::tick_count::now() - start).seconds() << " s" << std::endl;
#endif
                        }

        tbb::tick_count start;
    };
}
}
}

// Serialization for tbb::concurrent_vector<...>
namespace boost
{
    namespace serialization
    {
        template<class Archive, class T, class A>
        void save(Archive& ar, const tbb::concurrent_vector<T,A>& v, const unsigned int file_version)
        { stl::save_collection(ar, v); }

        template<class Archive, class T, class A>
        void load(Archive& ar, tbb::concurrent_vector<T,A>& v, const unsigned int file_version)
        {
            stl::load_collection<Archive,
                                 tbb::concurrent_vector<T,A>,
                                 stl::archive_input_seq< Archive, tbb::concurrent_vector<T,A> >,
                                 stl::reserve_imp< tbb::concurrent_vector<T,A> >
                                >(ar, v);
        }

        template<class Archive, class T, class A>
        void serialize(Archive& ar, tbb::concurrent_vector<T,A>& v, const unsigned int file_version)
        { split_free(ar, v, file_version); }

        template<class Archive, class T>
        void save(Archive& ar, const tbb::atomic<T>& v, const unsigned int file_version)
        { T v_ = v; ar << v_; }

        template<class Archive, class T>
        void load(Archive& ar, tbb::atomic<T>& v, const unsigned int file_version)
        { T v_; ar >> v_; v = v_; }

        template<class Archive, class T>
        void serialize(Archive& ar, tbb::atomic<T>& v, const unsigned int file_version)
        { split_free(ar, v, file_version); }
    }
}

#else

#include <algorithm>
#include <map>
#include <boost/progress.hpp>

namespace hera {
namespace bt {
namespace dnn
{
    template<class T>
    struct vector
    {
        typedef         ::std::vector<T>                        type;
    };

    template<class T>
    struct atomic
    {
        typedef         T                                       type;
        static T        compare_and_swap(type& v, T n, T o)     { if (v != o) return v; v = n; return o; }
    };

    template<class Iterator, class F>
    void                do_foreach(Iterator begin, Iterator end, const F& f)    { std::for_each(begin, end, f); }

    template<class F>
    void                for_each_range(size_t from, size_t to, const F& f)
    {
        for (size_t i = from; i < to; ++i)
            f(i);
    }

    template<class Container, class F>
    void                for_each_range(Container& c, const F& f)
    {
        BOOST_FOREACH(const typename Container::value_type& i, c)
            f(i);
    }

    template<class Container, class F>
    void                for_each_range(const Container& c, const F& f)
    {
        BOOST_FOREACH(const typename Container::value_type& i, c)
            f(i);
    }

    struct mutex
    {
        struct scoped_lock
        {
                        scoped_lock()                   {}
                        scoped_lock(mutex& )            {}
            void        acquire(mutex& ) const          {}
            void        release() const                 {}
        };
    };

    struct task_scheduler_init
    {
                        task_scheduler_init(unsigned)   {}
        void            initialize(unsigned)            {}
        static const unsigned automatic = 0;
        static const unsigned deferred  = 0;
    };

    struct task_group
    {
        template<class Functor>
        void    run(const Functor& f) const             { f(); }
        void    wait() const                            {}
    };

    template<class ID, class NodePointer, class IDTraits, class Allocator>
    struct map_traits
    {
        typedef         std::map<ID, NodePointer,
                                 typename IDTraits::Comparison,
                                 Allocator>                                             type;
        typedef         type                                                            range;
    };

    using boost::progress_timer;
}
}
}

#endif // TBB

namespace dnn
{
    template<class Range, class F>
    void                do_foreach(const Range& range, const F& f)                      { do_foreach(boost::begin(range), boost::end(range), f); }
}

#endif
