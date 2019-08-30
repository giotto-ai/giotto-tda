#ifndef HERA_WS_DNN_UTILS_H
#define HERA_WS_DNN_UTILS_H

#include <boost/random/uniform_int.hpp>
#include <boost/foreach.hpp>
#include <boost/typeof/typeof.hpp>

namespace hera
{
namespace ws
{
namespace dnn
{

template <typename T, typename... Args>
struct has_coordinates
{
    template <typename C, typename = decltype( std::declval<C>().coordinate(std::declval<Args>()...) )>
    static std::true_type test(int);

    template <typename C>
    static std::false_type test(...);

    static constexpr bool value = decltype(test<T>(0))::value;
};

template<class RandomIt, class UniformRandomNumberGenerator, class SwapFunctor>
void random_shuffle(RandomIt first, RandomIt last, UniformRandomNumberGenerator& g, const SwapFunctor& swap)
{
    size_t n = last - first;
    boost::uniform_int<size_t> uniform(0,n);
    for (size_t i = n-1; i > 0; --i)
        swap(first[i], first[uniform(g,i+1)]);      // picks a random number in [0,i] range
}

template<class RandomIt, class UniformRandomNumberGenerator>
void random_shuffle(RandomIt first, RandomIt last, UniformRandomNumberGenerator& g)
{
    typedef     decltype(*first)            T;
    random_shuffle(first, last, g, [](T& x, T& y) { std::swap(x,y); });
}

} // dnn
} // ws
} // hera

#endif
