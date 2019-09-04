#ifndef WASSERSTEIN_HERA_INFINITY_H
#define WASSERSTEIN_HERA_INFINITY_H

// we cannot assume that template parameter Real will always provide infinity() value,
// so value -1.0 is used to encode infinity (l_inf norm is used by default)

namespace hera {

    template<class Real = double>
    inline bool is_infinity(const Real& x)
    {
        return x == Real(-1);
    };

    template<class Real = double>
    inline constexpr Real get_infinity()
    {
        return Real(-1);
    }
}

#endif //WASSERSTEIN_HERA_INFINITY_H
