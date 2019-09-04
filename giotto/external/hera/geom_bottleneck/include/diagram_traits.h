#ifndef HERA_DIAGRAM_TRAITS_H
#define HERA_DIAGRAM_TRAITS_H

namespace hera {

template<class PairContainer_, class PointType_ = typename std::remove_reference< decltype(*std::declval<PairContainer_>().begin())>::type >
struct DiagramTraits
{
    using Container = PairContainer_;
    using PointType = PointType_;
    using RealType  = typename std::remove_reference< decltype(std::declval<PointType>()[0]) >::type;

    static RealType get_x(const PointType& p)       { return p[0]; }
    static RealType get_y(const PointType& p)       { return p[1]; }
};


template<class PairContainer_>
struct DiagramTraits<PairContainer_, std::pair<double, double>>
{
    using PointType = std::pair<double, double>;
    using RealType  = double;
    using Container = std::vector<PointType>;

    static RealType get_x(const PointType& p)       { return p.first; }
    static RealType get_y(const PointType& p)       { return p.second; }
};


template<class PairContainer_>
struct DiagramTraits<PairContainer_, std::pair<float, float>>
{
    using PointType = std::pair<float, float>;
    using RealType  = float;
    using Container = std::vector<PointType>;

    static RealType get_x(const PointType& p)       { return p.first; }
    static RealType get_y(const PointType& p)       { return p.second; }
};


} // end namespace hera


#endif // HERA_DIAGRAM_TRAITS_H
