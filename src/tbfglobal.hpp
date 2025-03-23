#ifndef TBFGLOBAL_HPP
#define TBFGLOBAL_HPP


template <long int Dim_T, class ConfigurationClass_T, const bool IsPeriodic_v>
class TbfMortonSpaceIndex;

template <class RealType_T, long int Dim_T = 3>
class TbfSpacialConfiguration;

template <class RealType>
using TbfDefaultSpaceIndexType = TbfMortonSpaceIndex<3, TbfSpacialConfiguration<RealType, 3>, false>;

template <class RealType>
using TbfDefaultSpaceIndexType2D = TbfMortonSpaceIndex<2, TbfSpacialConfiguration<RealType, 2>, false>;

template <class RealType>
using TbfDefaultSpaceIndexTypePeriodic = TbfMortonSpaceIndex<3, TbfSpacialConfiguration<RealType, 3>, true>;

template <class RealType>
using TbfDefaultSpaceIndexTypePeriodic2D = TbfMortonSpaceIndex<2, TbfSpacialConfiguration<RealType, 2>, true>;

constexpr static long int TbfDefaultMemoryAlignement = 64;

constexpr static long int TbfDefaultLastLevel = 2;
constexpr static long int TbfDefaultLastLevelPeriodic = 1;

#endif

