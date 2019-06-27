#ifndef TBFGLOBAL_HPP
#define TBFGLOBAL_HPP


template <long int Dim_T, class ConfigurationClass_T>
class TbfMortonSpaceIndex;

template <class RealType_T, long int Dim_T = 3>
class TbfSpacialConfiguration;

template <class RealType>
using TbfDefaultSpaceIndexType = TbfMortonSpaceIndex<3, TbfSpacialConfiguration<RealType, 3>>;

constexpr static long int TbfDefaultMemoryAlignement = 64;

#endif

