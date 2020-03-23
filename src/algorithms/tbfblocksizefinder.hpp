#ifndef TBFBLOCKSIZEFINDER_HPP
#define TBFBLOCKSIZEFINDER_HPP

#include "tbfglobal.hpp"

#include <set>

namespace TbfBlockSizeFinder{

template <class RealType, class ParticleContainer, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType_T>>
int Estimate(const ParticleContainer& inParticlePositions,
             const TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>& inConfiguration,
             const int inNbThreads){
    using IndexType = SpaceIndexType::IndexType;
    constexpr long int Dim = SpaceIndexType::Dim;

    SpaceIndexType spaceSystem(inConfiguration);

    std::set<IndexType> allIndexes;

    for(long int idxPart = 0 ; idxPart < static_cast<long int>(inParticlePositions) ; ++idxPart){
        const auto index = spaceSystem.getIndexFromPosition(inParticlePositions[idxPart]);
        allIndexes.insert(index);
    }

    return std::max(1, allIndexes.size()/(inNbThreads*inNbThreads))
}

}

#endif
