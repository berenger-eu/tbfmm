#ifndef TBFBLOCKSIZEFINDER_HPP
#define TBFBLOCKSIZEFINDER_HPP

#include "tbfglobal.hpp"

#include <thread>
#include <set>

namespace TbfBlockSizeFinder{

template <class RealType, class ParticleContainer, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType>>
int Estimate(const ParticleContainer& inParticlePositions,
             const TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>& inConfiguration,
             const int inNbThreads = static_cast<int>(std::thread::hardware_concurrency())){
    using IndexType = typename SpaceIndexType::IndexType;

    SpaceIndexType spaceSystem(inConfiguration);

    std::set<IndexType> allIndexes;

    for(long int idxPart = 0 ; idxPart < static_cast<long int>(std::size(inParticlePositions)) ; ++idxPart){
        const auto index = spaceSystem.getIndexFromPosition(inParticlePositions[idxPart]);
        allIndexes.insert(index);
    }

    return std::max(1, static_cast<int>(allIndexes.size()/(inNbThreads*inNbThreads)));
}

template <class RealType, class ParticleContainerSource, class ParticleContainerTarget, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType>>
int EstimateTsm(const ParticleContainerSource& inParticlePositionsSource,
                const ParticleContainerTarget& inParticlePositionsTarget,
             const TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>& inConfiguration,
             const int inNbThreads = static_cast<int>(std::thread::hardware_concurrency())){
    using IndexType = typename SpaceIndexType::IndexType;

    SpaceIndexType spaceSystem(inConfiguration);

    std::set<IndexType> allIndexes;

    for(long int idxPart = 0 ; idxPart < static_cast<long int>(std::size(inParticlePositionsSource)) ; ++idxPart){
        const auto index = spaceSystem.getIndexFromPosition(inParticlePositionsSource[idxPart]);
        allIndexes.insert(index);
    }

    for(long int idxPart = 0 ; idxPart < static_cast<long int>(std::size(inParticlePositionsTarget)) ; ++idxPart){
        const auto index = spaceSystem.getIndexFromPosition(inParticlePositionsTarget[idxPart]);
        allIndexes.insert(index);
    }

    return std::max(1, static_cast<int>(allIndexes.size()/(inNbThreads*inNbThreads)));
}

}

#endif
