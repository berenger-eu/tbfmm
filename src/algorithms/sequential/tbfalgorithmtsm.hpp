#ifndef TBFALGORITHMTSM_HPP
#define TBFALGORITHMTSM_HPP

#include "tbfglobal.hpp"

#include "tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <cassert>
#include <iterator>

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfAlgorithmTsm {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;

    TbfGroupKernelInterface<SpaceIndexType> kernelWrapper;
    KernelClass kernel;

    template <class TreeClass>
    void P2M(TreeClass& inTree){
        if(configuration.getTreeHeight() > 2){
            auto& leafGroups = inTree.getLeafGroupsSource();
            const auto& particleGroups = inTree.getParticleGroupsSource();

            assert(std::size(leafGroups) == std::size(particleGroups));

            auto currentLeafGroup = leafGroups.begin();
            auto currentParticleGroup = particleGroups.cbegin();

            const auto endLeafGroup = leafGroups.end();
            const auto endParticleGroup = particleGroups.cend();

            while(currentLeafGroup != endLeafGroup && currentParticleGroup != endParticleGroup){
                assert((*currentParticleGroup).getStartingSpacialIndex() == (*currentLeafGroup).getStartingSpacialIndex()
                       && (*currentParticleGroup).getEndingSpacialIndex() == (*currentLeafGroup).getEndingSpacialIndex()
                       && (*currentParticleGroup).getNbLeaves() == (*currentLeafGroup).getNbCells());
                kernelWrapper.P2M(kernel, *currentParticleGroup, *currentLeafGroup);
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(TreeClass& inTree){
        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= 2 ; --idxLevel){// TODO parent
            auto& upperCellGroup = inTree.getCellGroupsAtLevelSource(idxLevel);
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevelSource(idxLevel+1);

            auto currentUpperGroup = upperCellGroup.begin();
            auto currentLowerGroup = lowerCellGroup.cbegin();

            const auto endUpperGroup = upperCellGroup.end();
            const auto endLowerGroup = lowerCellGroup.cend();

            while(currentUpperGroup != endUpperGroup && currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()
                       || currentUpperGroup->getStartingSpacialIndex() <= spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));
                kernelWrapper.M2M(idxLevel, kernel, *currentLowerGroup, *currentUpperGroup);
                if(spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()){
                    ++currentLowerGroup;
                    if(currentLowerGroup != endLowerGroup && currentUpperGroup->getEndingSpacialIndex() < spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex())){
                        ++currentUpperGroup;
                    }
                }
                else{
                    ++currentUpperGroup;
                }
            }
        }
    }

    template <class TreeClass>
    void M2L(TreeClass& inTree){
//        const auto& spacialSystem = inTree.getSpacialSystem();

//        for(long int idxLevel = 2 ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
//            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

//            auto currentCellGroup = cellGroups.begin();
//            const auto endCellGroup = cellGroups.end();

//            while(currentCellGroup != endCellGroup){

//                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel);
//                TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), cellGroups, std::distance(cellGroups.begin(),currentCellGroup),
//                                               [&](auto& groupTarget, const auto& groupSrc, const auto& indexes){
//                    assert(&groupTarget == &*currentCellGroup);
//                    kernelWrapper.M2LBetweenGroups(idxLevel, kernel, groupTarget, groupSrc, indexes);
//                });

//                kernelWrapper.M2LInGroup(idxLevel, kernel, *currentCellGroup, indexesForGroup.first);


//                ++currentCellGroup;
//            }
//        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree){
        for(long int idxLevel = 2 ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
            const auto& upperCellGroup = inTree.getCellGroupsAtLevelTarget(idxLevel);
            auto& lowerCellGroup = inTree.getCellGroupsAtLevelTarget(idxLevel+1);

            auto currentUpperGroup = upperCellGroup.cbegin();
            auto currentLowerGroup = lowerCellGroup.begin();

            const auto endUpperGroup = upperCellGroup.cend();
            const auto endLowerGroup = lowerCellGroup.end();

            while(currentUpperGroup != endUpperGroup && currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()
                       || currentUpperGroup->getStartingSpacialIndex() <= spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));
                kernelWrapper.L2L(idxLevel, kernel, *currentUpperGroup, *currentLowerGroup);
                if(spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()){
                    ++currentLowerGroup;
                    if(currentLowerGroup != endLowerGroup && currentUpperGroup->getEndingSpacialIndex() < spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex())){
                        ++currentUpperGroup;
                    }
                }
                else{
                    ++currentUpperGroup;
                }
            }
        }
    }

    template <class TreeClass>
    void L2P(TreeClass& inTree){
        if(configuration.getTreeHeight() > 2){
            const auto& leafGroups = inTree.getLeafGroupsTarget();
            auto& particleGroups = inTree.getParticleGroupsTarget();

            assert(std::size(leafGroups) == std::size(particleGroups));

            auto currentLeafGroup = leafGroups.cbegin();
            auto currentParticleGroup = particleGroups.begin();

            const auto endLeafGroup = leafGroups.cend();
            const auto endParticleGroup = particleGroups.end();

            while(currentLeafGroup != endLeafGroup && currentParticleGroup != endParticleGroup){
                assert((*currentParticleGroup).getStartingSpacialIndex() == (*currentLeafGroup).getStartingSpacialIndex()
                       && (*currentParticleGroup).getEndingSpacialIndex() == (*currentLeafGroup).getEndingSpacialIndex()
                       && (*currentParticleGroup).getNbLeaves() == (*currentLeafGroup).getNbCells());
                kernelWrapper.L2P(kernel, *currentLeafGroup, *currentParticleGroup);
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(TreeClass& inTree){
        const auto& spacialSystem = inTree.getSpacialSystem();

        auto& particleGroupsTarget = inTree.getParticleGroupsTarget();
        auto& particleGroupsSource = inTree.getParticleGroupsSource();

        auto currentParticleGroupTarget = particleGroupsTarget.begin();
        const auto endParticleGroupTarget = particleGroupsTarget.end();

        while(currentParticleGroupTarget != endParticleGroupTarget){

            auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroupTarget, configuration.getTreeHeight()-1, true);
            TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), particleGroupsSource, std::distance(particleGroupsTarget.begin(), currentParticleGroupTarget),
                                           [&](auto& groupTarget, auto& groupSrc, const auto& indexes){
                assert(&groupTarget == &*currentParticleGroupTarget);
                kernelWrapper.P2PBetweenGroupsTsm(kernel, groupTarget, groupSrc, indexes);
            });

            ++currentParticleGroupTarget;
        }
    }

public:
    TbfAlgorithmTsm(const SpacialConfiguration& inConfiguration)
        : configuration(inConfiguration), spaceSystem(configuration), kernelWrapper(configuration), kernel(configuration){
    }

    template <class SourceKernelClass>
    TbfAlgorithmTsm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel)
        : configuration(inConfiguration), spaceSystem(configuration), kernelWrapper(configuration), kernel(std::forward<SourceKernelClass>(inKernel)){
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::LFmmOperations::LFmmNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        if(inOperationToProceed & TbfAlgorithmUtils::LFmmP2M){
            P2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::LFmmM2M){
            M2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::LFmmM2L){
            M2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::LFmmL2L){
            L2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::LFmmL2P){
            L2P(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::LFmmP2P){
            P2P(inTree);
        }
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        inFunc(kernel);
    }
};

#endif
