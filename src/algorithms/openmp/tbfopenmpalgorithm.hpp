#ifndef TBFOPENMPALGORITHM_HPP
#define TBFOPENMPALGORITHM_HPP

#include "tbfglobal.hpp"

#include "../sequential/tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <omp.h>


#include <cassert>
#include <iterator>

// Change when OpenMP will allow it
#define commute inout

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfOpenmpAlgorithm {
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
            auto& leafGroups = inTree.getLeafGroups();
            const auto& particleGroups = inTree.getParticleGroups();

            assert(std::size(leafGroups) == std::size(particleGroups));

            auto currentLeafGroup = leafGroups.begin();
            auto currentParticleGroup = particleGroups.cbegin();

            const auto endLeafGroup = leafGroups.end();
            const auto endParticleGroup = particleGroups.cend();

            while(currentLeafGroup != endLeafGroup && currentParticleGroup != endParticleGroup){
                assert((*currentParticleGroup).getStartingSpacialIndex() == (*currentLeafGroup).getStartingSpacialIndex()
                       && (*currentParticleGroup).getEndingSpacialIndex() == (*currentLeafGroup).getEndingSpacialIndex()
                       && (*currentParticleGroup).getNbLeaves() == (*currentLeafGroup).getNbCells());
                auto leafGroupObj = &(*currentLeafGroup);
                const auto particleGroupObj = &(*currentParticleGroup);

                const auto particleGroupObjGetDataPtr = particleGroupObj->getDataPtr();
                auto leafGroupObjGetMultipolePtr = leafGroupObj->getMultipolePtr();

#pragma omp task depend(in:particleGroupObjGetDataPtr[0]) depend(commute:leafGroupObjGetMultipolePtr[0]) default(shared) firstprivate(particleGroupObj, leafGroupObj)
                {
                    kernelWrapper.P2M(kernel, *particleGroupObj, *leafGroupObj);
                }
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(TreeClass& inTree){
        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= 2 ; --idxLevel){// TODO parent
            auto& upperCellGroup = inTree.getCellGroupsAtLevel(idxLevel);
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevel+1);

            auto currentUpperGroup = upperCellGroup.begin();
            auto currentLowerGroup = lowerCellGroup.cbegin();

            const auto endUpperGroup = upperCellGroup.end();
            const auto endLowerGroup = lowerCellGroup.cend();

            while(currentUpperGroup != endUpperGroup && currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()
                       || currentUpperGroup->getStartingSpacialIndex() <= spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));

                auto upperGroup = &(*currentUpperGroup);
                const auto lowerGroup = &(*currentLowerGroup);

                const auto lowerGroupGetMultipolePtr = lowerGroup->getMultipolePtr();
                const auto upperGroupGetMultipolePtr = upperGroup->getMultipolePtr();

#pragma omp task depend(in:lowerGroupGetMultipolePtr[0]) depend(commute:upperGroupGetMultipolePtr[0]) default(shared) firstprivate(upperGroup, lowerGroup)
                {
                    kernelWrapper.M2M(idxLevel, kernel, *lowerGroup, *upperGroup);
                }

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
        const auto& spacialSystem = inTree.getSpacialSystem();

        for(long int idxLevel = 2 ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){
                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel);
                TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), cellGroups, std::distance(cellGroups.begin(),currentCellGroup),
                                               [&](auto& groupTarget, const auto& groupSrc, const auto& indexes){
                    const auto groupSrcPtr = &groupSrc;
                    auto groupTargetPtr = &groupTarget;
                    assert(&groupTarget == &*currentCellGroup);
                    auto indexesVec = TbfUtils::CreateNew(indexes.toStdVector());

                    auto groupTargetGetLocalPtr = groupTarget.getLocalPtr();
                    const auto groupSrcGetMultipolePtr = groupSrc.getMultipolePtr();
#pragma omp task depend(in:groupSrcGetMultipolePtr[0]) depend(commute:groupTargetGetLocalPtr[0]) default(shared) firstprivate(idxLevel, indexesVec, groupSrcPtr, groupTargetPtr)
                    {
                        kernelWrapper.M2LBetweenGroups(idxLevel, kernel, *groupTargetPtr, *groupSrcPtr, std::move(*indexesVec));
                        delete indexesVec;
                    }
                });

                auto currentGroup = &(*currentCellGroup);
                auto indexesForGroup_first = TbfUtils::CreateNew(std::move(indexesForGroup.first));

                const auto currentGroupGetMultipolePtr = currentGroup->getMultipolePtr();
                auto currentGroupGetLocalPtr = currentGroup->getLocalPtr();

#pragma omp task depend(in:currentGroupGetMultipolePtr[0]) depend(commute:currentGroupGetLocalPtr[0]) default(shared) firstprivate(idxLevel, indexesForGroup_first, currentGroup)
                {
                    kernelWrapper.M2LInGroup(idxLevel, kernel, *currentGroup, std::move(*indexesForGroup_first));
                    delete indexesForGroup_first;
                }

                ++currentCellGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree){
        for(long int idxLevel = 2 ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
            const auto& upperCellGroup = inTree.getCellGroupsAtLevel(idxLevel);
            auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevel+1);

            auto currentUpperGroup = upperCellGroup.cbegin();
            auto currentLowerGroup = lowerCellGroup.begin();

            const auto endUpperGroup = upperCellGroup.cend();
            const auto endLowerGroup = lowerCellGroup.end();

            while(currentUpperGroup != endUpperGroup && currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()
                       || currentUpperGroup->getStartingSpacialIndex() <= spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));

                const auto upperGroup = &(*currentUpperGroup);
                auto lowerGroup = &(*currentLowerGroup);

                const auto upperGroupGetLocalPtr = upperGroup->getLocalPtr();
                auto lowerGroupGetLocalPtr = lowerGroup->getLocalPtr();

#pragma omp task depend(in:upperGroupGetLocalPtr[0]) depend(commute:lowerGroupGetLocalPtr[0]) default(shared) firstprivate(idxLevel, upperGroup, lowerGroup)
                {
                    kernelWrapper.L2L(idxLevel, kernel, *upperGroup, *lowerGroup);
                }

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
            const auto& leafGroups = inTree.getLeafGroups();
            auto& particleGroups = inTree.getParticleGroups();

            assert(std::size(leafGroups) == std::size(particleGroups));

            auto currentLeafGroup = leafGroups.cbegin();
            auto currentParticleGroup = particleGroups.begin();

            const auto endLeafGroup = leafGroups.cend();
            const auto endParticleGroup = particleGroups.end();

            while(currentLeafGroup != endLeafGroup && currentParticleGroup != endParticleGroup){
                assert((*currentParticleGroup).getStartingSpacialIndex() == (*currentLeafGroup).getStartingSpacialIndex()
                       && (*currentParticleGroup).getEndingSpacialIndex() == (*currentLeafGroup).getEndingSpacialIndex()
                       && (*currentParticleGroup).getNbLeaves() == (*currentLeafGroup).getNbCells());

                const auto leafGroupObj = &(*currentLeafGroup);
                auto particleGroupObj = &(*currentParticleGroup);

                const auto leafGroupObjGetLocalPtr = leafGroupObj->getLocalPtr();
                auto particleGroupObjGetRhsPtr = particleGroupObj->getRhsPtr();

#pragma omp task depend(in:leafGroupObjGetLocalPtr[0]) depend(commute:particleGroupObjGetRhsPtr[0]) default(shared) firstprivate(leafGroupObj, particleGroupObj)
                {
                    kernelWrapper.L2P(kernel, *leafGroupObj, *particleGroupObj);
                }

                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(TreeClass& inTree){
        const auto& spacialSystem = inTree.getSpacialSystem();

        auto& particleGroups = inTree.getParticleGroups();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        while(currentParticleGroup != endParticleGroup){

            auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroup, configuration.getTreeHeight()-1, true);
            TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), particleGroups, std::distance(particleGroups.begin(), currentParticleGroup),
                                           [&](auto& groupTarget, auto& groupSrc, const auto& indexes){
                assert(&groupTarget == &*currentParticleGroup);

                auto groupSrcPtr = &groupSrc;
                auto groupTargetPtr = &groupTarget;

                auto groupSrcGetDataPtr = groupSrc.getDataPtr();
                auto groupTargetGetRhsPtr = groupTarget.getRhsPtr();

                auto indexesVec = TbfUtils::CreateNew(indexes.toStdVector());
#pragma omp task depend(commute:groupSrcGetDataPtr[0],groupTargetGetRhsPtr[0]) default(shared) firstprivate(indexesVec, groupSrcPtr, groupTargetPtr)
                {
                    kernelWrapper.P2PBetweenGroups(kernel, *groupTargetPtr, *groupSrcPtr, std::move(*indexesVec));
                    delete indexesVec;
                }
            });

            auto currentGroup = &(*currentParticleGroup);

            const auto currentGroupGetDataPtr = currentGroup->getDataPtr();
            auto currentGroupGetRhsPtr = currentGroup->getRhsPtr();

            auto indexesForGroup_first = TbfUtils::CreateNew(std::move(indexesForGroup.first));
#pragma omp task depend(in:currentGroupGetDataPtr[0]) depend(commute:currentGroupGetRhsPtr[0]) default(shared) firstprivate(currentGroup, indexesForGroup_first)
            {
                kernelWrapper.P2PInGroup(kernel, *currentGroup, std::move(*indexesForGroup_first));
                delete indexesForGroup_first;

                kernelWrapper.P2PInner(kernel, *currentGroup);
            }

            ++currentParticleGroup;
        }
    }

public:
    TbfOpenmpAlgorithm(const SpacialConfiguration& inConfiguration)
        : configuration(inConfiguration), spaceSystem(configuration), kernelWrapper(configuration), kernel(configuration){
    }

    template <class SourceKernelClass>
    TbfOpenmpAlgorithm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel)
        : configuration(inConfiguration), spaceSystem(configuration), kernelWrapper(configuration), kernel(std::forward<SourceKernelClass>(inKernel)){
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::LFmmOperations::LFmmNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

#pragma omp parallel
#pragma omp master
{

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
#pragma omp taskwait
}// master
    }
};

#endif
