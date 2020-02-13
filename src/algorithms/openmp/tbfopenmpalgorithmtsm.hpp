#ifndef TBFOPENMPALGORITHMTSM_HPP
#define TBFOPENMPALGORITHMTSM_HPP

#include "tbfglobal.hpp"

#include "../sequential/tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <omp.h>


#include <cassert>
#include <iterator>

#if _OPENMP >= 201811
#ifndef mutexinout
#define commute mutexinout
#endif
#else
#define commute inout
#endif

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfOpenmpAlgorithmTsm {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;

    const long int stopUpperLevel;

    TbfGroupKernelInterface<SpaceIndexType> kernelWrapper;
    std::vector<KernelClass> kernels;

    TbfAlgorithmUtils::TbfOperationsPriorities priorities;

    template <class TreeClass>
    void P2M(TreeClass& inTree){
        if(configuration.getTreeHeight() > stopUpperLevel){
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
                auto leafGroupObj = &(*currentLeafGroup);
                const auto particleGroupObj = &(*currentParticleGroup);

                const auto particleGroupObjGetDataPtr = particleGroupObj->getDataPtr();
                auto leafGroupObjGetMultipolePtr = leafGroupObj->getMultipolePtr();

#pragma omp task depend(in:particleGroupObjGetDataPtr[0]) depend(commute:leafGroupObjGetMultipolePtr[0]) default(shared) firstprivate(particleGroupObj, leafGroupObj) priority(priorities.getP2MPriority())
                {
                    kernelWrapper.P2M(kernels[omp_get_thread_num()], *particleGroupObj, *leafGroupObj);
                }
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(TreeClass& inTree){
        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= stopUpperLevel ; --idxLevel){
            auto& upperCellGroup = inTree.getCellGroupsAtLevelSource(idxLevel);
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevelSource(idxLevel+1);

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

#pragma omp task depend(in:lowerGroupGetMultipolePtr[0]) depend(commute:upperGroupGetMultipolePtr[0]) default(shared) firstprivate(upperGroup, lowerGroup)  priority(priorities.getM2MPriority(idxLevel))
                {
                    kernelWrapper.M2M(idxLevel, kernels[omp_get_thread_num()], *lowerGroup, *upperGroup);
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

        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
            auto& cellGroupsTarget = inTree.getCellGroupsAtLevelTarget(idxLevel);
            auto& cellGroupsSource = inTree.getCellGroupsAtLevelSource(idxLevel);

            auto currentCellGroup = cellGroupsTarget.begin();
            const auto endCellGroup = cellGroupsTarget.end();

            while(currentCellGroup != endCellGroup){
                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel, false);

                indexesForGroup.second.reserve(std::size(indexesForGroup.first) + std::size(indexesForGroup.first));
                indexesForGroup.second.insert(indexesForGroup.second.end(), indexesForGroup.first.begin(), indexesForGroup.first.end());

                TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), cellGroupsSource,
                                                          std::distance(cellGroupsTarget.begin(),currentCellGroup), cellGroupsTarget,
                                               [&](auto& groupTarget, const auto& groupSrc, const auto& indexes){
                    const auto groupSrcPtr = &groupSrc;
                    auto groupTargetPtr = &groupTarget;
                    assert(&groupTarget == &*currentCellGroup);
                    auto indexesVec = TbfUtils::CreateNew(indexes.toStdVector());

                    auto groupTargetGetLocalPtr = groupTarget.getLocalPtr();
                    const auto groupSrcGetMultipolePtr = groupSrc.getMultipolePtr();
#pragma omp task depend(in:groupSrcGetMultipolePtr[0]) depend(commute:groupTargetGetLocalPtr[0]) default(shared) firstprivate(idxLevel, indexesVec, groupSrcPtr, groupTargetPtr)  priority(priorities.getM2LPriority(idxLevel))
                    {
                        kernelWrapper.M2LBetweenGroups(idxLevel, kernels[omp_get_thread_num()], *groupTargetPtr, *groupSrcPtr, std::move(*indexesVec));
                        delete indexesVec;
                    }
                });

                ++currentCellGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree){
        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
            const auto& upperCellGroup = inTree.getCellGroupsAtLevelTarget(idxLevel);
            auto& lowerCellGroup = inTree.getCellGroupsAtLevelTarget(idxLevel+1);

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

#pragma omp task depend(in:upperGroupGetLocalPtr[0]) depend(commute:lowerGroupGetLocalPtr[0]) default(shared) firstprivate(idxLevel, upperGroup, lowerGroup)  priority(priorities.getL2LPriority(idxLevel))
                {
                    kernelWrapper.L2L(idxLevel, kernels[omp_get_thread_num()], *upperGroup, *lowerGroup);
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
        if(configuration.getTreeHeight() > stopUpperLevel){
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

                const auto leafGroupObj = &(*currentLeafGroup);
                auto particleGroupObj = &(*currentParticleGroup);

                const auto leafGroupObjGetLocalPtr = leafGroupObj->getLocalPtr();
                auto particleGroupObjGetRhsPtr = particleGroupObj->getRhsPtr();

#pragma omp task depend(in:leafGroupObjGetLocalPtr[0]) depend(commute:particleGroupObjGetRhsPtr[0]) default(shared) firstprivate(leafGroupObj, particleGroupObj)  priority(priorities.getL2PPriority())
                {
                    kernelWrapper.L2P(kernels[omp_get_thread_num()], *leafGroupObj, *particleGroupObj);
                }

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
            auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroupTarget, configuration.getTreeHeight()-1, false, false);

            indexesForGroup.second.reserve(std::size(indexesForGroup.first) + std::size(indexesForGroup.first));
            indexesForGroup.second.insert(indexesForGroup.second.end(), indexesForGroup.first.begin(), indexesForGroup.first.end());

            auto indexesForSelfGroup = spacialSystem.getSelfListForBlock(*currentParticleGroupTarget);
            indexesForGroup.second.insert(indexesForGroup.second.end(), indexesForSelfGroup.begin(), indexesForSelfGroup.end());

            TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), particleGroupsSource,
                                                      std::distance(particleGroupsTarget.begin(), currentParticleGroupTarget), particleGroupsTarget,
                                           [&](auto& groupTarget, auto& groupSrc, const auto& indexes){
                assert(&groupTarget == &*currentParticleGroupTarget);

                auto groupSrcPtr = &groupSrc;
                auto groupTargetPtr = &groupTarget;

                auto groupSrcGetDataPtr = groupSrc.getDataPtr();
                auto groupTargetGetRhsPtr = groupTarget.getRhsPtr();

                auto indexesVec = TbfUtils::CreateNew(indexes.toStdVector());
#pragma omp task depend(commute:groupSrcGetDataPtr[0],groupTargetGetRhsPtr[0]) default(shared) firstprivate(indexesVec, groupSrcPtr, groupTargetPtr) priority(priorities.getP2PPriority())
                {
                    kernelWrapper.P2PBetweenGroupsTsm(kernels[omp_get_thread_num()], *groupTargetPtr, *groupSrcPtr, std::move(*indexesVec));
                    delete indexesVec;
                }
            });

            ++currentParticleGroupTarget;
        }
    }

    void increaseNumberOfKernels(){
        for(long int idxThread = kernels.size() ; idxThread < omp_get_max_threads() ; ++idxThread){
            kernels.emplace_back(kernels[0]);
        }
    }

public:
    explicit TbfOpenmpAlgorithmTsm(const SpacialConfiguration& inConfiguration, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(configuration);
        increaseNumberOfKernels();
    }

    template <class SourceKernelClass,
              typename = typename std::enable_if<!std::is_same<long int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value
                                                 && !std::is_same<int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value, void>::type>
    TbfOpenmpAlgorithmTsm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(std::forward<SourceKernelClass>(inKernel));
        increaseNumberOfKernels();
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::TbfOperations::TbfNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        increaseNumberOfKernels();

#pragma omp parallel
#pragma omp master
{

        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2M){
            P2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2M){
            M2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2L){
            M2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2L){
            L2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2P){
            L2P(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2P){
            P2P(inTree);
        }
#pragma omp taskwait
}// master
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        for(const auto& kernel : kernels){
            inFunc(kernel);
        }
    }
};

#endif
