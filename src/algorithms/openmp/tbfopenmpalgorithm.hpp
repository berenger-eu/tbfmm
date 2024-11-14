#ifndef TBFOPENMPALGORITHM_HPP
#define TBFOPENMPALGORITHM_HPP

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
class TbfOpenmpAlgorithm {
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

                const unsigned char* ptr_particleGroupObjGetDataPtr = reinterpret_cast<const unsigned char*>(&particleGroupObjGetDataPtr[0]);
                const unsigned char* ptr_leafGroupObjGetMultipolePtr = reinterpret_cast<const unsigned char*>(&leafGroupObjGetMultipolePtr[0]);

                auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_particleGroupObjGetDataPtr[0]) depend(commute:ptr_leafGroupObjGetMultipolePtr[0]) default(shared) firstprivate(particleGroupObj, leafGroupObj, kernelsPtr) priority(priorities.getP2MPriority())
                {
                    kernelWrapper.P2M(kernelsPtr[omp_get_thread_num()], *particleGroupObj, *leafGroupObj);
                }
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(TreeClass& inTree){
        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= stopUpperLevel ; --idxLevel){
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

                const unsigned char* ptr_lowerGroupGetMultipolePtr = reinterpret_cast<const unsigned char*>(&lowerGroupGetMultipolePtr[0]);
                const unsigned char* ptr_upperGroupGetMultipolePtr = reinterpret_cast<const unsigned char*>(&upperGroupGetMultipolePtr[0]);

                auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_lowerGroupGetMultipolePtr[0]) depend(commute:ptr_upperGroupGetMultipolePtr[0]) default(shared) firstprivate(upperGroup, lowerGroup, kernelsPtr)  priority(priorities.getM2MPriority(idxLevel))
                {
                    kernelWrapper.M2M(idxLevel, kernelsPtr[omp_get_thread_num()], *lowerGroup, *upperGroup);
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

                    const unsigned char* ptr_groupSrcGetMultipolePtr = reinterpret_cast<const unsigned char*>(&groupSrcGetMultipolePtr[0]);
                    const unsigned char* ptr_groupTargetGetLocalPtr = reinterpret_cast<const unsigned char*>(&groupTargetGetLocalPtr[0]);

                    auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_groupSrcGetMultipolePtr[0]) depend(commute:ptr_groupTargetGetLocalPtr[0]) default(shared) firstprivate(idxLevel, indexesVec, groupSrcPtr, groupTargetPtr, kernelsPtr)  priority(priorities.getM2LPriority(idxLevel))
                    {
                        kernelWrapper.M2LBetweenGroups(idxLevel, kernelsPtr[omp_get_thread_num()], *groupTargetPtr, *groupSrcPtr, std::move(*indexesVec));
                        delete indexesVec;
                    }
                });

                auto currentGroup = &(*currentCellGroup);
                auto indexesForGroup_first = TbfUtils::CreateNew(std::move(indexesForGroup.first));

                const auto currentGroupGetMultipolePtr = currentGroup->getMultipolePtr();
                auto currentGroupGetLocalPtr = currentGroup->getLocalPtr();

                const unsigned char* ptr_currentGroupGetMultipolePtr = reinterpret_cast<const unsigned char*>(&currentGroupGetMultipolePtr[0]);
                const unsigned char* ptr_currentGroupGetLocalPtr = reinterpret_cast<const unsigned char*>(&currentGroupGetLocalPtr[0]);

                auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_currentGroupGetMultipolePtr[0]) depend(commute:ptr_currentGroupGetLocalPtr[0]) default(shared) firstprivate(idxLevel, indexesForGroup_first, currentGroup, kernelsPtr)  priority(priorities.getM2LPriority(idxLevel))
                {
                    kernelWrapper.M2LInGroup(idxLevel, kernelsPtr[omp_get_thread_num()], *currentGroup, std::move(*indexesForGroup_first));
                    delete indexesForGroup_first;
                }

                ++currentCellGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree){
        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
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

                const unsigned char* ptr_upperGroupGetLocalPtr = reinterpret_cast<const unsigned char*>(&upperGroupGetLocalPtr[0]);
                const unsigned char* ptr_lowerGroupGetLocalPtr = reinterpret_cast<const unsigned char*>(&lowerGroupGetLocalPtr[0]);

                auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_upperGroupGetLocalPtr[0]) depend(commute:ptr_lowerGroupGetLocalPtr[0]) default(shared) firstprivate(idxLevel, upperGroup, lowerGroup, kernelsPtr)  priority(priorities.getL2LPriority(idxLevel))
                {
                    kernelWrapper.L2L(idxLevel, kernelsPtr[omp_get_thread_num()], *upperGroup, *lowerGroup);
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
                auto particleGroupObjGetDataPtr = particleGroupObj->getDataPtr();
                auto particleGroupObjGetRhsPtr = particleGroupObj->getRhsPtr();

                const unsigned char* ptr_leafGroupObjGetLocalPtr = reinterpret_cast<const unsigned char*>(&leafGroupObjGetLocalPtr[0]);
                const unsigned char* ptr_particleGroupObjGetDataPtr = reinterpret_cast<const unsigned char*>(&particleGroupObjGetDataPtr[0]);
                const unsigned char* ptr_particleGroupObjGetRhsPtr = reinterpret_cast<const unsigned char*>(&particleGroupObjGetRhsPtr[0]);

                auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_leafGroupObjGetLocalPtr[0],ptr_particleGroupObjGetDataPtr[0]) depend(commute:ptr_particleGroupObjGetRhsPtr[0]) default(shared) firstprivate(leafGroupObj, particleGroupObj, kernelsPtr)  priority(priorities.getL2PPriority())
                {
                    kernelWrapper.L2P(kernelsPtr[omp_get_thread_num()], *leafGroupObj, *particleGroupObj);
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
                auto groupSrcGetRhsPtr = groupSrc.getRhsPtr();
                auto groupTargetGetRhsPtr = groupTarget.getRhsPtr();
                auto groupTargetGetDataPtr = groupTarget.getDataPtr();

                auto indexesVec = TbfUtils::CreateNew(indexes.toStdVector());

                const unsigned char* ptr_groupSrcGetDataPtr = reinterpret_cast<const unsigned char*>(&groupSrcGetDataPtr[0]);
                const unsigned char* ptr_groupSrcGetRhsPtr = reinterpret_cast<const unsigned char*>(&groupSrcGetRhsPtr[0]);
                const unsigned char* ptr_groupTargetGetDataPtr = reinterpret_cast<const unsigned char*>(&groupTargetGetDataPtr[0]);
                const unsigned char* ptr_groupTargetGetRhsPtr = reinterpret_cast<const unsigned char*>(&groupTargetGetRhsPtr[0]);

                auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_groupSrcGetDataPtr[0],ptr_groupTargetGetDataPtr[0]) depend(commute:ptr_groupSrcGetRhsPtr[0],ptr_groupTargetGetRhsPtr[0]) default(shared) firstprivate(indexesVec, groupSrcPtr, groupTargetPtr, kernelsPtr) priority(priorities.getP2PPriority())
                {
                    kernelWrapper.P2PBetweenGroups(kernelsPtr[omp_get_thread_num()], *groupSrcPtr, *groupTargetPtr, std::move(*indexesVec));
                    delete indexesVec;
                }
            });

            auto currentGroup = &(*currentParticleGroup);

            const auto currentGroupGetDataPtr = currentGroup->getDataPtr();
            auto currentGroupGetRhsPtr = currentGroup->getRhsPtr();

            auto indexesForGroup_first = TbfUtils::CreateNew(std::move(indexesForGroup.first));

            const unsigned char* ptr_currentGroupGetDataPtr = reinterpret_cast<const unsigned char*>(&currentGroupGetDataPtr[0]);
            const unsigned char* ptr_currentGroupGetRhsPtr = reinterpret_cast<const unsigned char*>(&currentGroupGetRhsPtr[0]);

            auto* kernelsPtr = kernels.data();

#pragma omp task depend(in:ptr_currentGroupGetDataPtr[0]) depend(commute:ptr_currentGroupGetRhsPtr[0]) default(shared) firstprivate(currentGroup, indexesForGroup_first, kernelsPtr) priority(priorities.getP2PPriority())
            {
                kernelWrapper.P2PInGroup(kernelsPtr[omp_get_thread_num()], *currentGroup, std::move(*indexesForGroup_first));
                delete indexesForGroup_first;

                kernelWrapper.P2PInner(kernelsPtr[omp_get_thread_num()], *currentGroup);
            }

            ++currentParticleGroup;
        }
    }

    void increaseNumberOfKernels(){
        kernels.reserve(omp_get_max_threads());
        for(std::ptrdiff_t idxThread = kernels.size() ; idxThread < omp_get_max_threads() ; ++idxThread){
            kernels.emplace_back(kernels[0]);
        }
    }

public:
    explicit TbfOpenmpAlgorithm(const SpacialConfiguration& inConfiguration, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(configuration);
        increaseNumberOfKernels();
    }

    template <class SourceKernelClass,
              typename = typename std::enable_if<!std::is_same<long int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value
                                                 && !std::is_same<int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value, void>::type>
    TbfOpenmpAlgorithm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inStopUpperLevel = TbfDefaultLastLevel)
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
        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2P){
            P2P(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2P){
            L2P(inTree);
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

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfOpenmpAlgorithm& inAlgo) {
        inStream << "TbfOpenmpAlgorithm @ " << &inAlgo << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inAlgo.configuration << "\n";
        inStream << " - Space system: " << "\n";
        inStream << inAlgo.spaceSystem << "\n";
        return inStream;
    }

    static int GetNbThreads(){
        return omp_get_max_threads();
    }

    static const char* GetName(){
        return "TbfOpenmpAlgorithm";
    }
};

#endif
