#ifndef TBFSMSPECXALGORITHMTSM_HPP
#define TBFSMSPECXALGORITHMTSM_HPP

#include "tbfglobal.hpp"

#include "../sequential/tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"
#include "tbfsmstarpucallbacks.hpp"

#include <cassert>
#include <iterator>
#include <list>

#include <starpu.h>

#include "tbfsmstarpuutils.hpp"

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfSmStarpuAlgorithmTsm {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
    using ThisClass = TbfSmStarpuAlgorithmTsm<RealType_T, KernelClass_T, SpaceIndexType_T>;

    using CellSrcHandleContainer = typename TbfStarPUHandleBuilderTsm::CellSrcHandleContainer;
    using ParticleSrcHandleContainer = typename TbfStarPUHandleBuilderTsm::ParticleSrcHandleContainer;

    using CellTgtHandleContainer = typename TbfStarPUHandleBuilderTsm::CellTgtHandleContainer;
    using ParticleTgtHandleContainer = typename TbfStarPUHandleBuilderTsm::ParticleTgtHandleContainer;

    using VecOfIndexes = std::vector<TbfXtoXInteraction<typename SpaceIndexType::IndexType>>;
    std::list<VecOfIndexes> vecIndexBuffer;

    starpu_codelet p2m_cl;
    starpu_perfmodel p2m_cl_model;

    starpu_codelet m2m_cl;
    starpu_perfmodel m2m_cl_model;

    starpu_codelet l2l_cl;
    starpu_perfmodel l2l_cl_model;

    starpu_codelet l2l_cl_nocommute;
    starpu_perfmodel l2l_cl_nocommute_model;

    starpu_codelet l2p_cl;
    starpu_perfmodel l2p_cl_model;


    starpu_codelet m2l_cl_between_groups;
    starpu_perfmodel m2l_cl_between_groups_model;


    starpu_codelet p2p_cl_twoleaves;
    starpu_perfmodel p2p_cl_twoleaves_model;


    friend TbfSmStarpuCallbacks;

    template<class CellContainerClassSource, class ParticleContainerClassSource,
             class CellContainerClassTarget, class ParticleContainerClassTarget>
    void initCodelet(){
        memset(&p2m_cl, 0, sizeof(p2m_cl));
        p2m_cl.cpu_funcs[0] = &TbfSmStarpuCallbacks::P2MCallback<ThisClass, CellContainerClassSource, ParticleContainerClassSource>;
        p2m_cl.where |= STARPU_CPU;
        p2m_cl.nbuffers = 3;
        p2m_cl.modes[0] = STARPU_R;
        p2m_cl.modes[1] = STARPU_R;
        p2m_cl.modes[2] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2m_cl.name = "p2m_cl";

        memset(&m2m_cl, 0, sizeof(m2m_cl));
        m2m_cl.cpu_funcs[0] = &TbfSmStarpuCallbacks::M2MCallback<ThisClass, CellContainerClassSource>;
        m2m_cl.where |= STARPU_CPU;
        m2m_cl.nbuffers = 4;
        m2m_cl.modes[0] = STARPU_R;
        m2m_cl.modes[1] = STARPU_R;
        m2m_cl.modes[2] = STARPU_R;
        m2m_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2m_cl.name = "m2m_cl";

        memset(&l2l_cl, 0, sizeof(l2l_cl));
        l2l_cl.cpu_funcs[0] = &TbfSmStarpuCallbacks::L2LCallback<ThisClass, CellContainerClassTarget>;
        l2l_cl.where |= STARPU_CPU;
        l2l_cl.nbuffers = 4;
        l2l_cl.modes[0] = STARPU_R;
        l2l_cl.modes[1] = STARPU_R;
        l2l_cl.modes[2] = STARPU_R;
        l2l_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        l2l_cl.name = "l2l_cl";

        memset(&l2p_cl, 0, sizeof(l2p_cl));
        l2p_cl.cpu_funcs[0] = &TbfSmStarpuCallbacks::L2PCallback<ThisClass, CellContainerClassTarget, ParticleContainerClassTarget>;
        l2p_cl.where |= STARPU_CPU;
        l2p_cl.nbuffers = 4;
        l2p_cl.modes[0] = STARPU_R;
        l2p_cl.modes[1] = STARPU_R;
        l2p_cl.modes[2] = STARPU_R;
        l2p_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        l2p_cl.name = "l2p_cl";

        memset(&p2p_cl_twoleaves, 0, sizeof(p2p_cl_twoleaves));
        p2p_cl_twoleaves.cpu_funcs[0] = &TbfSmStarpuCallbacks::P2PTsmBetweenLeavesCallback<ThisClass, ParticleContainerClassSource, ParticleContainerClassTarget>;
        p2p_cl_twoleaves.where |= STARPU_CPU;
        p2p_cl_twoleaves.nbuffers = 3;
        p2p_cl_twoleaves.modes[0] = STARPU_R;
        p2p_cl_twoleaves.modes[1] = STARPU_R;
        p2p_cl_twoleaves.modes[2] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_twoleaves.name = "p2p_cl_twoleaves";

        memset(&m2l_cl_between_groups, 0, sizeof(m2l_cl_between_groups));
        m2l_cl_between_groups.cpu_funcs[0] = &TbfSmStarpuCallbacks::M2LTsmCallback<ThisClass, CellContainerClassSource, CellContainerClassTarget>;
        m2l_cl_between_groups.where |= STARPU_CPU;
        m2l_cl_between_groups.nbuffers = 4;
        m2l_cl_between_groups.modes[0] = STARPU_R;
        m2l_cl_between_groups.modes[1] = STARPU_R;
        m2l_cl_between_groups.modes[2] = STARPU_R;
        m2l_cl_between_groups.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2l_cl_between_groups.name = "m2l_cl_between_groups";

        p2m_cl_model.type = STARPU_HISTORY_BASED;
        p2m_cl_model.symbol = "p2m_cl";
        p2m_cl.model = &p2m_cl_model;
        m2m_cl_model.type = STARPU_HISTORY_BASED;
        m2m_cl_model.symbol = "m2m_cl";
        m2m_cl.model = &m2m_cl_model;
        l2l_cl_model.type = STARPU_HISTORY_BASED;
        l2l_cl_model.symbol = "l2l_cl";
        l2l_cl.model = &l2l_cl_model;
        l2l_cl_nocommute_model.type = STARPU_HISTORY_BASED;
        l2l_cl_nocommute_model.symbol = "l2l_cl_nocommute";
        l2l_cl_nocommute.model = &l2l_cl_nocommute_model;
        l2p_cl_model.type = STARPU_HISTORY_BASED;
        l2p_cl_model.symbol = "l2p_cl";
        l2p_cl.model = &l2p_cl_model;
        m2l_cl_between_groups_model.type = STARPU_HISTORY_BASED;
        m2l_cl_between_groups_model.symbol = "m2l_cl_between_groups";
        m2l_cl_between_groups.model = &m2l_cl_between_groups_model;
        p2p_cl_twoleaves_model.type = STARPU_HISTORY_BASED;
        p2p_cl_twoleaves_model.symbol = "p2p_cl_twoleaves";
        p2p_cl_twoleaves.model = &p2p_cl_twoleaves_model;
    }

    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;

    const long int stopUpperLevel;

    TbfGroupKernelInterface<SpaceIndexType> kernelWrapper;
    std::vector<KernelClass> kernels;

    TbfAlgorithmUtils::TbfOperationsPriorities priorities;

    template <class TreeClass>
    void P2M(TreeClass& inTree, CellSrcHandleContainer& cellSrcHandles, ParticleSrcHandleContainer& particleSrcHandles){
        if(configuration.getTreeHeight() > stopUpperLevel){
            auto& leafGroups = inTree.getLeafGroupsSource();
            const auto& particleGroups = inTree.getParticleGroupsSource();

            assert(std::size(leafGroups) == std::size(particleGroups));

            auto currentLeafGroup = leafGroups.begin();
            auto currentParticleGroup = particleGroups.cbegin();

            const auto endLeafGroup = leafGroups.end();
            const auto endParticleGroup = particleGroups.cend();
            int idxGroup = 0;

            while(currentLeafGroup != endLeafGroup && currentParticleGroup != endParticleGroup){
                assert((*currentParticleGroup).getStartingSpacialIndex() == (*currentLeafGroup).getStartingSpacialIndex()
                       && (*currentParticleGroup).getEndingSpacialIndex() == (*currentLeafGroup).getEndingSpacialIndex()
                       && (*currentParticleGroup).getNbLeaves() == (*currentLeafGroup).getNbCells());
                auto* thisptr = this;
                unsigned char* groupCellsData = inTree.getLeafGroupsSource()[idxGroup].getDataPtr();
                size_t groupCellsDataSize = inTree.getLeafGroupsSource()[idxGroup].getDataSize();
                unsigned char* groupParticlesData = inTree.getParticleGroupsSource()[idxGroup].getDataPtr();
                size_t groupParticlesDataSize = inTree.getParticleGroupsSource()[idxGroup].getDataSize();
                starpu_insert_task(&p2m_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &groupCellsData, sizeof(void*),
                                   STARPU_VALUE, &groupCellsDataSize, sizeof(size_t),
                                   STARPU_VALUE, &groupParticlesData, sizeof(void*),
                                   STARPU_VALUE, &groupParticlesDataSize, sizeof(size_t),
                                   STARPU_PRIORITY, priorities.getP2MPriority(),
                                   STARPU_R, particleSrcHandles[idxGroup][0],
                                   STARPU_R, cellSrcHandles[configuration.getTreeHeight()-1][idxGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellSrcHandles[configuration.getTreeHeight()-1][idxGroup][1],
                                   STARPU_NAME, "P2M",
                                   0);

                ++currentParticleGroup;
                ++currentLeafGroup;
                ++idxGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(TreeClass& inTree, CellSrcHandleContainer& cellSrcHandles){
        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= stopUpperLevel ; --idxLevel){
            auto& upperCellGroup = inTree.getCellGroupsAtLevelSource(idxLevel);
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevelSource(idxLevel+1);

            auto currentUpperGroup = upperCellGroup.begin();
            auto currentLowerGroup = lowerCellGroup.cbegin();

            const auto endUpperGroup = upperCellGroup.end();
            const auto endLowerGroup = lowerCellGroup.cend();

            int idxUpperGroup = 0;
            int idxLowerGroup = 0;

            while(currentUpperGroup != endUpperGroup && currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()
                       || currentUpperGroup->getStartingSpacialIndex() <= spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));
                auto* thisptr = this;
                unsigned char* groupCellsLowerData = inTree.getCellGroupsAtLevelSource(idxLevel+1)[idxLowerGroup].getDataPtr();
                size_t groupCellsLowerDataSize = inTree.getCellGroupsAtLevelSource(idxLevel+1)[idxLowerGroup].getDataSize();
                unsigned char* groupParticlesUpperData = inTree.getCellGroupsAtLevelSource(idxLevel)[idxUpperGroup].getDataPtr();
                size_t groupParticlesUpperDataSize = inTree.getCellGroupsAtLevelSource(idxLevel)[idxUpperGroup].getDataSize();
                starpu_insert_task(&m2m_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(long int),
                                   STARPU_VALUE, &groupCellsLowerData, sizeof(void*),
                                   STARPU_VALUE, &groupCellsLowerDataSize, sizeof(size_t),
                                   STARPU_VALUE, &groupParticlesUpperData, sizeof(void*),
                                   STARPU_VALUE, &groupParticlesUpperDataSize, sizeof(size_t),
                                   STARPU_PRIORITY, priorities.getM2MPriority(idxLevel),
                                   STARPU_R, cellSrcHandles[idxLevel+1][idxLowerGroup][0],
                                   STARPU_R, cellSrcHandles[idxLevel+1][idxLowerGroup][1],
                                   STARPU_R, cellSrcHandles[idxLevel][idxUpperGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellSrcHandles[idxLevel][idxUpperGroup][1],
                                   STARPU_NAME, "M2M",
                                   0);

                if(spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()){
                    ++currentLowerGroup;
                    ++idxLowerGroup;
                    if(currentLowerGroup != endLowerGroup && currentUpperGroup->getEndingSpacialIndex() < spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex())){
                        ++currentUpperGroup;
                        ++idxUpperGroup;
                    }
                }
                else{
                    ++currentUpperGroup;
                    ++idxUpperGroup;
                }
            }
        }
    }

    template <class TreeClass>
    void M2L(TreeClass& inTree, CellSrcHandleContainer& cellSrcHandles, CellTgtHandleContainer& cellTgtHandles){
        const auto& spacialSystem = inTree.getSpacialSystem();

        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
            auto& cellGroupsTarget = inTree.getCellGroupsAtLevelTarget(idxLevel);
            auto& cellGroupsSource = inTree.getCellGroupsAtLevelSource(idxLevel);

            auto currentCellGroup = cellGroupsTarget.begin();
            const auto endCellGroup = cellGroupsTarget.end();
            int idxGroup = 0;

            while(currentCellGroup != endCellGroup){
                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel, false);

                indexesForGroup.second.reserve(std::size(indexesForGroup.first) + std::size(indexesForGroup.first));
                indexesForGroup.second.insert(indexesForGroup.second.end(), indexesForGroup.first.begin(), indexesForGroup.first.end());

                TbfAlgorithmUtils::TbfMapIndexesAndBlocksIndexes(std::move(indexesForGroup.second), cellGroupsSource,
                                                                 std::distance(cellGroupsTarget.begin(),currentCellGroup), cellGroupsTarget,
                                                                 [&](auto& groupTargetIdx, const auto& groupSrcIdx, const auto& indexes){
                                                                     auto* thisptr = this;
                                                                     vecIndexBuffer.push_back(indexes.toStdVector());
                                                                     VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
                                                                     unsigned char* groupCellsSrcData = inTree.getCellGroupsAtLevelSource(idxLevel)[groupSrcIdx].getDataPtr();
                                                                     size_t groupCellsDataSrcSize = inTree.getCellGroupsAtLevelSource(idxLevel)[groupSrcIdx].getDataSize();
                                                                     unsigned char* groupCellsTgtData = inTree.getCellGroupsAtLevelTarget(idxLevel)[groupTargetIdx].getDataPtr();
                                                                     size_t groupCellsDataTgtSize = inTree.getCellGroupsAtLevelTarget(idxLevel)[groupTargetIdx].getDataSize();
                                                                     starpu_insert_task(&m2l_cl_between_groups,
                                                                                        STARPU_VALUE, &thisptr, sizeof(void*),
                                                                                        STARPU_VALUE, &idxLevel, sizeof(int),
                                                                                        STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                                                                                        STARPU_VALUE, &groupCellsSrcData, sizeof(void*),
                                                                                        STARPU_VALUE, &groupCellsDataSrcSize, sizeof(size_t),
                                                                                        STARPU_VALUE, &groupCellsTgtData, sizeof(void*),
                                                                                        STARPU_VALUE, &groupCellsDataTgtSize, sizeof(size_t),
                                                                                        STARPU_PRIORITY, priorities.getM2LPriority(idxLevel),
                                                                                        STARPU_R, cellSrcHandles[idxLevel][groupSrcIdx][0],
                                                                                        STARPU_R, cellSrcHandles[idxLevel][groupSrcIdx][1],
                                                                                        STARPU_R, cellTgtHandles[idxLevel][groupTargetIdx][0],
                                                                                        starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellTgtHandles[idxLevel][groupTargetIdx][1],
                                                                                        STARPU_NAME, "M2L",
                                                                                        0);
                                                                 });

                ++currentCellGroup;
                ++idxGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree, CellTgtHandleContainer& cellTgtHandles){
        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
            const auto& upperCellGroup = inTree.getCellGroupsAtLevelTarget(idxLevel);
            auto& lowerCellGroup = inTree.getCellGroupsAtLevelTarget(idxLevel+1);

            auto currentUpperGroup = upperCellGroup.cbegin();
            auto currentLowerGroup = lowerCellGroup.begin();

            const auto endUpperGroup = upperCellGroup.cend();
            const auto endLowerGroup = lowerCellGroup.end();

            int idxUpperGroup = 0;
            int idxLowerGroup = 0;

            while(currentUpperGroup != endUpperGroup && currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()
                       || currentUpperGroup->getStartingSpacialIndex() <= spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));
                auto* thisptr = this;
                unsigned char* groupParticlesUpperData = inTree.getCellGroupsAtLevelTarget(idxLevel)[idxUpperGroup].getDataPtr();
                size_t groupParticlesUpperDataSize = inTree.getCellGroupsAtLevelTarget(idxLevel)[idxUpperGroup].getDataSize();
                unsigned char* groupCellsLowerData = inTree.getCellGroupsAtLevelTarget(idxLevel+1)[idxLowerGroup].getDataPtr();
                size_t groupCellsLowerDataSize = inTree.getCellGroupsAtLevelTarget(idxLevel+1)[idxLowerGroup].getDataSize();
                starpu_insert_task(&l2l_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(long int),
                                   STARPU_VALUE, &groupParticlesUpperData, sizeof(void*),
                                   STARPU_VALUE, &groupParticlesUpperDataSize, sizeof(size_t),
                                   STARPU_VALUE, &groupCellsLowerData, sizeof(void*),
                                   STARPU_VALUE, &groupCellsLowerDataSize, sizeof(size_t),
                                   STARPU_PRIORITY, priorities.getL2LPriority(idxLevel),
                                   STARPU_R, cellTgtHandles[idxLevel][idxUpperGroup][0],
                                   STARPU_R, cellTgtHandles[idxLevel][idxUpperGroup][1],
                                   STARPU_R, cellTgtHandles[idxLevel+1][idxLowerGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellTgtHandles[idxLevel+1][idxLowerGroup][1],
                                   STARPU_NAME, "L2L",
                                   0);

                if(spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()) <= currentUpperGroup->getEndingSpacialIndex()){
                    ++currentLowerGroup;
                    ++idxLowerGroup;
                    if(currentLowerGroup != endLowerGroup && currentUpperGroup->getEndingSpacialIndex() < spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex())){
                        ++currentUpperGroup;
                        ++idxUpperGroup;
                    }
                }
                else{
                    ++currentUpperGroup;
                    ++idxUpperGroup;
                }
            }
        }
    }

    template <class TreeClass>
    void L2P(TreeClass& inTree, CellTgtHandleContainer& cellTgtHandles, ParticleTgtHandleContainer& particleTgtHandles){
        if(configuration.getTreeHeight() > stopUpperLevel){
            const auto& leafGroups = inTree.getLeafGroupsTarget();
            auto& particleGroups = inTree.getParticleGroupsTarget();

            assert(std::size(leafGroups) == std::size(particleGroups));

            auto currentLeafGroup = leafGroups.cbegin();
            auto currentParticleGroup = particleGroups.begin();

            const auto endLeafGroup = leafGroups.cend();
            const auto endParticleGroup = particleGroups.end();

            int idxGroup = 0;

            while(currentLeafGroup != endLeafGroup && currentParticleGroup != endParticleGroup){
                assert((*currentParticleGroup).getStartingSpacialIndex() == (*currentLeafGroup).getStartingSpacialIndex()
                       && (*currentParticleGroup).getEndingSpacialIndex() == (*currentLeafGroup).getEndingSpacialIndex()
                       && (*currentParticleGroup).getNbLeaves() == (*currentLeafGroup).getNbCells());

                auto* thisptr = this;
                unsigned char* groupCellsData = inTree.getLeafGroupsTarget()[idxGroup].getDataPtr();
                size_t groupCellsDataSize = inTree.getLeafGroupsTarget()[idxGroup].getDataSize();
                unsigned char* groupParticlesData = inTree.getParticleGroupsTarget()[idxGroup].getDataPtr();
                size_t groupParticlesDataSize = inTree.getParticleGroupsTarget()[idxGroup].getDataSize();
                starpu_insert_task(&l2p_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &groupCellsData, sizeof(void*),
                                   STARPU_VALUE, &groupCellsDataSize, sizeof(size_t),
                                   STARPU_VALUE, &groupParticlesData, sizeof(void*),
                                   STARPU_VALUE, &groupParticlesDataSize, sizeof(size_t),
                                   STARPU_PRIORITY, priorities.getL2PPriority(),
                                   STARPU_R, cellTgtHandles[configuration.getTreeHeight()-1][idxGroup][0],
                                   STARPU_R, cellTgtHandles[configuration.getTreeHeight()-1][idxGroup][1],
                                   STARPU_R, particleTgtHandles[idxGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleTgtHandles[idxGroup][1],
                                   STARPU_NAME, "L2P",
                                   0);

                ++currentParticleGroup;
                ++currentLeafGroup;
                ++idxGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(TreeClass& inTree, ParticleSrcHandleContainer& particleSrcHandles, ParticleTgtHandleContainer& particleTgtHandles){
        const auto& spacialSystem = inTree.getSpacialSystem();

        auto& particleGroupsTarget = inTree.getParticleGroupsTarget();
        auto& particleGroupsSource = inTree.getParticleGroupsSource();

        auto currentParticleGroupTarget = particleGroupsTarget.begin();
        const auto endParticleGroupTarget = particleGroupsTarget.end();

        int idxGroup = 0;

        while(currentParticleGroupTarget != endParticleGroupTarget){
            auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroupTarget, configuration.getTreeHeight()-1, false, false);

            indexesForGroup.second.reserve(std::size(indexesForGroup.first) + std::size(indexesForGroup.first));
            indexesForGroup.second.insert(indexesForGroup.second.end(), indexesForGroup.first.begin(), indexesForGroup.first.end());

            auto indexesForSelfGroup = spacialSystem.getSelfListForBlock(*currentParticleGroupTarget);
            indexesForGroup.second.insert(indexesForGroup.second.end(), indexesForSelfGroup.begin(), indexesForSelfGroup.end());

            TbfAlgorithmUtils::TbfMapIndexesAndBlocksIndexes(std::move(indexesForGroup.second), particleGroupsSource,
                                                             std::distance(particleGroupsTarget.begin(), currentParticleGroupTarget), particleGroupsTarget,
                                                             [&](auto& groupTargetIdx, auto& groupSrcIdx, const auto& indexes){
                                                                 auto* thisptr = this;
                                                                 vecIndexBuffer.push_back(indexes.toStdVector());
                                                                 VecOfIndexes* vecIndexesPtr = &vecIndexBuffer.back();
                                                                 unsigned char* srcData = inTree.getParticleGroupsSource()[groupSrcIdx].getDataPtr();
                                                                 size_t srcDataSize = inTree.getParticleGroupsSource()[groupSrcIdx].getDataSize();
                                                                 unsigned char* tgtData = inTree.getParticleGroupsTarget()[groupTargetIdx].getDataPtr();
                                                                 size_t tgtDataSize = inTree.getParticleGroupsTarget()[groupTargetIdx].getDataSize();
                                                                 starpu_insert_task(&p2p_cl_twoleaves,
                                                                                    STARPU_VALUE, &thisptr, sizeof(void*),
                                                                                    STARPU_VALUE, &vecIndexesPtr, sizeof(void*),
                                                                                    STARPU_VALUE, &srcData, sizeof(void*),
                                                                                    STARPU_VALUE, &srcDataSize, sizeof(size_t),
                                                                                    STARPU_VALUE, &tgtData, sizeof(void*),
                                                                                    STARPU_VALUE, &tgtDataSize, sizeof(size_t),
                                                                                    STARPU_PRIORITY, priorities.getP2PPriority(),
                                                                                    STARPU_R, particleSrcHandles[groupSrcIdx][0],
                                                                                    STARPU_R, particleTgtHandles[groupTargetIdx][0],
                                                                                    starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleTgtHandles[groupTargetIdx][1],
                                                                                    STARPU_NAME, "P2P-INOUT",
                                                                                    0);
                                                             });


            ++currentParticleGroupTarget;
            ++idxGroup;
        }
    }

    void increaseNumberOfKernels(const int inNbThreads){
        for(long int idxThread = kernels.size() ; idxThread < inNbThreads ; ++idxThread){
            kernels.emplace_back(kernels[0]);
        }
    }

public:
    explicit TbfSmStarpuAlgorithmTsm(const SpacialConfiguration& inConfiguration, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
        kernelWrapper(configuration),
        priorities(configuration.getTreeHeight()){
        kernels.emplace_back(configuration);

        [[maybe_unused]] const int ret = starpu_init(NULL);
        assert(ret == 0);
        starpu_pause();
    }

    template <class SourceKernelClass,
             typename = typename std::enable_if<!std::is_same<long int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value
                                                    && !std::is_same<int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value, void>::type>
    TbfSmStarpuAlgorithmTsm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
        kernelWrapper(configuration),
        priorities(configuration.getTreeHeight()){
        kernels.emplace_back(std::forward<SourceKernelClass>(inKernel));

        [[maybe_unused]] const int ret = starpu_init(NULL);
        assert(ret == 0);

        pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
        TbStarPUUtils::ExecOnWorkers(STARPU_CUDA|STARPU_CPU, [&](){
            pthread_mutex_lock(&lock);
            increaseNumberOfKernels(starpu_worker_get_id()+1);
            pthread_mutex_unlock(&lock);
        });
        pthread_mutex_destroy(&lock);

        starpu_pause();
    }

    ~TbfSmStarpuAlgorithmTsm(){
        starpu_resume();
        starpu_shutdown();
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::TbfOperations::TbfNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        auto allCellSrcHandles = TbfStarPUHandleBuilderTsm::GetCellSrcHandles(inTree, configuration);
        auto allParticlesSrcHandles = TbfStarPUHandleBuilderTsm::GetParticleSrcHandles(inTree);

        auto allCellTgtHandles = TbfStarPUHandleBuilderTsm::GetCellTgtHandles(inTree, configuration);
        auto allParticlesTgtHandles = TbfStarPUHandleBuilderTsm::GetParticleTgtHandles(inTree);

        using CellContainerClassSource = typename TreeClass::CellGroupClassSource;
        using ParticleContainerClassSource = typename TreeClass::LeafGroupClassSource;

        using CellContainerClassTarget = typename TreeClass::CellGroupClassTarget;
        using ParticleContainerClassTarget = typename TreeClass::LeafGroupClassTarget;

        initCodelet<CellContainerClassSource, ParticleContainerClassSource,
                    CellContainerClassTarget, ParticleContainerClassTarget>();

        starpu_resume();

        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2M){
            P2M(inTree, allCellSrcHandles, allParticlesSrcHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2M){
            M2M(inTree, allCellSrcHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2L){
            M2L(inTree, allCellSrcHandles, allCellTgtHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2L){
            L2L(inTree, allCellTgtHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2P){
            P2P(inTree, allParticlesSrcHandles, allParticlesTgtHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2P){
            L2P(inTree, allCellTgtHandles, allParticlesTgtHandles);
        }

        starpu_task_wait_for_all();

        starpu_pause();
        vecIndexBuffer.clear();
        TbfStarPUHandleBuilderTsm::CleanCellHandles(allCellSrcHandles);
        TbfStarPUHandleBuilderTsm::CleanParticleHandles(allParticlesSrcHandles);
        TbfStarPUHandleBuilderTsm::CleanCellHandles(allCellTgtHandles);
        TbfStarPUHandleBuilderTsm::CleanParticleHandles(allParticlesTgtHandles);
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        for(const auto& kernel : kernels){
            inFunc(kernel);
        }
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfSmStarpuAlgorithmTsm& inAlgo) {
        inStream << "TbfSmStarpuAlgorithmTsm @ " << &inAlgo << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inAlgo.configuration << "\n";
        inStream << " - Space system: " << "\n";
        inStream << inAlgo.spaceSystem << "\n";
        inStream << " - Total workers: " << starpu_worker_get_count() << "\n";
        inStream << "  - CPU " << starpu_cpu_worker_get_count() << "\n";
        return inStream;
    }

    static int GetNbThreads(){
        return starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    }

    static const char* GetName(){
        return "TbfSmStarpuAlgorithmTsm";
    }
};

#endif
