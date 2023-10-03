#ifndef TBFSMSPECXALGORITHM_HPP
#define TBFSMSPECXALGORITHM_HPP

#include "tbfglobal.hpp"

#include "../sequential/tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"


#include <cassert>
#include <iterator>
#include <list>

#include <starpu.h>

#include "tbfsmstarpuutils.hpp"

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfSmStarpuAlgorithm {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
    using ThisClass = TbfSmStarpuAlgorithm<RealType_T, KernelClass_T, SpaceIndexType_T>;

    using CellHandleContainer = typename TbfStarPUHandleBuilder::CellHandleContainer;
    using ParticleHandleContainer = typename TbfStarPUHandleBuilder::ParticleHandleContainer;

    using VecOfIndexes = std::vector<TbfXtoXInteraction<typename SpaceIndexType::IndexType>>;
    std::list<VecOfIndexes> vecIndexBuffer;

    starpu_codelet p2m_cl;
    starpu_codelet m2m_cl;
    starpu_codelet l2l_cl;
    starpu_codelet l2l_cl_nocommute;
    starpu_codelet l2p_cl;

    starpu_codelet m2l_cl_between_groups;
    starpu_codelet m2l_cl_inside;

    starpu_codelet p2p_cl_oneleaf;
    starpu_codelet p2p_cl_twoleaves;

    template<class CellContainerClass, class ParticleContainerClass>
    static void P2MCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        starpu_codelet_unpack_args(cl_arg, &thisptr);

        unsigned char* particleData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* leafData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t leafDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* leafMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t leafMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        CellContainerClass leafGroupObj(leafData, leafDataSize,
                                        leafMultipole, leafMultipoleSize,
                                        nullptr, 0);
        const ParticleContainerClass particleGroupObj(particleData, particleDataSize,
                                                nullptr, 0);

        thisptr->kernelWrapper.P2M(thisptr->kernels[starpu_worker_get_id()], particleGroupObj, leafGroupObj);
    }

    template<class ParticleContainerClass>
    static void P2PBetweenLeavesCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        VecOfIndexes* indexesForGroup_first;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* tgtRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t tgtRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        ParticleContainerClass groupSrc(srcData, srcDataSize,
                                        srcRhs, srcRhsSize);
        ParticleContainerClass groupTarget(tgtData, tgtDataSize,
                                           tgtRhs, tgtRhsSize);

        thisptr->kernelWrapper.P2PBetweenGroups(thisptr->kernels[starpu_worker_get_id()], groupTarget, groupSrc, *indexesForGroup_first);
    }

    template<class ParticleContainerClass>
    static void P2POneLeafCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        VecOfIndexes* indexesForGroup_first;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first);

        unsigned char* particleData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* particleRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t particleRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        ParticleContainerClass currentGroup(particleData, particleDataSize,
                                            particleRhs, particleRhsSize);

        thisptr->kernelWrapper.P2PInGroup(thisptr->kernels[starpu_worker_get_id()], currentGroup, *indexesForGroup_first);
        thisptr->kernelWrapper.P2PInner(thisptr->kernels[starpu_worker_get_id()], currentGroup);
    }

    template<class CellContainerClass>
    static void M2MCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        long int idxLevel;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel);

        unsigned char* lowerData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t lowerDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* lowerMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t lowerMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* upperData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t upperDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* upperMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t upperMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass lowerGroupObj(lowerData, lowerDataSize, lowerMultipole, lowerMultipoleSize,
                                               nullptr, 0);
        CellContainerClass upperGroupObj(upperData, upperDataSize, upperMultipole, upperMultipoleSize,
                                         nullptr, 0);

        thisptr->kernelWrapper.M2M(idxLevel, thisptr->kernels[starpu_worker_get_id()], lowerGroupObj, upperGroupObj);
    }

    template<class CellContainerClass>
    static void M2LCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        VecOfIndexes* indexesForGroup_first;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* tgtLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t tgtLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass groupSrc(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                          nullptr, 0);
        CellContainerClass groupTarget(tgtData, tgtDataSize, nullptr, 0, tgtLocal, tgtLocalSize);

        thisptr->kernelWrapper.M2LBetweenGroups(idxLevel, thisptr->kernels[starpu_worker_get_id()], groupTarget, groupSrc, *indexesForGroup_first);
    }

    template<class CellContainerClass>
    static void M2LInnerCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        VecOfIndexes* indexesForGroup_first;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* srcLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t srcLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        CellContainerClass currentGroup(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                         srcLocal, srcLocalSize);

        thisptr->kernelWrapper.M2LInGroup(idxLevel, thisptr->kernels[starpu_worker_get_id()], currentGroup, *indexesForGroup_first);
    }

    template<class CellContainerClass>
    static void L2LCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        long int idxLevel;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel);

        unsigned char* upperData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t upperDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* upperLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t upperLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* lowerData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t lowerDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* lowerLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t lowerLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass upperGroupObj(upperData, upperDataSize, nullptr, 0, upperLocal, upperLocalSize);
        CellContainerClass lowerGroupObj(lowerData, lowerDataSize, nullptr, 0, lowerLocal, lowerLocalSize);

        thisptr->kernelWrapper.L2L(idxLevel, thisptr->kernels[starpu_worker_get_id()], upperGroupObj, lowerGroupObj);
    }

    template<class CellContainerClass, class ParticleContainerClass>
    static void L2PCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        starpu_codelet_unpack_args(cl_arg, &thisptr);

        unsigned char* leafData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t leafDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* leafLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t leafLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* particleData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* particleRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t particleRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass leafGroupObj(leafData, leafDataSize, nullptr, 0, leafLocal, leafLocalSize);
        ParticleContainerClass particleGroupObj(particleData, particleDataSize, particleRhs, particleRhsSize);

        thisptr->kernelWrapper.L2P(thisptr->kernels[starpu_worker_get_id()], leafGroupObj, particleGroupObj);
    }

    template<class CellContainerClass, class ParticleContainerClass>
    void initCodelet(){
        memset(&p2m_cl, 0, sizeof(p2m_cl));
        p2m_cl.cpu_funcs[0] = &P2MCallback<CellContainerClass, ParticleContainerClass>;
        p2m_cl.where |= STARPU_CPU;
        p2m_cl.nbuffers = 3;
        p2m_cl.modes[0] = STARPU_R;
        p2m_cl.modes[1] = STARPU_R;
        p2m_cl.modes[2] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2m_cl.name = "p2m_cl";

        memset(&m2m_cl, 0, sizeof(m2m_cl));
        m2m_cl.cpu_funcs[0] = &M2MCallback<CellContainerClass>;
        m2m_cl.where |= STARPU_CPU;
        m2m_cl.nbuffers = 4;
        m2m_cl.modes[0] = STARPU_R;
        m2m_cl.modes[1] = STARPU_R;
        m2m_cl.modes[2] = STARPU_R;
        m2m_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2m_cl.name = "m2m_cl";

        memset(&l2l_cl, 0, sizeof(l2l_cl));
        l2l_cl.cpu_funcs[0] = &L2LCallback<CellContainerClass>;
        l2l_cl.where |= STARPU_CPU;
        l2l_cl.nbuffers = 4;
        l2l_cl.modes[0] = STARPU_R;
        l2l_cl.modes[1] = STARPU_R;
        l2l_cl.modes[2] = STARPU_R;
        l2l_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        l2l_cl.name = "l2l_cl";

        memset(&l2p_cl, 0, sizeof(l2p_cl));
        l2p_cl.cpu_funcs[0] = &L2PCallback<CellContainerClass, ParticleContainerClass>;
        l2p_cl.where |= STARPU_CPU;
        l2p_cl.nbuffers = 4;
        l2p_cl.modes[0] = STARPU_R;
        l2p_cl.modes[1] = STARPU_R;
        l2p_cl.modes[2] = STARPU_R;
        l2p_cl.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        l2p_cl.name = "l2p_cl";

        memset(&p2p_cl_oneleaf, 0, sizeof(p2p_cl_oneleaf));
        p2p_cl_oneleaf.cpu_funcs[0] = &P2POneLeafCallback<ParticleContainerClass>;
        p2p_cl_oneleaf.where |= STARPU_CPU;
        p2p_cl_oneleaf.nbuffers = 2;
        p2p_cl_oneleaf.modes[0] = STARPU_R;
        p2p_cl_oneleaf.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_oneleaf.name = "p2p_cl_oneleaf";

        memset(&p2p_cl_twoleaves, 0, sizeof(p2p_cl_twoleaves));
        p2p_cl_twoleaves.cpu_funcs[0] = &P2PBetweenLeavesCallback<ParticleContainerClass>;
        p2p_cl_twoleaves.where |= STARPU_CPU;
        p2p_cl_twoleaves.nbuffers = 4;
        p2p_cl_twoleaves.modes[0] = STARPU_R;
        p2p_cl_twoleaves.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_twoleaves.modes[2] = STARPU_R;
        p2p_cl_twoleaves.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_twoleaves.name = "p2p_cl_twoleaves";

        memset(&m2l_cl_between_groups, 0, sizeof(m2l_cl_between_groups));
        m2l_cl_between_groups.cpu_funcs[0] = M2LCallback<CellContainerClass>;
        m2l_cl_between_groups.where |= STARPU_CPU;
        m2l_cl_between_groups.nbuffers = 4;
        m2l_cl_between_groups.modes[0] = STARPU_R;
        m2l_cl_between_groups.modes[1] = STARPU_R;
        m2l_cl_between_groups.modes[2] = STARPU_R;
        m2l_cl_between_groups.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2l_cl_between_groups.name = "m2l_cl_between_groups";

        memset(&m2l_cl_inside, 0, sizeof(m2l_cl_inside));
        m2l_cl_inside.cpu_funcs[0] = M2LInnerCallback<CellContainerClass>;
        m2l_cl_inside.where |= STARPU_CPU;
        m2l_cl_inside.nbuffers = 3;
        m2l_cl_inside.modes[0] = STARPU_R;
        m2l_cl_inside.modes[1] = STARPU_R;
        m2l_cl_inside.modes[2] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2l_cl_inside.name = "m2l_cl_inside";
    }

    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;

    const long int stopUpperLevel;

    TbfGroupKernelInterface<SpaceIndexType> kernelWrapper;
    std::vector<KernelClass> kernels;

    TbfAlgorithmUtils::TbfOperationsPriorities priorities;

    template <class TreeClass>
    void P2M(TreeClass& inTree, CellHandleContainer& cellHandles, ParticleHandleContainer& particleHandles){
        using CellContainerClass = typename TreeClass::CellGroupClass;
        using ParticleContainerClass = typename TreeClass::LeafGroupClass;
        if(configuration.getTreeHeight() > stopUpperLevel){
            auto& leafGroups = inTree.getLeafGroups();
            const auto& particleGroups = inTree.getParticleGroups();

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
                starpu_insert_task(&p2m_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getP2MPriority(),
                                   STARPU_R, particleHandles[idxGroup][0],
                                   STARPU_R, cellHandles[configuration.getTreeHeight()-1][idxGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellHandles[configuration.getTreeHeight()-1][idxGroup][1],
                                   STARPU_NAME, "P2M",
                                   0);

                ++currentParticleGroup;
                ++currentLeafGroup;
                ++idxGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(TreeClass& inTree, CellHandleContainer& cellHandles){
        using CellContainerClass = typename TreeClass::CellGroupClass;
        for(long int idxLevel = configuration.getTreeHeight()-2 ; idxLevel >= stopUpperLevel ; --idxLevel){
            auto& upperCellGroup = inTree.getCellGroupsAtLevel(idxLevel);
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevel+1);

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
                starpu_insert_task(&m2m_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(long int),
                                   STARPU_PRIORITY, priorities.getM2MPriority(idxLevel),
                                   STARPU_R, cellHandles[idxLevel+1][idxLowerGroup][0],
                                   STARPU_R, cellHandles[idxLevel+1][idxLowerGroup][1],
                                   STARPU_R, cellHandles[idxLevel][idxUpperGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellHandles[idxLevel][idxUpperGroup][1],
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
    void M2L(TreeClass& inTree, CellHandleContainer& cellHandles){
        using CellContainerClass = typename TreeClass::CellGroupClass;
        const auto& spacialSystem = inTree.getSpacialSystem();

        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();
            int idxGroup = 0;

            while(currentCellGroup != endCellGroup){
                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel);
                TbfAlgorithmUtils::TbfMapIndexesAndBlocksIndexes(std::move(indexesForGroup.second), cellGroups, std::distance(cellGroups.begin(),currentCellGroup),
                                               [&](auto& groupTargetIdx, const auto& groupSrcIdx, const auto& indexes){
                    auto* thisptr = this;
                    vecIndexBuffer.push_back(indexes.toStdVector());
                    VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
                    starpu_insert_task(&m2l_cl_between_groups,
                                       STARPU_VALUE, &thisptr, sizeof(void*),
                                       STARPU_VALUE, &idxLevel, sizeof(int),
                                       STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                                       STARPU_PRIORITY, priorities.getM2LPriority(idxLevel),
                                       STARPU_R, cellHandles[idxLevel][groupSrcIdx][0],
                                       STARPU_R, cellHandles[idxLevel][groupSrcIdx][1],
                                       STARPU_R, cellHandles[idxLevel][groupTargetIdx][0],
                                       starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellHandles[idxLevel][groupTargetIdx][2],
                                       STARPU_NAME, "M2L",
                                       0);
                });

                auto* thisptr = this;
                vecIndexBuffer.push_back(std::move(indexesForGroup.first));
                VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
                starpu_insert_task(&m2l_cl_inside,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(int),
                                   STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getM2LPriority(idxLevel),
                                   STARPU_R, cellHandles[idxLevel][idxGroup][0],
                                   STARPU_R, cellHandles[idxLevel][idxGroup][1],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellHandles[idxLevel][idxGroup][2],
                                   STARPU_NAME, "M2L-IN",
                                   0);

                ++currentCellGroup;
                ++idxGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree, CellHandleContainer& cellHandles){
        using CellContainerClass = typename TreeClass::CellGroupClass;
        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
            const auto& upperCellGroup = inTree.getCellGroupsAtLevel(idxLevel);
            auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevel+1);

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
                starpu_insert_task(&l2l_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(long int),
                                   STARPU_PRIORITY, priorities.getL2LPriority(idxLevel),
                                   STARPU_R, cellHandles[idxLevel][idxUpperGroup][0],
                                   STARPU_R, cellHandles[idxLevel][idxUpperGroup][2],
                                   STARPU_R, cellHandles[idxLevel+1][idxLowerGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellHandles[idxLevel+1][idxLowerGroup][2],
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
    void L2P(TreeClass& inTree, CellHandleContainer& cellHandles, ParticleHandleContainer& particleHandles){
        using CellContainerClass = typename TreeClass::CellGroupClass;
        using ParticleContainerClass = typename TreeClass::LeafGroupClass;
        if(configuration.getTreeHeight() > stopUpperLevel){
            const auto& leafGroups = inTree.getLeafGroups();
            auto& particleGroups = inTree.getParticleGroups();

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
                starpu_insert_task(&l2p_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getL2PPriority(),
                                   STARPU_R, cellHandles[configuration.getTreeHeight()-1][idxGroup][0],
                                   STARPU_R, cellHandles[configuration.getTreeHeight()-1][idxGroup][2],
                                   STARPU_R, particleHandles[idxGroup][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleHandles[idxGroup][1],
                                   STARPU_NAME, "L2P",
                                   0);

                ++currentParticleGroup;
                ++currentLeafGroup;
                ++idxGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(TreeClass& inTree, ParticleHandleContainer& particleHandles){
        using ParticleContainerClass = typename TreeClass::LeafGroupClass;
        const auto& spacialSystem = inTree.getSpacialSystem();

        auto& particleGroups = inTree.getParticleGroups();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        int idxGroup = 0;

        while(currentParticleGroup != endParticleGroup){

            auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroup, configuration.getTreeHeight()-1, true);
            TbfAlgorithmUtils::TbfMapIndexesAndBlocksIndexes(std::move(indexesForGroup.second), particleGroups, std::distance(particleGroups.begin(), currentParticleGroup),
                                           [&](auto& groupTargetIdx, auto& groupSrcIdx, const auto& indexes){
                auto* thisptr = this;
                vecIndexBuffer.push_back(indexes.toStdVector());
                VecOfIndexes* vecIndexesPtr = &vecIndexBuffer.back();
                starpu_insert_task(&p2p_cl_twoleaves,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &vecIndexesPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getP2PPriority(),
                                   STARPU_R, particleHandles[groupSrcIdx][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleHandles[groupSrcIdx][1],
                                   STARPU_R, particleHandles[groupTargetIdx][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleHandles[groupTargetIdx][1],
                                   STARPU_NAME, "P2P-INOUT",
                                   0);
            });

            auto* thisptr = this;
            vecIndexBuffer.push_back(std::move(indexesForGroup.first));
            VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
            starpu_insert_task(&p2p_cl_oneleaf,
                               STARPU_VALUE, &thisptr, sizeof(void*),
                               STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                               STARPU_PRIORITY, priorities.getP2PPriority(),
                               STARPU_R, particleHandles[idxGroup][0],
                               starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleHandles[idxGroup][1],
                               STARPU_NAME, "P2P",
                               0);

            ++currentParticleGroup;
            ++idxGroup;
        }
    }

    void increaseNumberOfKernels(const int inNbThreads){
        for(long int idxThread = kernels.size() ; idxThread < inNbThreads ; ++idxThread){
            kernels.emplace_back(kernels[0]);
        }
    }

public:
    explicit TbfSmStarpuAlgorithm(const SpacialConfiguration& inConfiguration, const long int inStopUpperLevel = TbfDefaultLastLevel)
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
    TbfSmStarpuAlgorithm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(std::forward<SourceKernelClass>(inKernel));

        [[maybe_unused]] const int ret = starpu_init(NULL);
        assert(ret == 0);
        starpu_pause();
    }

    ~TbfSmStarpuAlgorithm(){
        starpu_resume();
        starpu_shutdown();
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::TbfOperations::TbfNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        auto allCellHandles = TbfStarPUHandleBuilder::GetCellHandles(inTree, configuration);
        auto allParticlesHandles = TbfStarPUHandleBuilder::GetParticleHandles(inTree);

        using CellContainerClass = typename TreeClass::CellGroupClass;
        using ParticleContainerClass = typename TreeClass::LeafGroupClass;

        initCodelet<CellContainerClass, ParticleContainerClass>();

        starpu_resume();

        increaseNumberOfKernels(starpu_worker_get_count_by_type(STARPU_CPU_WORKER));

        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2M){
            P2M(inTree, allCellHandles, allParticlesHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2M){
            M2M(inTree, allCellHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2L){
            M2L(inTree, allCellHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2L){
            L2L(inTree, allCellHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2P){
            P2P(inTree, allParticlesHandles);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2P){
            L2P(inTree, allCellHandles, allParticlesHandles);
        }

        starpu_task_wait_for_all();

        starpu_pause();
        vecIndexBuffer.clear();
        TbfStarPUHandleBuilder::CleanCellHandles(allCellHandles);
        TbfStarPUHandleBuilder::CleanParticleHandles(allParticlesHandles);
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        for(const auto& kernel : kernels){
            inFunc(kernel);
        }
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfSmStarpuAlgorithm& inAlgo) {
        inStream << "TbfSmStarpuAlgorithm @ " << &inAlgo << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inAlgo.configuration << "\n";
        inStream << " - Space system: " << "\n";
        inStream << inAlgo.spaceSystem << "\n";
        return inStream;
    }

    static int GetNbThreads(){
        return starpu_worker_get_count_by_type(STARPU_CPU_WORKER);
    }

    static const char* GetName(){
        return "TbfSmStarpuAlgorithm";
    }
};

#endif
