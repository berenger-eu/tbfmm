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


template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfSmStarpuAlgorithm {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
    using ThisClass = TbfSmStarpuAlgorithm<RealType_T, KernelClass_T, SpaceIndexType_T>;

    using CellHandleContainer = std::vector<std::vector<std::array<starpu_data_handle_t, 3>>>;
    using ParticleHandleContainer = std::vector<std::array<starpu_data_handle_t,2>>;

    void CleanCellHandles(CellHandleContainer& inCellHandles) const{
        for(auto& handlePerLevel : inCellHandles){
            for(auto& handleGroup : handlePerLevel){
                for(auto& handle : handleGroup){
                    starpu_data_unregister(handle);
                }
            }
        }
    }

    template <class TreeClass>
    auto GetCellHandles(TreeClass& inTree) const{
        CellHandleContainer allCellHandles(configuration.getTreeHeight());

        for(long int idxLevel = 0 ; idxLevel < configuration.getTreeHeight() ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){
                starpu_data_handle_t handleData;
                starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                            uintptr_t(currentCellGroup->getDataPtr()),
                                            uint32_t(currentCellGroup->getDataSize()));

                starpu_data_handle_t handleMultipole;
                starpu_variable_data_register(&handleMultipole, STARPU_MAIN_RAM,
                                            uintptr_t(currentCellGroup->getMultipolePtr()),
                                            uint32_t(currentCellGroup->getMultipoleSize()));

                starpu_data_handle_t handleLocal;
                starpu_variable_data_register(&handleLocal, STARPU_MAIN_RAM,
                                            uintptr_t(currentCellGroup->getLocalPtr()),
                                            uint32_t(currentCellGroup->getLocalSize()));

                std::array<starpu_data_handle_t, 3> cellHandles{handleData, handleMultipole, handleLocal};
                allCellHandles[idxLevel].push_back(cellHandles);

                ++currentCellGroup;
            }
        }
        return allCellHandles;
    }


    void CleanParticleHandles(ParticleHandleContainer& inParticleHandles) const{
        for(auto& handleGroup : inParticleHandles){
            for(auto& handle : handleGroup){
                starpu_data_unregister(handle);
            }
        }
    }

    template <class TreeClass>
    auto GetParticleHandles(TreeClass& inTree) const{
        ParticleHandleContainer allParticlesHandles;

        auto& particleGroups = inTree.getParticleGroups();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        while(currentParticleGroup != endParticleGroup){
            starpu_data_handle_t handleData;
            starpu_variable_data_register(&handleData, STARPU_MAIN_RAM,
                                        uintptr_t(currentParticleGroup->getDataPtr()),
                                        uint32_t(currentParticleGroup->getDataSize()));

            starpu_data_handle_t handleRhs;
            starpu_variable_data_register(&handleRhs, STARPU_MAIN_RAM,
                                        uintptr_t(currentParticleGroup->getRhsPtr()),
                                        uint32_t(currentParticleGroup->getRhsSize()));

            std::array<starpu_data_handle_t,2> particlesHandles{handleData, handleRhs};
            allParticlesHandles.push_back(particlesHandles);

            ++currentParticleGroup;
        }
        return allParticlesHandles;
    }

    // TODO release dependencies

    using VecOfIndexes = std::vector<TbfXtoXInteraction<typename SpaceIndexType::IndexType>>;
    std::list<VecOfIndexes> vecIndexBuffer;

    starpu_codelet p2m_cl;
    starpu_codelet m2m_cl;
    starpu_codelet l2l_cl;
    starpu_codelet l2l_cl_nocommute;
    starpu_codelet l2p_cl;

    starpu_codelet m2l_cl_in;
    starpu_codelet m2l_cl_inout;

    starpu_codelet p2p_cl_in;
    starpu_codelet p2p_cl_inout;

    template<class CellContainerClass, class ParticleContainerClass>
    static void P2MCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        CellContainerClass* leafGroupObj;
        ParticleContainerClass* particleGroupObj;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &leafGroupObj, &particleGroupObj);
        [[maybe_unused]] void* particleData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        [[maybe_unused]] void* leafData = (void*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        [[maybe_unused]] size_t leafDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        thisptr->kernelWrapper.P2M(thisptr->kernels[starpu_worker_get_id()], *particleGroupObj, *leafGroupObj);
    }

    template<class ParticleContainerClass>
    static void P2PInOutCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        VecOfIndexes* indexesForGroup_first;
        ParticleContainerClass* groupSrc;
        ParticleContainerClass* groupTarget;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first, &groupSrc, &groupTarget);

        [[maybe_unused]] void* srcData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        [[maybe_unused]] void* tgtData = (void*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        [[maybe_unused]] size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        thisptr->kernelWrapper.P2PBetweenGroups(thisptr->kernels[starpu_worker_get_id()], *groupTarget, *groupSrc, *indexesForGroup_first);
    }

    template<class ParticleContainerClass>
    static void P2PCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        VecOfIndexes* indexesForGroup_first;
        ParticleContainerClass* currentGroup;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first, &currentGroup);

        [[maybe_unused]] void* particleData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        thisptr->kernelWrapper.P2PInGroup(thisptr->kernels[starpu_worker_get_id()], *currentGroup, *indexesForGroup_first);
        thisptr->kernelWrapper.P2PInner(thisptr->kernels[starpu_worker_get_id()], *currentGroup);
    }

    template<class CellContainerClass>
    static void M2MCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        long int idxLevel;
        CellContainerClass* upperGroupObj;
        CellContainerClass* lowerGroupObj;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &upperGroupObj, &lowerGroupObj);

        [[maybe_unused]] void* lowerData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t lowerDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        [[maybe_unused]] void* upperData = (void*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        [[maybe_unused]] size_t upperDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        thisptr->kernelWrapper.M2M(idxLevel, thisptr->kernels[starpu_worker_get_id()], *lowerGroupObj, *upperGroupObj);
    }

    template<class CellContainerClass>
    static void M2LCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        VecOfIndexes* indexesForGroup_first;
        CellContainerClass* groupSrc;
        CellContainerClass* groupTarget;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first, &groupSrc, &groupTarget);

        [[maybe_unused]] void* srcData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        [[maybe_unused]] void* tgtData = (void*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        [[maybe_unused]] size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        thisptr->kernelWrapper.M2LBetweenGroups(idxLevel, thisptr->kernels[starpu_worker_get_id()], *groupTarget, *groupSrc, *indexesForGroup_first);
    }

    template<class CellContainerClass>
    static void M2LInnerCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        VecOfIndexes* indexesForGroup_first;
        CellContainerClass* currentGroup;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first, &currentGroup);

        [[maybe_unused]] void* srcData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        thisptr->kernelWrapper.M2LInGroup(idxLevel, thisptr->kernels[starpu_worker_get_id()], *currentGroup, *indexesForGroup_first);
    }

    template<class CellContainerClass>
    static void L2LCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        long int idxLevel;
        CellContainerClass* upperGroupObj;
        CellContainerClass* lowerGroupObj;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &upperGroupObj, &lowerGroupObj);

        [[maybe_unused]] void* upperData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t upperDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        [[maybe_unused]] void* lowerData = (void*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        [[maybe_unused]] size_t lowerDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        thisptr->kernelWrapper.L2L(idxLevel, thisptr->kernels[starpu_worker_get_id()], *upperGroupObj, *lowerGroupObj);
    }

    template<class CellContainerClass, class ParticleContainerClass>
    static void L2PCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        CellContainerClass* leafGroupObj;
        ParticleContainerClass* particleGroupObj;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &leafGroupObj, &particleGroupObj);

        [[maybe_unused]] void* leafData = (void*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        [[maybe_unused]] size_t leafDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        [[maybe_unused]] void* particleData = (void*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        [[maybe_unused]] size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        thisptr->kernelWrapper.L2P(thisptr->kernels[starpu_worker_get_id()], *leafGroupObj, *particleGroupObj);
    }

    template<class CellContainerClass, class ParticleContainerClass>
    void initCodelet(){
        memset(&p2m_cl, 0, sizeof(p2m_cl));
        p2m_cl.cpu_funcs[0] = &P2MCallback<CellContainerClass, ParticleContainerClass>;
        p2m_cl.where |= STARPU_CPU;
        p2m_cl.nbuffers = 2;
        p2m_cl.modes[0] = STARPU_R;
        p2m_cl.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2m_cl.name = "p2m_cl";

        memset(&m2m_cl, 0, sizeof(m2m_cl));
        m2m_cl.cpu_funcs[0] = &M2MCallback<CellContainerClass>;
        m2m_cl.where |= STARPU_CPU;
        m2m_cl.nbuffers = 2;
        m2m_cl.modes[0] = STARPU_R;
        m2m_cl.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2m_cl.name = "m2m_cl";

        memset(&l2l_cl, 0, sizeof(l2l_cl));
        l2l_cl.cpu_funcs[0] = &L2LCallback<CellContainerClass>;
        l2l_cl.where |= STARPU_CPU;
        l2l_cl.nbuffers = 2;
        l2l_cl.modes[0] = STARPU_R;
        l2l_cl.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        l2l_cl.name = "l2l_cl";

        memset(&l2p_cl, 0, sizeof(l2p_cl));
        l2p_cl.cpu_funcs[0] = &L2PCallback<CellContainerClass, ParticleContainerClass>;
        l2p_cl.where |= STARPU_CPU;
        l2p_cl.nbuffers = 2;
        l2p_cl.modes[0] = STARPU_R;
        l2p_cl.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        l2p_cl.name = "l2p_cl";

        memset(&p2p_cl_in, 0, sizeof(p2p_cl_in));
        p2p_cl_in.cpu_funcs[0] = &P2PCallback<ParticleContainerClass>;
        p2p_cl_in.where |= STARPU_CPU;
        p2p_cl_in.nbuffers = 2;
        p2p_cl_in.modes[0] = STARPU_R;
        p2p_cl_in.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_in.name = "p2p_cl_in";

        memset(&p2p_cl_inout, 0, sizeof(p2p_cl_inout));
        p2p_cl_inout.cpu_funcs[0] = &P2PInOutCallback<ParticleContainerClass>;
        p2p_cl_inout.where |= STARPU_CPU;
        p2p_cl_inout.nbuffers = 4;
        p2p_cl_inout.modes[0] = STARPU_R;
        p2p_cl_inout.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_inout.modes[2] = STARPU_R;
        p2p_cl_inout.modes[3] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        p2p_cl_inout.name = "p2p_cl_inout";

        memset(&m2l_cl_in, 0, sizeof(m2l_cl_in));
        m2l_cl_in.cpu_funcs[0] = M2LCallback<CellContainerClass>;
        m2l_cl_in.where |= STARPU_CPU;
        m2l_cl_in.nbuffers = 2;
        m2l_cl_in.modes[0] = STARPU_R;
        m2l_cl_in.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2l_cl_in.name = "m2l_cl_in";

        memset(&m2l_cl_inout, 0, sizeof(m2l_cl_inout));
        m2l_cl_inout.cpu_funcs[0] = M2LInnerCallback<CellContainerClass>;
        m2l_cl_inout.where |= STARPU_CPU;
        m2l_cl_inout.nbuffers = 2;
        m2l_cl_inout.modes[0] = STARPU_R;
        m2l_cl_inout.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
        m2l_cl_inout.name = "m2l_cl_inout";
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
                auto& leafGroupObj = *currentLeafGroup;
                const auto& particleGroupObj = *currentParticleGroup;
//                runtime.task(SpPriority(priorities.getP2MPriority()), SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*leafGroupObj.getMultipolePtr()),
//                                   [this, &leafGroupObj, &particleGroupObj](const unsigned char&, unsigned char&){
//                    kernelWrapper.P2M(kernels[SpUtils::GetThreadId()-1], particleGroupObj, leafGroupObj);
//                });
                auto* thisptr = this;
                CellContainerClass* leafGroupObjPtr = &leafGroupObj;
                const ParticleContainerClass* particleGroupObjPtr = &particleGroupObj;
                starpu_insert_task(&p2m_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &leafGroupObjPtr, sizeof(void*),
                                   STARPU_VALUE, &particleGroupObjPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getP2MPriority(),
                                   STARPU_R, particleHandles[idxGroup][0],
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

                auto& upperGroup = *currentUpperGroup;
                const auto& lowerGroup = *currentLowerGroup;
//                runtime.task(SpPriority(priorities.getM2MPriority(idxLevel)), SpRead(*lowerGroup.getMultipolePtr()), SpCommutativeWrite(*upperGroup.getMultipolePtr()),
//                                   [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
//                    kernelWrapper.M2M(idxLevel, kernels[SpUtils::GetThreadId()-1], lowerGroup, upperGroup);
//                });
                auto* thisptr = this;
                CellContainerClass* upperGroupPtr = &upperGroup;
                const CellContainerClass* lowerGroupPtr = &lowerGroup;
                starpu_insert_task(&m2m_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(long int),
                                   STARPU_VALUE, &upperGroupPtr, sizeof(void*),
                                   STARPU_VALUE, &lowerGroupPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getM2MPriority(idxLevel),
                                   STARPU_R, cellHandles[idxLevel+1][idxLowerGroup][1],
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
                      auto& groupTarget = cellGroups[groupTargetIdx];
                      const auto& groupSrc = cellGroups[groupSrcIdx];

                    assert(&groupTarget == &*currentCellGroup);

//                    runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*groupSrc.getMultipolePtr()), SpCommutativeWrite(*groupTarget.getLocalPtr()),
//                                       [this, idxLevel, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&){
//                        kernelWrapper.M2LBetweenGroups(idxLevel, kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
//                    });
                    auto* thisptr = this;
                    vecIndexBuffer.push_back(indexes.toStdVector());
                    VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
                    const CellContainerClass* groupSrcPtr = &groupSrc;
                    CellContainerClass* groupTargetPtr = &groupTarget;
                    starpu_insert_task(&m2l_cl_in,
                                       STARPU_VALUE, &thisptr, sizeof(void*),
                                       STARPU_VALUE, &idxLevel, sizeof(int),
                                       STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                                       STARPU_VALUE, &groupSrcPtr, sizeof(void*),
                                       STARPU_VALUE, &groupTargetPtr, sizeof(void*),
                                       STARPU_PRIORITY, priorities.getM2LPriority(idxLevel),
                                       STARPU_R, cellHandles[idxLevel][groupSrcIdx][1],
                                       starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), cellHandles[idxLevel][groupTargetIdx][2],
                                       STARPU_NAME, "M2L",
                                       0);
                });

                auto& currentGroup = *currentCellGroup;
//                runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*currentGroup.getMultipolePtr()), SpCommutativeWrite(*currentGroup.getLocalPtr()),
//                                   [this, idxLevel, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const unsigned char&, unsigned char&){
//                    kernelWrapper.M2LInGroup(idxLevel, kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);
//                });
                auto* thisptr = this;
                vecIndexBuffer.push_back(std::move(indexesForGroup.first));
                VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
                CellContainerClass* groupTargetPtr = &currentGroup;
                starpu_insert_task(&m2l_cl_inout,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(int),
                                   STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                                   STARPU_VALUE, &groupTargetPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getM2LPriority(idxLevel),
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

                const auto& upperGroup = *currentUpperGroup;
                auto& lowerGroup = *currentLowerGroup;
//                runtime.task(SpPriority(priorities.getL2LPriority(idxLevel)), SpRead(*upperGroup.getLocalPtr()), SpCommutativeWrite(*lowerGroup.getLocalPtr()),
//                                   [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
//                    kernelWrapper.L2L(idxLevel, kernels[SpUtils::GetThreadId()-1], upperGroup, lowerGroup);
//                });
                auto* thisptr = this;
                const CellContainerClass* upperGroupPtr = &upperGroup;
                CellContainerClass* lowerGroupPtr = &lowerGroup;
                starpu_insert_task(&l2l_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &idxLevel, sizeof(long int),
                                   STARPU_VALUE, &upperGroupPtr, sizeof(void*),
                                   STARPU_VALUE, &lowerGroupPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getL2LPriority(idxLevel),
                                   STARPU_R, cellHandles[idxLevel][idxUpperGroup][2],
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

                const auto& leafGroupObj = *currentLeafGroup;
                auto& particleGroupObj = *currentParticleGroup;
//                runtime.task(SpPriority(priorities.getL2PPriority()), SpRead(*leafGroupObj.getLocalPtr()),
//                             SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*particleGroupObj.getRhsPtr()),
//                                   [this, &leafGroupObj, &particleGroupObj](const unsigned char&, const unsigned char&, unsigned char&){
//                    kernelWrapper.L2P(kernels[SpUtils::GetThreadId()-1], leafGroupObj, particleGroupObj);
//                });
                auto* thisptr = this;
                const CellContainerClass* leafGroupObjPtr = &leafGroupObj;
                ParticleContainerClass* particleGroupObjPtr = &particleGroupObj;
                starpu_insert_task(&l2p_cl,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &leafGroupObjPtr, sizeof(void*),
                                   STARPU_VALUE, &particleGroupObjPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getL2PPriority(),
                                   STARPU_R, cellHandles[configuration.getTreeHeight()-1][idxGroup][2],
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
                auto& groupTarget = particleGroups[groupTargetIdx];
                auto& groupSrc = particleGroups[groupSrcIdx];
                assert(&groupTarget == &*currentParticleGroup);

//                runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*groupSrc.getDataPtr()), SpCommutativeWrite(*groupSrc.getRhsPtr()),
//                             SpRead(*groupTarget.getDataPtr()), SpCommutativeWrite(*groupTarget.getRhsPtr()),
//                                   [this, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&, const unsigned char&, unsigned char&){
//                    kernelWrapper.P2PBetweenGroups(kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
//                });
                auto* thisptr = this;
                vecIndexBuffer.push_back(indexes.toStdVector());
                VecOfIndexes* vecIndexesPtr = &vecIndexBuffer.back();
                ParticleContainerClass* groupSrcPtr = &groupSrc;
                ParticleContainerClass* groupTargetPtr = &groupTarget;
                starpu_insert_task(&p2p_cl_inout,
                                   STARPU_VALUE, &thisptr, sizeof(void*),
                                   STARPU_VALUE, &vecIndexesPtr, sizeof(void*),
                                   STARPU_VALUE, &groupSrcPtr, sizeof(void*),
                                   STARPU_VALUE, &groupTargetPtr, sizeof(void*),
                                   STARPU_PRIORITY, priorities.getP2PPriority(),
                                   STARPU_R, particleHandles[groupSrcIdx][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleHandles[groupSrcIdx][1],
                                   STARPU_R, particleHandles[groupTargetIdx][0],
                                   starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE), particleHandles[groupTargetIdx][1],
                                   STARPU_NAME, "P2P-INOUT",
                                   0);
            });

            auto& currentGroup = *currentParticleGroup;
//            runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*currentGroup.getDataPtr()),SpCommutativeWrite(*currentGroup.getRhsPtr()),
//                               [this, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const unsigned char&, unsigned char&){
//                kernelWrapper.P2PInGroup(kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);
//                kernelWrapper.P2PInner(kernels[SpUtils::GetThreadId()-1], currentGroup);
//            });
            auto* thisptr = this;
            vecIndexBuffer.push_back(std::move(indexesForGroup.first));
            VecOfIndexes* indexesForGroup_firstPtr = &vecIndexBuffer.back();
            ParticleContainerClass* currentGroupPtr = &currentGroup;
            starpu_insert_task(&p2p_cl_in,
                               STARPU_VALUE, &thisptr, sizeof(void*),
                               STARPU_VALUE, &indexesForGroup_firstPtr, sizeof(void*),
                               STARPU_VALUE, &currentGroupPtr, sizeof(void*),
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

        CellHandleContainer allCellHandles = GetCellHandles(inTree);
        ParticleHandleContainer allParticlesHandles = GetParticleHandles(inTree);

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
        CleanCellHandles(allCellHandles);
        CleanParticleHandles(allParticlesHandles);
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
