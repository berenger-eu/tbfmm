#ifndef TBFSMSPECXALGORITHM_HPP
#define TBFSMSPECXALGORITHM_HPP

#include "tbfglobal.hpp"

#include "../sequential/tbfgroupkernelinterface.hpp"
#include "../sequential/tbfgroupkernelinterfacecuda.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <Legacy/SpRuntime.hpp>


#include <cassert>
#include <iterator>

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfSmSpecxAlgorithm {
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
    TbfGroupKernelInterfaceCuda<SpaceIndexType> kernelWrapperCuda;
    std::vector<KernelClass> kernels;

    TbfAlgorithmUtils::TbfOperationsPriorities priorities;

    /////////////////////////////////////////////////////////////////

    template <const bool>
    class BoolSelecter;

    template <>
    class BoolSelecter<true> : public std::true_type {};

    template <>
    class BoolSelecter<false> : public std::false_type {};

    // See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4502.pdf.
    template <typename...>
    using void_t = void;

    #define CUDA_OP_DETECT(OP_NAME)\
        template <typename, template <typename> class, typename = void_t<>>\
            struct detect_##OP_NAME : std::false_type {};\
        \
            template <typename T, template <typename> class Op>\
            struct detect_##OP_NAME<T, Op, void_t<Op<T>>> : BoolSelecter<T::OP_NAME> {};\
        \
            template <typename T>\
            using OP_NAME##_test = decltype(T::OP_NAME);\
        \
            template <typename T>\
            using class_has_##OP_NAME = detect_##OP_NAME<T, OP_NAME##_test>;

    CUDA_OP_DETECT(CudaP2P)
    CUDA_OP_DETECT(CudaP2M)
    CUDA_OP_DETECT(CudaM2M)
    CUDA_OP_DETECT(CudaM2L)
    CUDA_OP_DETECT(CudaL2L)
    CUDA_OP_DETECT(CudaL2P)

    CUDA_OP_DETECT(CpuP2P)
    CUDA_OP_DETECT(CpuP2M)
    CUDA_OP_DETECT(CpuM2M)
    CUDA_OP_DETECT(CpuM2L)
    CUDA_OP_DETECT(CpuL2L)
    CUDA_OP_DETECT(CpuL2P)

    constexpr static bool CudaP2P = class_has_CudaP2P<KernelClass>::value;
    constexpr static bool CudaP2M = class_has_CudaP2M<KernelClass>::value;
    constexpr static bool CudaM2M = class_has_CudaM2M<KernelClass>::value;
    constexpr static bool CudaM2L = class_has_CudaM2L<KernelClass>::value;
    constexpr static bool CudaL2L = class_has_CudaL2L<KernelClass>::value;
    constexpr static bool CudaL2P = class_has_CudaL2P<KernelClass>::value;

    constexpr static bool CpuP2P = class_has_CpuP2P<KernelClass>::value;
    constexpr static bool CpuP2M = class_has_CpuP2M<KernelClass>::value;
    constexpr static bool CpuM2M = class_has_CpuM2M<KernelClass>::value;
    constexpr static bool CpuM2L = class_has_CpuM2L<KernelClass>::value;
    constexpr static bool CpuL2L = class_has_CpuL2L<KernelClass>::value;
    constexpr static bool CpuL2P = class_has_CpuL2P<KernelClass>::value;

    /////////////////////////////////////////////////////////////////

    template <class TreeClass>
    void P2M(SpRuntime<>& runtime, TreeClass& inTree){
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
                auto& leafGroupObj = *currentLeafGroup;
                const auto& particleGroupObj = *currentParticleGroup;
                if constexpr(CudaP2M && CpuP2M){
                    runtime.task(SpPriority(priorities.getP2MPriority()), SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*leafGroupObj.getMultipolePtr()),
                                 [this, &leafGroupObj, &particleGroupObj](const unsigned char&, unsigned char&){
                                     kernelWrapper.P2M(kernels[SpUtils::GetThreadId()-1], particleGroupObj, leafGroupObj);
                                 },
                        SpCuda([this, &leafGroupObj, &particleGroupObj](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                            //kernelWrapper.P2M(kernels[SpUtils::GetThreadId()-1], particleGroupObj, leafGroupObj);
                        }));
                }
                else if(CpuP2M){
                    runtime.task(SpPriority(priorities.getP2MPriority()), SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*leafGroupObj.getMultipolePtr()),
                                 [this, &leafGroupObj, &particleGroupObj](const unsigned char&, unsigned char&){
                        kernelWrapper.P2M(kernels[SpUtils::GetThreadId()-1], particleGroupObj, leafGroupObj);
                    });
                }
                else if(CudaP2M){
                    runtime.task(SpPriority(priorities.getP2MPriority()), SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*leafGroupObj.getMultipolePtr()),
                                 SpCuda([this, &leafGroupObj, &particleGroupObj](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                        //kernelWrapper.P2M(kernels[SpUtils::GetThreadId()-1], particleGroupObj, leafGroupObj);
                    }));
                }
                else{
                    assert(0);
                }
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(SpRuntime<>& runtime, TreeClass& inTree){
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

                auto& upperGroup = *currentUpperGroup;
                const auto& lowerGroup = *currentLowerGroup;

                if constexpr(CudaM2M && CpuM2M){
                    runtime.task(SpPriority(priorities.getM2MPriority(idxLevel)), SpRead(*lowerGroup.getMultipolePtr()), SpCommutativeWrite(*upperGroup.getMultipolePtr()),
                                       [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
                        kernelWrapper.M2M(idxLevel, kernels[SpUtils::GetThreadId()-1], lowerGroup, upperGroup);
                    },
                        SpCuda([this, idxLevel, &upperGroup, &lowerGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                            //kernelWrapper.M2M(idxLevel, kernels[SpUtils::GetThreadId()-1], lowerGroup, upperGroup);
                        }));
                }
                else if(CpuM2M){
                    runtime.task(SpPriority(priorities.getM2MPriority(idxLevel)), SpRead(*lowerGroup.getMultipolePtr()), SpCommutativeWrite(*upperGroup.getMultipolePtr()),
                                 [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
                                     kernelWrapper.M2M(idxLevel, kernels[SpUtils::GetThreadId()-1], lowerGroup, upperGroup);
                                 });
                }
                else if(CudaM2M){
                    runtime.task(SpPriority(priorities.getM2MPriority(idxLevel)), SpRead(*lowerGroup.getMultipolePtr()), SpCommutativeWrite(*upperGroup.getMultipolePtr()),
                                 SpCuda([this, idxLevel, &upperGroup, &lowerGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                     //kernelWrapper.M2M(idxLevel, kernels[SpUtils::GetThreadId()-1], lowerGroup, upperGroup);
                                 }));
                }
                else{
                    assert(0);
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
    void M2L(SpRuntime<>& runtime, TreeClass& inTree){
        const auto& spacialSystem = inTree.getSpacialSystem();

        for(long int idxLevel = stopUpperLevel ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){
                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel);
                TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), cellGroups, std::distance(cellGroups.begin(),currentCellGroup),
                                               [&](auto& groupTarget, const auto& groupSrc, const auto& indexes){
                    assert(&groupTarget == &*currentCellGroup);

                    if constexpr(CudaM2L && CpuM2L){
                        runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*groupSrc.getMultipolePtr()), SpCommutativeWrite(*groupTarget.getLocalPtr()),
                                           [this, idxLevel, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&){
                            kernelWrapper.M2LBetweenGroups(idxLevel, kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                        },
                            SpCuda([this, idxLevel, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                //kernelWrapper.M2LBetweenGroups(idxLevel, kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                            }));
                    }
                    else if constexpr(CpuM2L){
                        runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*groupSrc.getMultipolePtr()), SpCommutativeWrite(*groupTarget.getLocalPtr()),
                                     [this, idxLevel, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&){
                             kernelWrapper.M2LBetweenGroups(idxLevel, kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                         });
                    }
                    else  if constexpr(CudaM2L){
                        runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*groupSrc.getMultipolePtr()), SpCommutativeWrite(*groupTarget.getLocalPtr()),
                                     SpCuda([this, idxLevel, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                         //kernelWrapper.M2LBetweenGroups(idxLevel, kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                                     }));
                    }
                    else{
                        assert(0);
                    }
                });

                auto& currentGroup = *currentCellGroup;

                if constexpr(CudaM2L && CpuM2L){
                    runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*currentGroup.getMultipolePtr()), SpCommutativeWrite(*currentGroup.getLocalPtr()),
                                       [this, idxLevel, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const unsigned char&, unsigned char&){
                        kernelWrapper.M2LInGroup(idxLevel, kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);
                    },
                        SpCuda([this, idxLevel, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                            //kernelWrapper.M2LInGroup(idxLevel, kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);
                        }));
                }
                else if constexpr(CpuM2L){
                    runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*currentGroup.getMultipolePtr()), SpCommutativeWrite(*currentGroup.getLocalPtr()),
                                 [this, idxLevel, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const unsigned char&, unsigned char&){
                                     kernelWrapper.M2LInGroup(idxLevel, kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);
                                 });
                }
                else if constexpr(CudaM2L){
                    runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*currentGroup.getMultipolePtr()), SpCommutativeWrite(*currentGroup.getLocalPtr()),
                                 SpCuda([this, idxLevel, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                     //kernelWrapper.M2LInGroup(idxLevel, kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);
                                 }));
                }
                else{
                    assert(0);
                }


                ++currentCellGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(SpRuntime<>& runtime, TreeClass& inTree){
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

                const auto& upperGroup = *currentUpperGroup;
                auto& lowerGroup = *currentLowerGroup;

                if constexpr(CudaL2L && CpuL2L){
                    runtime.task(SpPriority(priorities.getL2LPriority(idxLevel)), SpRead(*upperGroup.getLocalPtr()), SpCommutativeWrite(*lowerGroup.getLocalPtr()),
                                       [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
                        kernelWrapper.L2L(idxLevel, kernels[SpUtils::GetThreadId()-1], upperGroup, lowerGroup);
                    },
                        SpCuda([this, idxLevel, &upperGroup, &lowerGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                            //kernelWrapper.L2L(idxLevel, kernels[SpUtils::GetThreadId()-1], upperGroup, lowerGroup);
                        }));
                }
                else if constexpr(CpuL2L){
                    runtime.task(SpPriority(priorities.getL2LPriority(idxLevel)), SpRead(*upperGroup.getLocalPtr()), SpCommutativeWrite(*lowerGroup.getLocalPtr()),
                                 [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
                                     kernelWrapper.L2L(idxLevel, kernels[SpUtils::GetThreadId()-1], upperGroup, lowerGroup);
                                 });
                }
                else if constexpr(CudaL2L){
                    runtime.task(SpPriority(priorities.getL2LPriority(idxLevel)), SpRead(*upperGroup.getLocalPtr()), SpCommutativeWrite(*lowerGroup.getLocalPtr()),
                                 SpCuda([this, idxLevel, &upperGroup, &lowerGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                     //kernelWrapper.L2L(idxLevel, kernels[SpUtils::GetThreadId()-1], upperGroup, lowerGroup);
                                 }));
                }
                else{
                    assert(0);
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
    void L2P(SpRuntime<>& runtime, TreeClass& inTree){
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

                const auto& leafGroupObj = *currentLeafGroup;
                auto& particleGroupObj = *currentParticleGroup;

                if constexpr(CudaL2P && CpuL2P){
                    runtime.task(SpPriority(priorities.getL2PPriority()), SpRead(*leafGroupObj.getLocalPtr()),
                                 SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*particleGroupObj.getRhsPtr()),
                                       [this, &leafGroupObj, &particleGroupObj](const unsigned char&, const unsigned char&, unsigned char&){
                        kernelWrapper.L2P(kernels[SpUtils::GetThreadId()-1], leafGroupObj, particleGroupObj);
                    },
                        SpCuda([this, &leafGroupObj, &particleGroupObj](const SpDeviceDataView<const unsigned char>, const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                            //kernelWrapper.L2P(kernels[SpUtils::GetThreadId()-1], leafGroupObj, particleGroupObj);
                        }));
                }
                else if constexpr(CpuL2P){
                    runtime.task(SpPriority(priorities.getL2PPriority()), SpRead(*leafGroupObj.getLocalPtr()),
                                 SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*particleGroupObj.getRhsPtr()),
                                 [this, &leafGroupObj, &particleGroupObj](const unsigned char&, const unsigned char&, unsigned char&){
                                     kernelWrapper.L2P(kernels[SpUtils::GetThreadId()-1], leafGroupObj, particleGroupObj);
                                 });
                }
                else if constexpr(CudaL2P){
                    runtime.task(SpPriority(priorities.getL2PPriority()), SpRead(*leafGroupObj.getLocalPtr()),
                                 SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*particleGroupObj.getRhsPtr()),
                                 SpCuda([this, &leafGroupObj, &particleGroupObj](const SpDeviceDataView<const unsigned char>, const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                     //kernelWrapper.L2P(kernels[SpUtils::GetThreadId()-1], leafGroupObj, particleGroupObj);
                                 }));
                }
                else{
                    assert(0);
                }

                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(SpRuntime<>& runtime, TreeClass& inTree){
        const auto& spacialSystem = inTree.getSpacialSystem();

        auto& particleGroups = inTree.getParticleGroups();

        auto currentParticleGroup = particleGroups.begin();
        const auto endParticleGroup = particleGroups.end();

        while(currentParticleGroup != endParticleGroup){

            auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroup, configuration.getTreeHeight()-1, true);
            TbfAlgorithmUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), particleGroups, std::distance(particleGroups.begin(), currentParticleGroup),
                                           [&](auto& groupTarget, auto& groupSrc, const auto& indexes){
                assert(&groupTarget == &*currentParticleGroup);

                if constexpr(CudaP2P && CpuP2P){
                    runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*groupSrc.getDataPtr()), SpCommutativeWrite(*groupSrc.getRhsPtr()),
                                 SpRead(*groupTarget.getDataPtr()), SpCommutativeWrite(*groupTarget.getRhsPtr()),
                                       [this, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&, const unsigned char&, unsigned char&){
                        kernelWrapper.P2PBetweenGroups(kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                    },
                        SpCuda([this, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>, const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                            //kernelWrapper.P2PBetweenGroups(kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                        }));
                }
                else if constexpr(CpuP2P){
                    runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*groupSrc.getDataPtr()), SpCommutativeWrite(*groupSrc.getRhsPtr()),
                                 SpRead(*groupTarget.getDataPtr()), SpCommutativeWrite(*groupTarget.getRhsPtr()),
                                 [this, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&, const unsigned char&, unsigned char&){
                                     kernelWrapper.P2PBetweenGroups(kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                                 });
                }
                else if constexpr(CudaP2P){
                    runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*groupSrc.getDataPtr()), SpCommutativeWrite(*groupSrc.getRhsPtr()),
                                 SpRead(*groupTarget.getDataPtr()), SpCommutativeWrite(*groupTarget.getRhsPtr()),
                                 SpCuda([this, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>, const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                     //kernelWrapper.P2PBetweenGroups(kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                                 }));
                }
                else{
                    assert(0);
                }

            });

            auto& currentGroup = *currentParticleGroup;

            if constexpr(CudaP2P && CpuP2P){
                runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*currentGroup.getDataPtr()),SpCommutativeWrite(*currentGroup.getRhsPtr()),
                                   [this, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const unsigned char&, unsigned char&){
                    kernelWrapper.P2PInGroup(kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);

                    kernelWrapper.P2PInner(kernels[SpUtils::GetThreadId()-1], currentGroup);
                },
                    SpCuda([this, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                        //kernelWrapper.P2PInGroup(kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);

                        //kernelWrapper.P2PInner(kernels[SpUtils::GetThreadId()-1], currentGroup);
                    }));
            }
            else if constexpr(CpuP2P){
                runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*currentGroup.getDataPtr()),SpCommutativeWrite(*currentGroup.getRhsPtr()),
                             [this, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const unsigned char&, unsigned char&){
                                 kernelWrapper.P2PInGroup(kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);

                                 kernelWrapper.P2PInner(kernels[SpUtils::GetThreadId()-1], currentGroup);
                             });
            }
            else if constexpr(CudaP2P){
                runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*currentGroup.getDataPtr()),SpCommutativeWrite(*currentGroup.getRhsPtr()),
                             SpCuda([this, indexesForGroup_first = std::move(indexesForGroup.first), &currentGroup](const SpDeviceDataView<const unsigned char>, SpDeviceDataView<unsigned char>){
                                 //kernelWrapper.P2PInGroup(kernels[SpUtils::GetThreadId()-1], currentGroup, indexesForGroup_first);

                                 //kernelWrapper.P2PInner(kernels[SpUtils::GetThreadId()-1], currentGroup);
                             }));
            }
            else{
                assert(0);
            }

            ++currentParticleGroup;
        }
    }

    void increaseNumberOfKernels(const int inNbThreads){
        for(long int idxThread = kernels.size() ; idxThread < inNbThreads ; ++idxThread){
            kernels.emplace_back(kernels[0]);
        }
    }

public:
    explicit TbfSmSpecxAlgorithm(const SpacialConfiguration& inConfiguration, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(configuration);
    }

    template <class SourceKernelClass,
              typename = typename std::enable_if<!std::is_same<long int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value
                                                 && !std::is_same<int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value, void>::type>
    TbfSmSpecxAlgorithm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(std::forward<SourceKernelClass>(inKernel));
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::TbfOperations::TbfNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        SpRuntime runtime;

        increaseNumberOfKernels(runtime.getNbThreads());

        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2M){
            P2M(runtime, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2M){
            M2M(runtime, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2L){
            M2L(runtime, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2L){
            L2L(runtime, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2P){
            P2P(runtime, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2P){
            L2P(runtime, inTree);
        }

        runtime.waitAllTasks();
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        for(const auto& kernel : kernels){
            inFunc(kernel);
        }
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfSmSpecxAlgorithm& inAlgo) {
        inStream << "TbfSmSpecxAlgorithm @ " << &inAlgo << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inAlgo.configuration << "\n";
        inStream << " - Space system: " << "\n";
        inStream << inAlgo.spaceSystem << "\n";
        return inStream;
    }

    static int GetNbThreads(){
        return SpUtils::DefaultNumThreads();
    }

    static const char* GetName(){
        return "TbfSmSpecxAlgorithm";
    }
};

#endif
