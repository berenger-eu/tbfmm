#ifndef TBFSMSPECXALGORITHMTSM_HPP
#define TBFSMSPECXALGORITHMTSM_HPP

#include "tbfglobal.hpp"

#include "../sequential/tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <Legacy/SpRuntime.hpp>


#include <cassert>
#include <iterator>

template <class RealType_T, class KernelClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfSmSpecxAlgorithmTsm {
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
    void P2M(SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC>& runtime, TreeClass& inTree){
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
                auto& leafGroupObj = *currentLeafGroup;
                const auto& particleGroupObj = *currentParticleGroup;
                runtime.task(SpPriority(priorities.getP2MPriority()), SpRead(*particleGroupObj.getDataPtr()), SpCommutativeWrite(*leafGroupObj.getMultipolePtr()),
                                   [this, &leafGroupObj, &particleGroupObj](const unsigned char&, unsigned char&){
                    kernelWrapper.P2M(kernels[SpUtils::GetThreadId()-1], particleGroupObj, leafGroupObj);
                });
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void M2M(SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC>& runtime, TreeClass& inTree){
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

                auto& upperGroup = *currentUpperGroup;
                const auto& lowerGroup = *currentLowerGroup;
                runtime.task(SpPriority(priorities.getM2MPriority(idxLevel)), SpRead(*lowerGroup.getMultipolePtr()), SpCommutativeWrite(*upperGroup.getMultipolePtr()),
                                   [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
                    kernelWrapper.M2M(idxLevel, kernels[SpUtils::GetThreadId()-1], lowerGroup, upperGroup);
                });

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
    void M2L(SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC>& runtime, TreeClass& inTree){
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
                    assert(&groupTarget == &*currentCellGroup);

                    runtime.task(SpPriority(priorities.getM2LPriority(idxLevel)), SpRead(*groupSrc.getMultipolePtr()), SpCommutativeWrite(*groupTarget.getLocalPtr()),
                                       [this, idxLevel, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, unsigned char&){
                        kernelWrapper.M2LBetweenGroups(idxLevel, kernels[SpUtils::GetThreadId()-1], groupTarget, groupSrc, std::move(indexesVec));
                    });
                });

                ++currentCellGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC>& runtime, TreeClass& inTree){
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

                const auto& upperGroup = *currentUpperGroup;
                auto& lowerGroup = *currentLowerGroup;
                runtime.task(SpPriority(priorities.getL2LPriority(idxLevel)), SpRead(*upperGroup.getLocalPtr()), SpCommutativeWrite(*lowerGroup.getLocalPtr()),
                                   [this, idxLevel, &upperGroup, &lowerGroup](const unsigned char&, unsigned char&){
                    kernelWrapper.L2L(idxLevel, kernels[SpUtils::GetThreadId()-1], upperGroup, lowerGroup);
                });

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
    void L2P(SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC>& runtime, TreeClass& inTree){
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

                const auto& leafGroupObj = *currentLeafGroup;
                auto& particleGroupObj = *currentParticleGroup;
                runtime.task(SpPriority(priorities.getL2PPriority()), SpRead(*leafGroupObj.getLocalPtr()), SpRead(*particleGroupObj.getDataPtr()),
                             SpCommutativeWrite(*particleGroupObj.getRhsPtr()),
                                   [this, &leafGroupObj, &particleGroupObj](const unsigned char&, const unsigned char&, unsigned char&){
                    kernelWrapper.L2P(kernels[SpUtils::GetThreadId()-1], leafGroupObj, particleGroupObj);
                });

                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC>& runtime, TreeClass& inTree){
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

                runtime.task(SpPriority(priorities.getP2PPriority()), SpRead(*groupSrc.getDataPtr()), SpRead(*groupTarget.getDataPtr()),
                             SpCommutativeWrite(*groupTarget.getRhsPtr()),
                                   [this, indexesVec = indexes.toStdVector(), &groupSrc, &groupTarget](const unsigned char&, const unsigned char&, unsigned char&){
                    kernelWrapper.P2PBetweenGroupsTsm(kernels[SpUtils::GetThreadId()-1], groupSrc, groupTarget, std::move(indexesVec));
                });

            });

            ++currentParticleGroupTarget;
        }
    }

    void increaseNumberOfKernels(const int inNbThreads){
        for(long int idxThread = kernels.size() ; idxThread < inNbThreads ; ++idxThread){
            kernels.emplace_back(kernels[0]);
        }
    }

public:
    explicit TbfSmSpecxAlgorithmTsm(const SpacialConfiguration& inConfiguration, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(configuration);
    }

    template <class SourceKernelClass,
              typename = typename std::enable_if<!std::is_same<long int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value
                                                 && !std::is_same<int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value, void>::type>
    TbfSmSpecxAlgorithmTsm(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inStopUpperLevel = TbfDefaultLastLevel)
        : configuration(inConfiguration), spaceSystem(configuration), stopUpperLevel(std::max(0L, inStopUpperLevel)),
          kernelWrapper(configuration),
          priorities(configuration.getTreeHeight()){
        kernels.emplace_back(std::forward<SourceKernelClass>(inKernel));
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::TbfOperations::TbfNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        SpComputeEngine ce(SpWorkerTeamBuilder::TeamOfCpuWorkers());
        SpTaskGraph<SpSpeculativeModel::SP_NO_SPEC> tg;
        tg.computeOn(ce);

        increaseNumberOfKernels(ce.getNbCpuWorkers());

        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2M){
            P2M(tg, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2M){
            M2M(tg, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2L){
            M2L(tg, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2L){
            L2L(tg, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfP2P){
            P2P(tg, inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2P){
            L2P(tg, inTree);
        }

        tg.waitAllTasks();
        ce.stopIfNotAlreadyStopped();
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        for(const auto& kernel : kernels){
            inFunc(kernel);
        }
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfSmSpecxAlgorithmTsm& inAlgo) {
        inStream << "TbfSmSpecxAlgorithmTsm @ " << &inAlgo << "\n";
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
        return "TbfSmSpecxAlgorithmTsm";
    }
};

#endif
