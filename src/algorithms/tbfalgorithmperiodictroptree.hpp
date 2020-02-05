#ifndef TBFALGORITHMPERIODICTOPTREE_HPP
#define TBFALGORITHMPERIODICTOPTREE_HPP

#include "tbfglobal.hpp"

#include "tbfgroupkernelinterface.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <cassert>
#include <iterator>

template <class RealType_T, class KernelClass_T, class MultipoleType_t,
          class LocalType_t, class SpaceIndexType_T = TbfDefaultSpaceIndexTypePeriodic<RealType_T>>
class TbfAlgorithmPeriodicTopTreePeriodicTopTree {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using MultipoleType = MultipoleType_t;
    using LocalType = LocalType_t;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

protected:
    const SpacialConfiguration configuration;
    const SpacialConfiguration extendedConfiguration;
    const SpaceIndexType spaceSystem;

    const long int nbLevelsAbove0;

    KernelClass kernel;

    std::vector<MultipoleType> multipoles;
    std::vector<LocalType> locals;

    template <class TreeClass>
    void M2M(TreeClass& inTree){
        {
            std::vector<std::reference_wrapper<const CellMultipoleType>> children;
            long int positionsOfChildren[spaceSystem.getNbChildrenPerCell()];
            long int nbChildren = 0;

            const long int idxLevel = 0;
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevel+1);

            auto currentLowerGroup = lowerCellGroup.cbegin();
            const auto endLowerGroup = lowerCellGroup.cend();

            assert(currentLowerGroup != endLowerGroup);


            while(currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) == 0
                       || 0 == spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));

                for(long int idxCell = 0 ; idxCell < inLowerGroup.getNbCells() ; ++idxCell){
                    assert(nbChildren < spaceSystem.getNbChildrenPerCell());
                    children.emplace_back(inLowerGroup.getCellMultipole(idxChild));
                    positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(inLowerGroup.getCellSpacialIndex(idxChild));
                    nbChildren += 1;
                }

                ++currentLowerGroup;
            }

            inKernel.M2M(inTree.getCellGroupsAtLevel(idxLevel).front().getCellSymbData(0),
                         idxLevel, TbfUtils::make_const(children), multipoles.front(),
                         positionsOfChildren, nbChildren);
        }
        for(int idxExtraLevel = 1 ; idxExtraLevel < nbLevelsAbove0 ; ++idxExtraLevel){
            std::vector<std::reference_wrapper<const CellMultipoleType>> children;
            long int positionsOfChildren[spaceSystem.getNbChildrenPerCell()];
            long int nbChildren = 0;

            const long int idxLevel = idxExtraLevel;// TODO

            for(long int idxCell = 0 ; idxCell < spaceSystem.getNbChildrenPerCell() ; ++idxCell){
                assert(nbChildren < spaceSystem.getNbChildrenPerCell());
                children.emplace_back(multipoles[TODO]);
                positionsOfChildren[nbChildren] = (idxChild));
                nbChildren += 1;
            }

            inKernel.M2M(inTree.getCellGroupsAtLevel(idxLevel).front().getCellSymbData(0),
                         idxLevel, TbfUtils::make_const(children), multipoles[TODO],
                         positionsOfChildren, nbChildren);
        }
    }

    template <class TreeClass>
    void M2L(TreeClass& inTree){
        const auto& spacialSystem = inTree.getSpacialSystem();

        for(long int idxLevel = nbLevelsAbove0 ; idxLevel <= configuration.getTreeHeight()-1 ; ++idxLevel){
            auto& cellGroups = inTree.getCellGroupsAtLevel(idxLevel);

            auto currentCellGroup = cellGroups.begin();
            const auto endCellGroup = cellGroups.end();

            while(currentCellGroup != endCellGroup){

                auto indexesForGroup = spacialSystem.getInteractionListForBlock(*currentCellGroup, idxLevel);
                TbfAlgorithmPeriodicTopTreeUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), cellGroups, std::distance(cellGroups.begin(),currentCellGroup),
                                               [&](auto& groupTarget, const auto& groupSrc, const auto& indexes){
                    assert(&groupTarget == &*currentCellGroup);
                    kernelWrapper.M2LBetweenGroups(idxLevel, kernel, groupTarget, groupSrc, indexes);
                });

                kernelWrapper.M2LInGroup(idxLevel, kernel, *currentCellGroup, indexesForGroup.first);


                ++currentCellGroup;
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree){
        for(long int idxLevel = nbLevelsAbove0 ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
            const auto& upperCellGroup = inTree.getCellGroupsAtLevel(idxLevel);
            auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevel+1);

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
        if(configuration.getTreeHeight() <= 2){
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
                kernelWrapper.L2P(kernel, *currentLeafGroup, *currentParticleGroup);
                ++currentParticleGroup;
                ++currentLeafGroup;
            }
        }
    }

    template <class TreeClass>
    void P2P(TreeClass& inTree){
        if(configuration.getTreeHeigt() == 1){
            const auto& spacialSystem = inTree.getSpacialSystem();

            auto& particleGroups = inTree.getParticleGroups();

            auto currentParticleGroup = particleGroups.begin();
            const auto endParticleGroup = particleGroups.end();

            while(currentParticleGroup != endParticleGroup){

                auto indexesForGroup = spacialSystem.getNeighborListForBlock(*currentParticleGroup, configuration.getTreeHeight()-1, true);
                TbfAlgorithmPeriodicTopTreeUtils::TbfMapIndexesAndBlocks(std::move(indexesForGroup.second), particleGroups, std::distance(particleGroups.begin(), currentParticleGroup),
                                               [&](auto& groupTarget, auto& groupSrc, const auto& indexes){
                    assert(&groupTarget == &*currentParticleGroup);
                    kernelWrapper.P2PBetweenGroups(kernel, groupTarget, groupSrc, indexes);
                });

                kernelWrapper.P2PInGroup(kernel, *currentParticleGroup, indexesForGroup.first);

                kernelWrapper.P2PInner(kernel, *currentParticleGroup);

                ++currentParticleGroup;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////

    static long int getExtendedTreeHeight(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0) {
        return inConfiguration.getTreeHeight() + inNbLevelsAbove0;
    }

    static long int  getExtendedTreeHeightBoundary(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0) {
        return inConfiguration.getTreeHeight() + inNbLevelsAbove0 + 1;
    }

    static long int GetNbRepetitionsPerDim(const long int inNbLevelsAbove0) {
        if( inNbLevelsAbove0 == -1 ){
            // We compute until the usual level 1
            // we know it is 3 times 3 box (-1;+1)
            return 3;
        }
        return 6 * (1 << inNbLevelsAbove0);
    }

    static auto GetExtendedBoxCenter(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0) {
        const auto originalBoxWidth = inConfiguration.getBoxWidth();
        const auto originalBoxCenter = inConfiguration.getBoxCenter();

        if( nbLevelsAboveRoot == -1 ){
            std::array<RealType, Dim> boxCenter;
            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                boxCenter[idxDim] = originalBoxCenter[idxDim] + originalBoxWidth[idxDim] * 0.5;
            }
            return  boxCenter;
        }
        else{
            const RealType offset = GetNbRepetitionsPerDim(inNbLevelsAbove0)/RealType(2.0);
            std::array<RealType, Dim> boxCenter;
            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                boxCenter[idxDim] = originalBoxCenter[idxDim] - (originalBoxWidth[idxDim] * 0.5) + offset;
            }
            return  boxCenter;
        }
    }

    static auto GetExtendedBoxCenterBoundary(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0) {
        const auto originalBoxWidth = inConfiguration.getBoxWidth();
        const auto originalBoxCenter = inConfiguration.getBoxCenter();

        std::array<RealType, Dim> boxCenter;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxCenter[idxDim] = originalBoxCenter[idxDim] + originalBoxWidth[idxDim] * 0.5;
        }
        return  boxCenter;
    }

    static auto GetExtendedBoxWidth(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0){
        auto boxWidth = inConfiguration.getBoxWidth();
        const RealType coef = (inNbLevelsAbove0 == -1 ? 2 : RealType(4<<(inNbLevelsAbove0)));
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxWidth[idxDim] *= coef;
        }
        return boxWidth;
    }

    static auto GetExtendedBoxWidthBoundary(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0){
        auto boxWidth = inConfiguration.getBoxWidth();
        const RealType coef = (inNbLevelsAbove0 == -1 ? 4 : RealType(8<<(inNbLevelsAbove0)));
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxWidth[idxDim] *= coef;
        }
        return boxWidth;
    }

public:
    SpacialConfiguration GenerateAboveTreeConfiguration(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0){
        const auto boxWidth = GetExtendedBoxWidth(inConfiguration.getBoxWidth(), inNbLevelsAbove0);
        [[maybe_unused]] const auto boxWidthBoundary = GetExtendedBoxWidthBoundary(inConfiguration.getBoxWidth(), inNbLevelsAbove0);

        // TODO does not have to be that long
        const long int treeHeight = getExtendedTreeHeight(inConfiguration, inNbLevelsAbove0);
        [[maybe_unused]] const long int treeHeightBoundary = getExtendedTreeHeightBoundary(inConfiguration, inNbLevelsAbove0);

        const auto boxCenter = GetExtendedBoxCenter(inConfiguration.getBoxWidth(), inNbLevelsAbove0);
        [[maybe_unused]] const auto boxCenterBoundary = GetExtendedBoxCenterBoundary(inConfiguration.getBoxWidth(), inNbLevelsAbove0);

        return SpacialConfiguration(treeHeight, boxWidth, boxCenter);
    }


    explicit TbfAlgorithmPeriodicTopTree(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0)
        : configuration(inConfiguration), extendedConfiguration(GenerateAboveTreeConfiguration(inConfiguration)),
          spaceSystem(configuration), nbLevelsAbove0(std::max(0L, inStopUpperLevel)), kernelWrapper(configuration), kernel(configuration){
    }

    template <class SourceKernelClass>
    TbfAlgorithmPeriodicTopTree(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inNbLevelsAbove0)
        : configuration(inConfiguration), extendedConfiguration(GenerateAboveTreeConfiguration(inConfiguration)),
          spaceSystem(configuration), nbLevelsAbove0(std::max(0L, inStopUpperLevel)), kernelWrapper(configuration), kernel(std::forward<SourceKernelClass>(inKernel)){
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmPeriodicTopTreeUtils::TbfOperations::TbfNearAndFarFields){
        assert(configuration == inTree.getSpacialConfiguration());

        if(inOperationToProceed & TbfAlgorithmPeriodicTopTreeUtils::TbfP2M){
            P2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmPeriodicTopTreeUtils::TbfM2M){
            M2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmPeriodicTopTreeUtils::TbfM2L){
            M2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmPeriodicTopTreeUtils::TbfL2L){
            L2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmPeriodicTopTreeUtils::TbfL2P){
            L2P(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmPeriodicTopTreeUtils::TbfP2P){
            P2P(inTree);
        }
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        inFunc(kernel);
    }

    ////////////////////////////////////////////////////////////////////////

    long int getNbRepetitionsPerDim() const {
        return GetNbRepetitionsPerDim(nbLevelsAbove0)
    }

    long int getNbTotalRepetitions() const {
        const long int nbRepeatInOneDim = getNbRepetitionsPerDim();
        long int totalRepeats = 1;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            totalRepeats *= nbRepeatInOneDim;
        }
        return totalRepeats;
    }


    auto getRepetitionsIntervals() const {
        if( nbLevelsAboveRoot == -1 ){
            // We know it is (-1;1)
            return std::pair<std::array<long int, Dim>,std::array<long int, Dim>>(
                        TbfUtils::make_array<long int, Dim>(-1),
                        TbfUtils::make_array<long int, Dim>(1));
        }
        else{
            const long int halfRepeated = int(getNbRepetitionsPerDim()/2);
            return std::pair<std::array<long int, Dim>,std::array<long int, Dim>>(
                        TbfUtils::make_array<long int, Dim>(-halfRepeated),
                        TbfUtils::make_array<long int, Dim>(halfRepeated-1));
        }
    }


};

#endif
