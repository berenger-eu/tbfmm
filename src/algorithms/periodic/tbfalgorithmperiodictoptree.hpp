#ifndef TBFALGORITHMPERIODICTOPTREE_HPP
#define TBFALGORITHMPERIODICTOPTREE_HPP

#include "tbfglobal.hpp"

#include "spacial/tbfspacialconfiguration.hpp"
#include "algorithms/tbfalgorithmutils.hpp"

#include <cassert>
#include <iterator>

template <class RealType_T, class KernelClass_T, class CellMultipoleType_t,
          class CellLocalType_t, class SpaceIndexType_T = TbfDefaultSpaceIndexTypePeriodic<RealType_T>>
class TbfAlgorithmPeriodicTopTree {
public:
    using RealType = RealType_T;
    using KernelClass = KernelClass_T;
    using CellMultipoleType = CellMultipoleType_t;
    using CellLocalType = CellLocalType_t;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;
    using IndexType = typename SpaceIndexType::IndexType;

    static constexpr long int Dim = SpaceIndexType::Dim;

protected:
    const SpacialConfiguration originalConfiguration;
    const SpacialConfiguration configuration;
    const SpaceIndexType originalSpaceSystem;
    const SpaceIndexType spaceSystem;

    const long int nbLevelsAbove0;

    KernelClass kernel;

    std::vector<CellMultipoleType> multipoles;
    std::vector<CellLocalType> locals;

    template <class TreeClass>
    void M2M(TreeClass& inTree){
        {
            assert(inTree.getHeight() > 1);
            std::vector<std::reference_wrapper<const CellMultipoleType>> children;
            long int* positionsOfChildren = new long int [spaceSystem.getNbChildrenPerCell()];
            long int nbChildren = 0;

            const long int idxLevelBase = 0;
            const auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevelBase+1);

            auto currentLowerGroup = lowerCellGroup.cbegin();
            const auto endLowerGroup = lowerCellGroup.cend();

            assert(currentLowerGroup != endLowerGroup);


            while(currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) == 0
                       || 0 == spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));

                for(long int idxCell = 0 ; idxCell < currentLowerGroup->getNbCells() ; ++idxCell){
                    assert(nbChildren < spaceSystem.getNbChildrenPerCell());
                    children.emplace_back(currentLowerGroup->getCellMultipole(idxCell));
                    positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(currentLowerGroup->getCellSpacialIndex(idxCell));
                    nbChildren += 1;
                }

                ++currentLowerGroup;
            }

            assert(std::size(inTree.getCellGroupsAtLevel(idxLevelBase)));
            assert(std::size(inTree.getCellGroupsAtLevel(0)));

            kernel.M2M(inTree.getCellGroupsAtLevel(0).front().getCellSymbData(0),
                         configuration.getTreeHeight()-2, TbfUtils::make_const(children), multipoles[configuration.getTreeHeight()-2],
                         positionsOfChildren, nbChildren);
        }

        for(long int idxLevel = configuration.getTreeHeight()-3 ; idxLevel >= 3 ; --idxLevel){
            std::vector<std::reference_wrapper<const CellMultipoleType>> children;
            long int* positionsOfChildren = new long int [spaceSystem.getNbChildrenPerCell()];
            long int nbChildren = 0;

            for(long int idxCell = 0 ; idxCell < spaceSystem.getNbChildrenPerCell() ; ++idxCell){
                assert(nbChildren < spaceSystem.getNbChildrenPerCell());
                children.emplace_back(multipoles[idxLevel+1]);
                positionsOfChildren[nbChildren] = (idxCell);
                nbChildren += 1;
            }

            assert(std::size(inTree.getCellGroupsAtLevel(0)));
            kernel.M2M(inTree.getCellGroupsAtLevel(0).front().getCellSymbData(0),
                         idxLevel, TbfUtils::make_const(children), multipoles[idxLevel],
                         positionsOfChildren, nbChildren);
        }
    }

    template <class TreeClass>
    void M2L(TreeClass& inTree){
        if(nbLevelsAbove0 == 0){
            const long int idxLevel = configuration.getTreeHeight()-2;
            assert(idxLevel == 3);
            std::vector<std::reference_wrapper<const CellMultipoleType>> neighbors;
            static_assert (Dim != 3 || 316 == (TbfUtils::lipow(7, Dim) - TbfUtils::lipow(3, Dim)), "Simple check");
            long int positionsOfNeighbors[TbfUtils::lipow(7, Dim) - TbfUtils::lipow(3, Dim)];
            long int nbNeighbors = 0;

            std::array<long int, Dim> minLimits = TbfUtils::make_array<long int, Dim>(-3);
            std::array<long int, Dim> maxLimits = TbfUtils::make_array<long int, Dim>(3);
            std::array<long int, Dim> currentCellTest = minLimits;

            while(true){
                {
                    long int currentIdx = Dim-1;

                    while(currentIdx >= 0 && currentCellTest[currentIdx] > maxLimits[currentIdx]){
                        currentCellTest[currentIdx] = minLimits[currentIdx];
                        currentIdx -= 1;
                        if(currentIdx >= 0){
                            currentCellTest[currentIdx] += 1;
                        }
                    }
                    if(currentIdx < 0){
                        break;
                    }
                }

                bool isTooClose = true;
                for(int idxDim = 0 ; isTooClose && idxDim < Dim ; ++idxDim){
                    if(std::abs(currentCellTest[idxDim]) > 1){
                        isTooClose = false;
                    }
                }
                if(isTooClose == false){
                    auto childPos = currentCellTest;
                    const IndexType childIndex = spaceSystem.getInteractionIndexFromRelativePos(childPos);

                    assert(nbNeighbors < TbfUtils::lipow(7, Dim) - TbfUtils::lipow(3, Dim));
                    neighbors.emplace_back(multipoles[idxLevel]);
                    positionsOfNeighbors[nbNeighbors] = (childIndex);
                    nbNeighbors += 1;
                }

                currentCellTest[Dim-1] += 1;
            }


            assert(std::size(inTree.getCellGroupsAtLevel(0)));

            kernel.M2L(inTree.getCellGroupsAtLevel(0).front().getCellSymbData(0),
                         idxLevel, TbfUtils::make_const(neighbors), positionsOfNeighbors, nbNeighbors, locals[idxLevel]);
        }
        else{
            for(long int idxLevel = 3 ; idxLevel <= configuration.getTreeHeight()-2 ; ++idxLevel){
                std::vector<std::reference_wrapper<const CellMultipoleType>> neighbors;
                long int* positionsOfNeighbors = new long int [spaceSystem.getNbInteractionsPerCell()];
                long int nbNeighbors = 0;

                std::array<long int, Dim> minLimits;
                std::array<long int, Dim> maxLimits;
                // First -3/2
                if(idxLevel == 3){
                    minLimits.fill(-3);
                    maxLimits.fill(2);
                }
                // Then -2/3
                else{
                    minLimits.fill(-2);
                    maxLimits.fill(3);
                }
                std::array<long int, Dim> currentCellTest = minLimits;

                while(true){
                    {
                        long int currentIdx = Dim-1;

                        while(currentIdx >= 0 && currentCellTest[currentIdx] > maxLimits[currentIdx]){
                            currentCellTest[currentIdx] = minLimits[currentIdx];
                            currentIdx -= 1;
                            if(currentIdx >= 0){
                                currentCellTest[currentIdx] += 1;
                            }
                        }
                        if(currentIdx < 0){
                            break;
                        }
                    }

                    bool isTooClose = true;
                    for(int idxDim = 0 ; isTooClose && idxDim < Dim ; ++idxDim){
                        if(std::abs(currentCellTest[idxDim]) > 1){
                            isTooClose = false;
                        }
                    }
                    if(isTooClose == false){
                        auto childPos = currentCellTest;
                        const IndexType childIndex = spaceSystem.getInteractionIndexFromRelativePos(childPos);

                        assert(nbNeighbors < spaceSystem.getNbInteractionsPerCell());
                        neighbors.emplace_back(multipoles[idxLevel]);
                        positionsOfNeighbors[nbNeighbors] = (childIndex);
                        nbNeighbors += 1;
                    }

                    currentCellTest[Dim-1] += 1;
                }

                assert(std::size(inTree.getCellGroupsAtLevel(0)));
                kernel.M2L(inTree.getCellGroupsAtLevel(0).front().getCellSymbData(0),
                             idxLevel, TbfUtils::make_const(neighbors), positionsOfNeighbors, nbNeighbors, locals[idxLevel]);
            }
        }
    }

    template <class TreeClass>
    void L2L(TreeClass& inTree){        
        for(long int idxLevel = 3 ; idxLevel <= configuration.getTreeHeight()-3 ; ++idxLevel){
            std::vector<std::reference_wrapper<CellLocalType>> children;
            long int* positionsOfChildren = new long int [spaceSystem.getNbChildrenPerCell()];

            children.emplace_back(locals[idxLevel+1]);
            positionsOfChildren[0] = (0);
            long int nbChildren = 1;

            kernel.L2L(inTree.getCellGroupsAtLevel(0).front().getCellSymbData(0),
                         idxLevel, TbfUtils::make_const(locals[idxLevel]), children,
                         positionsOfChildren, nbChildren);
        }
        {
            assert(inTree.getHeight() > 1);
            std::vector<std::reference_wrapper<CellLocalType>> children;
            long int* positionsOfChildren = new long int [spaceSystem.getNbChildrenPerCell()];
            long int nbChildren = 0;

            const long int idxLevelBase = 0;
            auto& lowerCellGroup = inTree.getCellGroupsAtLevel(idxLevelBase+1);

            auto currentLowerGroup = lowerCellGroup.begin();
            const auto endLowerGroup = lowerCellGroup.end();

            assert(currentLowerGroup != endLowerGroup);

            while(currentLowerGroup != endLowerGroup){
                assert(spaceSystem.getParentIndex(currentLowerGroup->getStartingSpacialIndex()) == 0
                       || 0 == spaceSystem.getParentIndex(currentLowerGroup->getEndingSpacialIndex()));

                for(long int idxCell = 0 ; idxCell < currentLowerGroup->getNbCells() ; ++idxCell){
                    assert(nbChildren < spaceSystem.getNbChildrenPerCell());
                    children.emplace_back(currentLowerGroup->getCellLocal(idxCell));
                    positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(currentLowerGroup->getCellSpacialIndex(idxCell));
                    nbChildren += 1;
                }

                ++currentLowerGroup;
            }

            assert(std::size(inTree.getCellGroupsAtLevel(idxLevelBase)));

            kernel.L2L(inTree.getCellGroupsAtLevel(0).front().getCellSymbData(0),
                         configuration.getTreeHeight()-2, TbfUtils::make_const(locals[configuration.getTreeHeight()-2]), children,
                         positionsOfChildren, nbChildren);
        }
    }


    ////////////////////////////////////////////////////////////////////////////////////

    static long int getExtendedTreeHeight(const SpacialConfiguration& /*inConfiguration*/, const long int inNbLevelsAbove0) {
        assert(-1 <= inNbLevelsAbove0);
        // return inConfiguration.getTreeHeight() + inNbLevelsAbove0;
        return inNbLevelsAbove0 + 4;
    }

    static long int  getExtendedTreeHeightBoundary(const SpacialConfiguration& /*inConfiguration*/, const long int inNbLevelsAbove0) {
        assert(-1 <= inNbLevelsAbove0);
        // return inConfiguration.getTreeHeight() + inNbLevelsAbove0 + 1;
        return inNbLevelsAbove0 + 5;
    }

    static long int GetNbRepetitionsPerDim(const long int inNbLevelsAbove0) {
        assert(-1 <= inNbLevelsAbove0);
        if( inNbLevelsAbove0 == -1 ){
            // We compute until the usual level 1
            // we know it is 3 times 3 box (-1;+1)
            return 3;
        }
        else if( inNbLevelsAbove0 == 0 ){
            // We compute until the usual level 1
            // we know it is 3 times 3 box (-1;+1)
            return 7;
        }
        return 6 * (1 << (inNbLevelsAbove0));
    }

    static auto GetExtendedBoxCenter(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0) {
        assert(-1 <= inNbLevelsAbove0);
        const auto originalBoxWidth = inConfiguration.getBoxWidths();
        const auto originalBoxCenter = inConfiguration.getBoxCenter();

        if( inNbLevelsAbove0 == -1 ){
            std::array<RealType, Dim> boxCenter;
            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                boxCenter[idxDim] = originalBoxCenter[idxDim] + originalBoxWidth[idxDim] * 0.5;
            }
            return  boxCenter;
        }
        else{
            const RealType offset = RealType(GetNbRepetitionsPerDim(inNbLevelsAbove0))/RealType(2.0);
            std::array<RealType, Dim> boxCenter;
            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                boxCenter[idxDim] = originalBoxCenter[idxDim] - (originalBoxWidth[idxDim] * 0.5) + offset;
            }
            return  boxCenter;
        }
    }

    static auto GetExtendedBoxCenterBoundary(const SpacialConfiguration& inConfiguration, [[maybe_unused]] const long int inNbLevelsAbove0) {
        assert(-1 <= inNbLevelsAbove0);
        const auto originalBoxWidth = inConfiguration.getBoxWidths();
        const auto originalBoxCenter = inConfiguration.getBoxCenter();

        std::array<RealType, Dim> boxCenter;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxCenter[idxDim] = originalBoxCenter[idxDim] + originalBoxWidth[idxDim] * 0.5;
        }
        return  boxCenter;
    }

    static auto GetExtendedBoxWidth(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0){
        assert(-1 <= inNbLevelsAbove0);
        auto boxWidths = inConfiguration.getBoxWidths();
        const RealType coef = (inNbLevelsAbove0 == -1 ? 2 : RealType(4<<(inNbLevelsAbove0)));
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxWidths[idxDim] *= coef;
        }
        return boxWidths;
    }

    static auto GetExtendedBoxWidthBoundary(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0){
        assert(-1 <= inNbLevelsAbove0);
        auto boxWidths = inConfiguration.getBoxWidths();
        const RealType coef = (inNbLevelsAbove0 == -1 ? 4 : RealType(8<<(inNbLevelsAbove0)));
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxWidths[idxDim] *= coef;
        }
        return boxWidths;
    }

public:
    static SpacialConfiguration GenerateAboveTreeConfiguration(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0){
        assert(-1 <= inNbLevelsAbove0);
        [[maybe_unused]] const auto boxWidths = GetExtendedBoxWidth(inConfiguration, inNbLevelsAbove0);
        [[maybe_unused]] const auto boxWidthsBoundary = GetExtendedBoxWidthBoundary(inConfiguration, inNbLevelsAbove0);

        [[maybe_unused]] const long int treeHeight = getExtendedTreeHeight(inConfiguration, inNbLevelsAbove0);
        [[maybe_unused]] const long int treeHeightBoundary = getExtendedTreeHeightBoundary(inConfiguration, inNbLevelsAbove0);

        [[maybe_unused]] const auto boxCenter = GetExtendedBoxCenter(inConfiguration, inNbLevelsAbove0);
        [[maybe_unused]] const auto boxCenterBoundary = GetExtendedBoxCenterBoundary(inConfiguration, inNbLevelsAbove0);

        return SpacialConfiguration(treeHeightBoundary, boxWidthsBoundary, boxCenterBoundary);
    }


    explicit TbfAlgorithmPeriodicTopTree(const SpacialConfiguration& inConfiguration, const long int inNbLevelsAbove0)
        : originalConfiguration(inConfiguration), configuration(GenerateAboveTreeConfiguration(inConfiguration, inNbLevelsAbove0)),
          originalSpaceSystem(originalConfiguration), spaceSystem(configuration), nbLevelsAbove0(inNbLevelsAbove0), kernel(configuration){

        multipoles.resize(configuration.getTreeHeight());
        locals.resize(configuration.getTreeHeight());
    }

    template <class SourceKernelClass,
              typename = typename std::enable_if<!std::is_same<long int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value
                                                 && !std::is_same<int, typename std::remove_const<typename std::remove_reference<SourceKernelClass>::type>::type>::value, void>::type>
    TbfAlgorithmPeriodicTopTree(const SpacialConfiguration& inConfiguration, SourceKernelClass&& inKernel, const long int inNbLevelsAbove0)
        : originalConfiguration(inConfiguration), configuration(GenerateAboveTreeConfiguration(inConfiguration, inNbLevelsAbove0)),
          originalSpaceSystem(originalConfiguration), spaceSystem(configuration), nbLevelsAbove0(inNbLevelsAbove0), kernel(std::forward<SourceKernelClass>(inKernel)){

        multipoles.resize(configuration.getTreeHeight());
        locals.resize(configuration.getTreeHeight());
    }

    template <class TreeClass>
    void execute(TreeClass& inTree, const int inOperationToProceed = TbfAlgorithmUtils::TbfOperations::TbfNearAndFarFields){
        // This has been done already by the real periodic FMM
        if(nbLevelsAbove0 < 0 || inTree.getHeight() == 0){
            return;
        }

        assert(originalConfiguration == inTree.getSpacialConfiguration());

        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2M){
            M2M(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfM2L){
            M2L(inTree);
        }
        if(inOperationToProceed & TbfAlgorithmUtils::TbfL2L){
            L2L(inTree);
        }
    }

    template <class FuncType>
    auto applyToAllKernels(FuncType&& inFunc) const {
        inFunc(kernel);
    }

    ////////////////////////////////////////////////////////////////////////

    long int getNbRepetitionsPerDim() const {
        return GetNbRepetitionsPerDim(nbLevelsAbove0);
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
        if( nbLevelsAbove0 == -1 ){
            // We know it is (-1;1)
            return std::pair<std::array<long int, Dim>,std::array<long int, Dim>>(
                        TbfUtils::make_array<long int, Dim>(-1),
                        TbfUtils::make_array<long int, Dim>(1));
        }
        else if( nbLevelsAbove0 == 0 ){
            // We know it is (-1;1)
            return std::pair<std::array<long int, Dim>,std::array<long int, Dim>>(
                        TbfUtils::make_array<long int, Dim>(-3),
                        TbfUtils::make_array<long int, Dim>(3));
        }
        else{
            const long int halfRepeated = int(getNbRepetitionsPerDim()/2);
            return std::pair<std::array<long int, Dim>,std::array<long int, Dim>>(
                        TbfUtils::make_array<long int, Dim>(-halfRepeated),
                        TbfUtils::make_array<long int, Dim>(halfRepeated-1));
        }
    }


    auto getExtendedIndex(const IndexType inIndexToFound, const long int inLevel) const{
        auto minBoxCorner = getRepetitionsIntervals().first;

        std::array<long int, Dim> margin;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            margin[idxDim] = std::abs(minBoxCorner[idxDim]) * (1 << inLevel);
        }

        auto coordToFound = spaceSystem.getBoxPosFromIndex(inIndexToFound);

        std::array<long int, Dim> extendedCoord;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            extendedCoord[idxDim] = margin[idxDim] + coordToFound[idxDim];
        }

        return spaceSystem.getIndexFromBoxPos(extendedCoord);
    }

    long int getExtendedLevel(const long int inLevel) const {
        return (configuration.getTreeHeight() - 2) + inLevel;
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfAlgorithmPeriodicTopTree& inAlgo) {
        inStream << "TbfAlgorithmPeriodicTopTree @ " << &inAlgo << "\n";
        inStream << " - nbLevelsAbove0: " << inAlgo.nbLevelsAbove0 << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inAlgo.configuration << "\n";
        inStream << " - Space system: " << "\n";
        inStream << inAlgo.spaceSystem << "\n";
        inStream << " - Original configuration: " << "\n";
        inStream << inAlgo.originalConfiguration << "\n";
        return inStream;
    }
};

#endif
