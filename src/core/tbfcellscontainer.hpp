#ifndef TBFCELLSCONTAINER_HPP
#define TBFCELLSCONTAINER_HPP

#include "tbfglobal.hpp"

#include "containers/tbfmemoryblock.hpp"
#include "containers/tbfmemoryscalar.hpp"
#include "containers/tbfmemoryvector.hpp"

#include <array>
#include <optional>

template <class RealType_T, class MultipoleClass_T, class LocalClass_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfCellsContainer{
public:
    using RealType = RealType_T;
    using MultipoleClass = MultipoleClass_T;
    using LocalClass = LocalClass_T;
    using SpaceIndexType = SpaceIndexType_T;
    using IndexType = typename SpaceIndexType::IndexType;

private:
    struct ContainerHeader {
        IndexType startingSpaceIndex;
        IndexType endingSpaceIndex;

        long int nbCells;
    };

    struct CellHeader {
        IndexType spaceIndex;
    };


    using SymbolcMemoryBlockType = TbfMemoryBlock<TbfMemoryScalar<ContainerHeader>,
                                          TbfMemoryVector<CellHeader>>;

    using MultipoleMemoryBlockType = TbfMemoryBlock<TbfMemoryVector<MultipoleClass>>;

    using LocalMemoryBlockType = TbfMemoryBlock<TbfMemoryVector<LocalClass>>;

    SymbolcMemoryBlockType objectData;
    MultipoleMemoryBlockType objectMultipole;
    LocalMemoryBlockType objectLocal;

public:
    template <class ContainerClass>
    explicit TbfCellsContainer(const ContainerClass& inCellSpatialIndexes){
        const long int nbCells = static_cast<long int>(std::size(inCellSpatialIndexes));

        if(nbCells == 0){
            const std::array<long int, 2> sizes{{1, 0}};
            objectData.resetBlocksFromSizes(sizes);
            ContainerHeader& header = objectData.template getViewerForBlock<0>().getItem();
            header.startingSpaceIndex   = 0;
            header.endingSpaceIndex     = 0;
            header.nbCells              = 0;

            objectMultipole.resetBlocksFromSizes(std::array<long int, 1>{{0}});
            objectLocal.resetBlocksFromSizes(std::array<long int, 1>{{0}});

            return;
        }

        const std::array<long int, 2> sizes{{1, nbCells}};
        objectData.resetBlocksFromSizes(sizes);
        objectMultipole.resetBlocksFromSizes(std::array<long int, 1>{{nbCells}});
        objectLocal.resetBlocksFromSizes(std::array<long int, 1>{{nbCells}});

        ContainerHeader& header = objectData.template getViewerForBlock<0>().getItem();
        header.startingSpaceIndex   = inCellSpatialIndexes.front();
        header.endingSpaceIndex     = inCellSpatialIndexes.back();
        header.nbCells              = nbCells;

        auto cellsViewer = objectData.template getViewerForBlock<1>();

        for(long int idxCell = 0 ; idxCell < nbCells ; ++idxCell){
            cellsViewer.getItem(idxCell).spaceIndex = inCellSpatialIndexes[idxCell];
        }
    }

    TbfCellsContainer(const TbfCellsContainer&) = delete;
    TbfCellsContainer& operator=(const TbfCellsContainer&) = delete;

    TbfCellsContainer(TbfCellsContainer&&) = default;
    TbfCellsContainer& operator=(TbfCellsContainer&&) = default;

    IndexType getStartingSpacialIndex() const{
        return objectData.template getViewerForBlockConst<0>().getItem().startingSpaceIndex;
    }

    IndexType getEndingSpacialIndex() const{
        return objectData.template getViewerForBlockConst<0>().getItem().endingSpaceIndex;
    }

    long int getNbCells() const{
        return objectData.template getViewerForBlockConst<0>().getItem().nbCells;
    }

    /////////////////////////////////////////////////////////////////////////////////////

    IndexType getCellSpacialIndex(const long int inIdxCell) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxCell).spaceIndex;
    }

    MultipoleClass& getCellMultipole(const long int inIdxCell) {
        return objectMultipole.template getViewerForBlock<0>().getItem(inIdxCell);
    }

    const MultipoleClass& getCellMultipole(const long int inIdxCell) const {
        return objectMultipole.template getViewerForBlockConst<0>().getItem(inIdxCell);
    }

    LocalClass& getCellLocal(const long int inIdxCell) {
        return objectLocal.template getViewerForBlock<0>().getItem(inIdxCell);
    }

    const LocalClass& getCellLocal(const long int inIdxCell) const {
        return objectLocal.template getViewerForBlockConst<0>().getItem(inIdxCell);
    }

    ///////////////////////////////////////////////////////////////////////////

    std::optional<long int> getElementFromSpacialIndex(const IndexType inIndex) const {
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        // TODO use binary search
        for (long int idxCell = 0 ; idxCell < header.nbCells ; ++idxCell) {
            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);
            if(cellHeader.spaceIndex == inIndex){
                return std::optional<long int>(idxCell);
            }
        }
        return std::nullopt;
    }


    std::optional<long int> getElementFromParentIndex(const SpaceIndexType& spaceSystem, const IndexType inParentIndex) const {
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        // TODO use binary search
        for (long int idxCell = 0 ; idxCell < header.nbCells ; ++idxCell) {
            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);
            if(spaceSystem.getParentIndex(cellHeader.spaceIndex) == inParentIndex){
                return std::optional<long int>(idxCell);
            }
        }
        return std::nullopt;
    }

    /////////////////////////////////////////////////////////////////////////////////////

    template <class FuncClass>
    void applyToAllCells(const long int inLevel, FuncClass&& inFunc) {
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        for (long int idxCell = 0 ; idxCell < header.nbCells ; ++idxCell) {
            CellHeader& cellHeader = objectData.template getViewerForBlock<1>().getItem(idxCell);

            std::optional<std::reference_wrapper<MultipoleClass>> cellMultipole;
            if(!objectMultipole.isEmpty()){
                cellMultipole = objectMultipole.template getViewerForBlock<0>().getItem(idxCell);
            }

            std::optional<std::reference_wrapper<LocalClass>> cellLocal;
            if(!objectLocal.isEmpty()){
                cellLocal = objectLocal.template getViewerForBlock<0>().getItem(idxCell);
            }

            inFunc(inLevel, cellHeader, cellMultipole, cellLocal);
        }
    }

    template <class FuncClass>
    void applyToAllCells(const long int inLevel, FuncClass&& inFunc) const {
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        for (long int idxCell = 0 ; idxCell < header.nbCells ; ++idxCell) {
            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);

            std::optional<std::reference_wrapper<const MultipoleClass>> cellMultipole;
            if(!objectMultipole.isEmpty()){
                cellMultipole = objectMultipole.template getViewerForBlockConst<0>().getItem(idxCell);
            }

            std::optional<std::reference_wrapper<const LocalClass>> cellLocal;
            if(!objectLocal.isEmpty()){
                cellLocal = objectLocal.template getViewerForBlockConst<0>().getItem(idxCell);
            }

            inFunc(inLevel, cellHeader, cellMultipole, cellLocal);
        }
    }
};


#endif