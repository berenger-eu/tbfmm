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

    static constexpr long int Dim = SpaceIndexType::Dim;

private:
    struct ContainerHeader {
        IndexType startingSpaceIndex;
        IndexType endingSpaceIndex;

        long int nbCells;
    };

    struct CellHeader {
        IndexType spaceIndex;
        std::array<long int, Dim> boxCoord;
    };


    using SymbolcMemoryBlockType = TbfMemoryBlock<TbfMemoryScalar<ContainerHeader>,
                                          TbfMemoryVector<CellHeader>>;

    using MultipoleMemoryBlockType = TbfMemoryBlock<TbfMemoryVector<MultipoleClass>>;

    using LocalMemoryBlockType = TbfMemoryBlock<TbfMemoryVector<LocalClass>>;

    SymbolcMemoryBlockType objectData;
    MultipoleMemoryBlockType objectMultipole;
    LocalMemoryBlockType objectLocal;

public:
#ifdef __NVCC__
    __device__ __host__
#endif
    explicit TbfCellsContainer(unsigned char* inObjectDataPtr, const size_t inObjectDataSize,
                               unsigned char* inObjectMultipolePtr, const size_t inObjectMultipoleSize,
                               unsigned char* inObjectLocalPtr, const size_t inObjectLocalSize,
                                   const bool inInitFromMemory = true)
        : objectData(inObjectDataPtr, inObjectDataSize, inInitFromMemory),
          objectMultipole(inObjectMultipolePtr, inObjectMultipoleSize, inInitFromMemory),
          objectLocal(inObjectLocalPtr, inObjectLocalSize, inInitFromMemory){

    }

#ifdef __NVCC__
    __device__ __host__
#endif
        explicit TbfCellsContainer(const std::array<std::pair<unsigned char*, size_t>, 3>& inPtrsSizes,
                                   const bool inInitFromMemory = true)
        : objectData(inPtrsSizes[0].first, inPtrsSizes[0].second, inInitFromMemory),
        objectMultipole(inPtrsSizes[1].first, inPtrsSizes[1].second, inInitFromMemory),
        objectLocal(inPtrsSizes[2].first, inPtrsSizes[2].second, inInitFromMemory){

    }

    template <class ContainerClass, class ConverterClass>
    explicit TbfCellsContainer(const ContainerClass& inCellSpatialIndexes, const ConverterClass& inConverter){
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
            cellsViewer.getItem(idxCell).boxCoord = inConverter.getBoxPosFromIndex(inCellSpatialIndexes[idxCell]);
        }
    }

    TbfCellsContainer(const TbfCellsContainer&) = delete;
    TbfCellsContainer& operator=(const TbfCellsContainer&) = delete;

    TbfCellsContainer(TbfCellsContainer&&) = default;
    TbfCellsContainer& operator=(TbfCellsContainer&&) = default;
#ifdef __NVCC__
    __device__ __host__
#endif
    IndexType getStartingSpacialIndex() const{
        return objectData.template getViewerForBlockConst<0>().getItem().startingSpaceIndex;
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    IndexType getEndingSpacialIndex() const{
        return objectData.template getViewerForBlockConst<0>().getItem().endingSpaceIndex;
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    long int getNbCells() const{
        return objectData.template getViewerForBlockConst<0>().getItem().nbCells;
    }

    /////////////////////////////////////////////////////////////////////////////////////
#ifdef __NVCC__
    __device__ __host__
#endif
    IndexType getCellSpacialIndex(const long int inIdxCell) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxCell).spaceIndex;
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    const std::array<long int, Dim>& getCellBoxCoord(const long int inIdxCell) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxCell).boxCoord;
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    const CellHeader& getCellSymbData(const long int inIdxCell) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxCell);
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    MultipoleClass& getCellMultipole(const long int inIdxCell) {
        return objectMultipole.template getViewerForBlock<0>().getItem(inIdxCell);
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    const MultipoleClass& getCellMultipole(const long int inIdxCell) const {
        return objectMultipole.template getViewerForBlockConst<0>().getItem(inIdxCell);
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    LocalClass& getCellLocal(const long int inIdxCell) {
        return objectLocal.template getViewerForBlock<0>().getItem(inIdxCell);
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    const LocalClass& getCellLocal(const long int inIdxCell) const {
        return objectLocal.template getViewerForBlockConst<0>().getItem(inIdxCell);
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef __NVCC__
    __device__ __host__
#endif
    void initMemoryBlockHeader(){
        objectData.initHeader();
        objectMultipole.initHeader();
        objectLocal.initHeader();
    }

    auto getDataPtrsAndSizes(){
        return std::array<std::pair<unsigned char*,size_t>,3>{std::pair<unsigned char*,size_t>{objectData.getPtr(), objectData.getAllocatedMemorySizeInByte()},
                                                                 std::pair<unsigned char*,size_t>{objectMultipole.getPtr(), objectMultipole.getAllocatedMemorySizeInByte()},
                                                                 std::pair<unsigned char*,size_t>{objectLocal.getPtr(), objectLocal.getAllocatedMemorySizeInByte()}};

    }

    auto getDataPtrsAndSizes() const{
        return std::array<std::pair<const unsigned char*,size_t>,3>{std::pair<unsigned char*,size_t>{objectData.getPtr(), objectData.getAllocatedMemorySizeInByte()},
                                                                 std::pair<unsigned char*,size_t>{objectMultipole.getPtr(), objectMultipole.getAllocatedMemorySizeInByte()},
                                                                 std::pair<unsigned char*,size_t>{objectLocal.getPtr(), objectLocal.getAllocatedMemorySizeInByte()}};

    }

    unsigned char* getDataPtr(){
        return objectData.getPtr();
    }

    const unsigned char* getDataPtr() const {
        return objectData.getPtr();
    }

    auto getDataSize() const {
        return objectData.getAllocatedMemorySizeInByte();
    }

    unsigned char* getMultipolePtr(){
        return objectMultipole.getPtr();
    }

    const unsigned char* getMultipolePtr() const {
        return objectMultipole.getPtr();
    }

    auto getMultipoleSize() const {
        return objectMultipole.getAllocatedMemorySizeInByte();
    }

    unsigned char* getLocalPtr(){
        return objectLocal.getPtr();
    }

    const unsigned char* getLocalPtr() const {
        return objectLocal.getPtr();
    }

    auto getLocalSize() const {
        return objectLocal.getAllocatedMemorySizeInByte();
    }

    ///////////////////////////////////////////////////////////////////////////

#ifdef __NVCC__
    __device__ __host__
#endif
    std::optional<long int> getElementFromSpacialIndex(const IndexType inIndex) const {
        //        for (long int idxCell = 0 ; idxCell < header.nbCells ; ++idxCell) {
        //            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);
        //            if(cellHeader.spaceIndex == inIndex){
        //                return std::optional<long int>(idxCell);
        //            }
        //        }
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        const long int idxCell = TbfUtils::lower_bound_indexes( 0, header.nbCells, inIndex, [this](const auto& idxCellIterate, const auto& index){
            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCellIterate);
            return cellHeader.spaceIndex < index;
        });

        if(idxCell == header.nbCells){
            return std::nullopt;
        }

        const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);
        if(cellHeader.spaceIndex != inIndex){
            return std::nullopt;
        }

        return std::optional<long int>(idxCell);
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    std::optional<long int> getElementFromParentIndex(const SpaceIndexType& spaceSystem, const IndexType inParentIndex) const {        
        //        for (long int idxCell = 0 ; idxCell < header.nbCells ; ++idxCell) {
        //            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);
        //            if(spaceSystem.getParentIndex(cellHeader.spaceIndex) == inParentIndex){
        //                return std::optional<long int>(idxCell);
        //            }
        //        }
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        const long int idxCell = TbfUtils::lower_bound_indexes( 0, header.nbCells, inParentIndex, [this, &spaceSystem](const auto& idxCellIterate, const auto& parentIndex){
            const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCellIterate);
            return (spaceSystem.getParentIndex(cellHeader.spaceIndex) < parentIndex);
        });

        if(idxCell == header.nbCells){
            return std::nullopt;
        }

        const CellHeader& cellHeader = objectData.template getViewerForBlockConst<1>().getItem(idxCell);
        if(spaceSystem.getParentIndex(cellHeader.spaceIndex) != inParentIndex){
            return std::nullopt;
        }

        return std::optional<long int>(idxCell);
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

    //////////////////////////////////////////////////////////////////////////////////////////////////

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfCellsContainer& inCellContainer) {
        inStream << "Cell block @ " << &inCellContainer << "\n";
        inStream << " - size " << inCellContainer.getNbCells() << "\n";
        inStream << " - starting index " << inCellContainer.getStartingSpacialIndex() << "\n";
        inStream << " - ending index " << inCellContainer.getEndingSpacialIndex() << "\n";
        return inStream;
    }
};



#endif
