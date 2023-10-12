#ifndef TBFPARTICLECONTAINER_HPP
#define TBFPARTICLECONTAINER_HPP

#include "tbfglobal.hpp"

#include "containers/tbfmemoryblock.hpp"
#include "containers/tbfmemoryscalar.hpp"
#include "containers/tbfmemoryvector.hpp"
#include "containers/tbfmemorymultirvector.hpp"
#include "tbfparticlesorter.hpp"

#include <array>
#include <optional>
#include <cassert>

template <class RealType_T, class DataType_T, long int NbDataValuesPerParticle_T,
          class RhsType_T, long int NbRhsValuesPerParticle_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfParticlesContainer{
public:
    using RealType = RealType_T;
    using DataType = DataType_T;
    constexpr static long int NbDataValuesPerParticle = NbDataValuesPerParticle_T;
    using RhsType = RhsType_T;
    constexpr static long int NbRhsValuesPerParticle = NbRhsValuesPerParticle_T;
    using SpaceIndexType = SpaceIndexType_T;
    using IndexType = typename SpaceIndexType::IndexType;

    static constexpr long int Dim = SpaceIndexType::Dim;

private:
    struct ContainerHeader {
        IndexType startingSpaceIndex;
        IndexType endingSpaceIndex;

        long int nbLeaves;
        long int nbParticles;
    };

    struct LeafHeader {
        IndexType spaceIndex;
        long int nbParticles;
        long int offSet;
        std::array<long int, Dim> boxCoord;
    };

    using SymbolcMemoryBlockType = TbfMemoryBlock<TbfMemoryScalar<ContainerHeader>,
                                          TbfMemoryVector<LeafHeader>,
                                          TbfMemoryVector<long int>,
                                          TbfMemoryMultiRVector<DataType, NbDataValuesPerParticle>>;

    using RhsMemoryBlockType = TbfMemoryBlock<TbfMemoryMultiRVector<RhsType, NbRhsValuesPerParticle>>;

    SymbolcMemoryBlockType objectData;
    RhsMemoryBlockType objectRhs;

public:
#ifdef __NVCC__
    __device__ __host__
#endif
    explicit TbfParticlesContainer(unsigned char* inObjectDataPtr, const size_t inObjectDataSize,
                               unsigned char* inObjectRhsPtr, const size_t inObjectRhsSize,
                                   const bool inInitFromMemory = true)
        : objectData(inObjectDataPtr, inObjectDataSize, inInitFromMemory),
        objectRhs(inObjectRhsPtr, inObjectRhsSize, inInitFromMemory){

    }

#ifdef __NVCC__
    __device__ __host__
#endif
    explicit TbfParticlesContainer(const std::array<std::pair<unsigned char*,size_t>,2>& inPtrsSizes,
                                       const bool inInitFromMemory = true)
        : objectData(inPtrsSizes[0].first, inPtrsSizes[0].second, inInitFromMemory),
        objectRhs(inPtrsSizes[1].first, inPtrsSizes[1].second, inInitFromMemory){

    }

    TbfParticlesContainer(const TbfParticlesContainer&) = delete;
    TbfParticlesContainer& operator=(const TbfParticlesContainer&) = delete;

    TbfParticlesContainer(TbfParticlesContainer&&) = default;
    TbfParticlesContainer& operator=(TbfParticlesContainer&&) = default;

    template <class GroupInfoClass, class ContainerClass, class ConverterClass>
    explicit TbfParticlesContainer(const GroupInfoClass& inParticleGroupInfo, const ContainerClass& inParticlePositions,
                                   const ConverterClass& inConverter){
        const long int nbParticles = inParticleGroupInfo.getNbParticles();

        const std::array<long int, 4> sizesData{{1, inParticleGroupInfo.getNbLeaves(),
                                           nbParticles*1,
                                           nbParticles*NbDataValuesPerParticle}};
        objectData.resetBlocksFromSizes(sizesData);

        ContainerHeader& header = objectData.template getViewerForBlock<0>().getItem();
        header.startingSpaceIndex   = inParticleGroupInfo.getSpacialIndexForLeaf(0);
        header.endingSpaceIndex     = inParticleGroupInfo.getSpacialIndexForLeaf(inParticleGroupInfo.getNbLeaves()-1);
        header.nbLeaves             = inParticleGroupInfo.getNbLeaves();
        header.nbParticles          = nbParticles;

        auto leavesViewer = objectData.template getViewerForBlock<1>();
        auto particlesIndexViewer = objectData.template getViewerForBlock<2>();
        auto particlesDataViewer = objectData.template getViewerForBlock<3>();

        long int idxCurrentLeaf = 0;
        leavesViewer.getItem(idxCurrentLeaf).spaceIndex = inParticleGroupInfo.getSpacialIndexForLeaf(0);
        leavesViewer.getItem(idxCurrentLeaf).nbParticles = 0;
        leavesViewer.getItem(idxCurrentLeaf).offSet = 0;
        leavesViewer.getItem(idxCurrentLeaf).boxCoord = inConverter.getBoxPosFromIndex(leavesViewer.getItem(idxCurrentLeaf).spaceIndex);

        for(long int idxPart = 0 ; idxPart < nbParticles ; ++idxPart){
            auto spacialIndexPart = inParticleGroupInfo.getSpacialIndexForParticle(idxPart);
            if(spacialIndexPart != leavesViewer.getItem(idxCurrentLeaf).spaceIndex){
                idxCurrentLeaf += 1;
                assert(idxCurrentLeaf < inParticleGroupInfo.getNbLeaves());
                leavesViewer.getItem(idxCurrentLeaf).spaceIndex = inParticleGroupInfo.getSpacialIndexForLeaf(idxCurrentLeaf);
                leavesViewer.getItem(idxCurrentLeaf).nbParticles = 0;
                leavesViewer.getItem(idxCurrentLeaf).offSet = idxPart;
                leavesViewer.getItem(idxCurrentLeaf).boxCoord = inConverter.getBoxPosFromIndex(leavesViewer.getItem(idxCurrentLeaf).spaceIndex);
            }
            leavesViewer.getItem(idxCurrentLeaf).nbParticles += 1;

            const long int originalParticleIdx = inParticleGroupInfo.getParticleIndex(idxPart);
            particlesIndexViewer.getItem(idxPart) = originalParticleIdx;

            for(long int idxValue = 0 ; idxValue < NbDataValuesPerParticle ; ++idxValue){
                particlesDataViewer.getItem(idxPart, idxValue) = inParticlePositions[originalParticleIdx][idxValue];
            }
        }

        const std::array<long int, 1> sizesRhs{{nbParticles*NbRhsValuesPerParticle}};
        objectRhs.resetBlocksFromSizes(sizesRhs);
        auto particlesRhsViewer = objectRhs.template getViewerForBlock<0>();
        for(long int idxPart = 0 ; idxPart < nbParticles ; ++idxPart){
            for(long int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                particlesRhsViewer.getItem(idxPart, idxValue) = RhsType();
            }
        }
    }

    template <class ContainerClass>
    explicit TbfParticlesContainer(const SpaceIndexType& inSpaceSystem, const ContainerClass& inParticlePositions){
        const long int nbParticles = static_cast<long int>(std::size(inParticlePositions));

        if(nbParticles == 0){
            const std::array<long int, 4> sizes{{1, 0, 0, 0}};
            objectData.resetBlocksFromSizes(sizes);
            ContainerHeader& header = objectData.template getViewerForBlock<0>().getItem();
            header.startingSpaceIndex   = 0;
            header.endingSpaceIndex     = 0;
            header.nbLeaves             = 0;
            header.nbParticles          = 0;
            return;
        }

        TbfParticleSorter<RealType> partSorter(inSpaceSystem, inParticlePositions);
        auto groups = partSorter.splitInGroups(partSorter.getNbLeaves());
        assert(std::size(groups) == 1);

        (*this) = TbfParticlesContainer(groups.front(), inParticlePositions, inSpaceSystem);
    }

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
    long int getNbParticles() const{
        return objectData.template getViewerForBlockConst<0>().getItem().nbParticles;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    long int getNbLeaves() const{
        return objectData.template getViewerForBlockConst<0>().getItem().nbLeaves;
    }

    ///////////////////////////////////////////////////////////////////////////

#ifdef __NVCC__
    __device__ __host__
#endif
    IndexType getLeafSpacialIndex(const long int inIdxLeaf) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxLeaf).spaceIndex;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    const std::array<long int, Dim>& getLeafBoxCoord(const long int inIdxLeaf) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxLeaf).boxCoord;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    const LeafHeader& getLeafSymbData(const long int inIdxLeaf) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxLeaf);
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    long int getNbParticlesInLeaf(const long int inIdxLeaf) const{
        return objectData.template getViewerForBlockConst<1>().getItem(inIdxLeaf).nbParticles;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
const long int* getParticleIndexes(const long int inIdxLeaf) const {
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();
        const auto& leafHeader = leavesViewer.getItem(inIdxLeaf);
        return &objectData.template getViewerForBlockConst<2>().getItem(leafHeader.offSet);
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    long int* getParticleIndexes(const long int inIdxLeaf) {
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();
        const auto& leafHeader = leavesViewer.getItem(inIdxLeaf);
        return &objectData.template getViewerForBlock<2>().getItem(leafHeader.offSet);
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    std::array<const DataType*, NbDataValuesPerParticle> getParticleData(const long int inIdxLeaf) const {
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();
        const auto& leafHeader = leavesViewer.getItem(inIdxLeaf);
        std::array<const DataType*, NbDataValuesPerParticle> particleDataPtr;
        for(long int idxData = 0 ; idxData < NbDataValuesPerParticle ; ++idxData){
            particleDataPtr[idxData] = &objectData.template getViewerForBlockConst<3>().getItem(leafHeader.offSet, idxData);
        }
        return particleDataPtr;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    std::array<DataType*, NbDataValuesPerParticle> getParticleData(const long int inIdxLeaf) {
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();
        const auto& leafHeader = leavesViewer.getItem(inIdxLeaf);
        std::array<DataType*, NbDataValuesPerParticle> particleDataPtr;
        for(long int idxData = 0 ; idxData < NbDataValuesPerParticle ; ++idxData){
            particleDataPtr[idxData] = &objectData.template getViewerForBlock<3>().getItem(leafHeader.offSet, idxData);
        }
        return particleDataPtr;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    const std::array<const RhsType*, NbRhsValuesPerParticle> getParticleRhs(const long int inIdxLeaf) const {
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();
        const auto& leafHeader = leavesViewer.getItem(inIdxLeaf);
        std::array<const RhsType*, NbRhsValuesPerParticle> particleRhsPtr;
        if(!objectRhs.isEmpty()){
            for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                particleRhsPtr[idxRhs] = &objectRhs.template getViewerForBlockConst<0>().getItem(leafHeader.offSet, idxRhs);
            }
        }
        else{
            for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                particleRhsPtr[idxRhs] = nullptr;
            }
        }
        return particleRhsPtr;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    std::array<RhsType*, NbRhsValuesPerParticle> getParticleRhs(const long int inIdxLeaf) {
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();
        const auto& leafHeader = leavesViewer.getItem(inIdxLeaf);
        std::array<RhsType*, NbRhsValuesPerParticle> particleRhsPtr;
        if(!objectRhs.isEmpty()){
            for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                particleRhsPtr[idxRhs] = &objectRhs.template getViewerForBlock<0>().getItem(leafHeader.offSet, idxRhs);
            }
        }
        else{
            for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                particleRhsPtr[idxRhs] = nullptr;
            }
        }
        return particleRhsPtr;
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef __NVCC__
    __device__ __host__
#endif
    std::optional<long int> getElementFromSpacialIndex(const IndexType inIndex) const {
        //        for(long int idxLeaf = 0 ; idxLeaf < header.nbLeaves ; ++idxLeaf){
        //            const auto& leafHeader = leavesViewer.getItem(idxLeaf);
        //            if(leafHeader.spaceIndex == inIndex){
        //                return std::optional<long int>(idxLeaf);
        //            }
        //        }
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();
        auto leavesViewer = objectData.template getViewerForBlockConst<1>();

        const long int idxLeaf = TbfUtils::lower_bound_indexes( 0, header.nbLeaves, inIndex, [&leavesViewer](const auto& idxLeafIterate, const auto& index){
            const auto& leafHeader = leavesViewer.getItem(idxLeafIterate);
            return (leafHeader.spaceIndex < index);
        });

        if(idxLeaf == header.nbLeaves){
            return std::nullopt;
        }

        const auto& leafHeader = leavesViewer.getItem(idxLeaf);
        if(leafHeader.spaceIndex != inIndex){
            return std::nullopt;
        }

        return std::optional<long int>(idxLeaf);
    }

    ///////////////////////////////////////////////////////////////////////////
#ifdef __NVCC__
    __device__ __host__
#endif
    void initMemoryBlockHeader(){
        objectData.initHeader();
        objectRhs.initHeader();
    }

    auto getDataPtrsAndSizes(){
        return std::array<std::pair<unsigned char*,size_t>,2>{std::pair<unsigned char*,size_t>{objectData.getPtr(), objectData.getAllocatedMemorySizeInByte()},
                                                                 std::pair<unsigned char*,size_t>{objectRhs.getPtr(), objectRhs.getAllocatedMemorySizeInByte()}};

    }

    auto getDataPtrsAndSizes() const{
        return std::array<std::pair<const unsigned char*,size_t>,2>{std::pair<unsigned char*,size_t>{objectData.getPtr(), objectData.getAllocatedMemorySizeInByte()},
                                                                 std::pair<unsigned char*,size_t>{objectRhs.getPtr(), objectRhs.getAllocatedMemorySizeInByte()}};

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

    unsigned char* getRhsPtr(){
        return objectRhs.getPtr();
    }

    const unsigned char* getRhsPtr() const {
        return objectRhs.getPtr();
    }

    auto getRhsSize() const {
        return objectRhs.getAllocatedMemorySizeInByte();
    }

    ///////////////////////////////////////////////////////////////////////////

    template <class FuncClass>
    void applyToAllLeaves(FuncClass&& inFunc) const {
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        auto leavesViewer = objectData.template getViewerForBlockConst<1>();

        for(long int idxLeaf = 0 ; idxLeaf < header.nbLeaves ; ++idxLeaf){
            const auto& leafHeader = leavesViewer.getItem(idxLeaf);

            const long int* particleIndexes = &objectData.template getViewerForBlockConst<2>().getItem(leafHeader.offSet);

            std::array<const DataType*, NbDataValuesPerParticle> particleDataPtr;
            for(long int idxData = 0 ; idxData < NbDataValuesPerParticle ; ++idxData){
                particleDataPtr[idxData] = &objectData.template getViewerForBlockConst<3>().getItem(leafHeader.offSet, idxData);
            }

            std::array<const RhsType*, NbRhsValuesPerParticle> particleRhsPtr;
            if(!objectRhs.isEmpty()){
                for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                    particleRhsPtr[idxRhs] = &objectRhs.template getViewerForBlockConst<0>().getItem(leafHeader.offSet, idxRhs);
                }
            }
            else{
                for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                    particleRhsPtr[idxRhs] = nullptr;
                }
            }

            inFunc(leafHeader, particleIndexes, particleDataPtr, particleRhsPtr);
        }
    }

    template <class FuncClass>
    void applyToAllLeaves(FuncClass&& inFunc) {
        const ContainerHeader& header = objectData.template getViewerForBlockConst<0>().getItem();

        auto leavesViewer = objectData.template getViewerForBlock<1>();

        for(long int idxLeaf = 0 ; idxLeaf < header.nbLeaves ; ++idxLeaf){
            auto& leafHeader = leavesViewer.getItem(idxLeaf);

            long int* particleIndexes = &objectData.template getViewerForBlock<2>().getItem(leafHeader.offSet);

            std::array<DataType*, NbDataValuesPerParticle> particleDataPtr;
            for(long int idxData = 0 ; idxData < NbDataValuesPerParticle ; ++idxData){
                particleDataPtr[idxData] = &objectData.template getViewerForBlock<3>().getItem(leafHeader.offSet, idxData);
            }

            std::array<RhsType*, NbRhsValuesPerParticle> particleRhsPtr;
            if(!objectRhs.isEmpty()){
                for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                    particleRhsPtr[idxRhs] = &objectRhs.template getViewerForBlock<0>().getItem(leafHeader.offSet, idxRhs);
                }
            }
            else{
                for(long int idxRhs = 0 ; idxRhs < NbRhsValuesPerParticle ; ++idxRhs){
                    particleRhsPtr[idxRhs] = nullptr;
                }
            }

            inFunc(leafHeader, particleIndexes, particleDataPtr, particleRhsPtr);
        }
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfParticlesContainer& inParticlesContainer) {
        inStream << "Particle block @ " << &inParticlesContainer << "\n";
        inStream << " - size " << inParticlesContainer.getNbLeaves() << "\n";
        inStream << " - starting index " << inParticlesContainer.getStartingSpacialIndex() << "\n";
        inStream << " - ending index " << inParticlesContainer.getEndingSpacialIndex() << "\n";
        return inStream;
    }
};


#endif

