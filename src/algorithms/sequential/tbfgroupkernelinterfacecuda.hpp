#ifndef TBFGROUPKERNELINTERFACECUDA_HPP
#define TBFGROUPKERNELINTERFACECUDA_HPP

#include "tbfglobal.hpp"
#include "utils/tbfutils.hpp"

#include <cassert>
#include <cuda_runtime.h>

#include <vector>
#include <utility>

#define CUDA_ASSERT(X)\
{\
        cudaError_t ___resCuda = (X);\
        if ( cudaSuccess != ___resCuda ){\
            printf("Error: fails, %s (%s line %d)\n", cudaGetErrorString(___resCuda), __FILE__, __LINE__ );\
            std::abort();\
    }\
}


namespace TbfGroupKernelInterfaceCuda_core{

__device__ static int GetThreadId(){
    return threadIdx.x + blockIdx.x*blockDim.x;
}

__device__ static int GetBlockId(){
    return blockIdx.x;
}

__device__ static int GetNbThreads(){
    return blockDim.x*gridDim.x;
}

__device__ static int GetNbBlocks(){
    return gridDim.x;
}




template <class KernelClass, class ParticleGroupClass, class LeafGroupClass>
__global__ void P2M_core(KernelClass inKernel,
                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCells) {
    const ParticleGroupClass inParticleGroup(ptrsAndSizeParticles);
    LeafGroupClass inLeafGroup(ptrsAndSizeCells);

    assert(inParticleGroup.getNbLeaves() == inLeafGroup.getNbCells());
    for(long int idxLeaf = GetBlockId() ; idxLeaf < inParticleGroup.getNbLeaves() ; idxLeaf += GetNbBlocks()){
        assert(inParticleGroup.getLeafSpacialIndex(idxLeaf) == inLeafGroup.getCellSpacialIndex(idxLeaf));
        const auto& symbData = TbfUtils::make_const(inLeafGroup).getCellSymbData(idxLeaf);
        const auto& particlesData = inParticleGroup.getParticleData(idxLeaf);
        auto&& leafData = inLeafGroup.getCellMultipole(idxLeaf);
        inKernel.CudaP2M(symbData, inParticleGroup.getParticleIndexes(idxLeaf), particlesData, inParticleGroup.getNbParticlesInLeaf(idxLeaf),
                         leafData);
    }
}


template <class SpaceIndexType, class KernelClass, class CellGroupClass>
__global__ void M2M_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass inKernel,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCellsLower,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCellsUpper,
                         const long int inIdxFirstParent, const long int inIdxLimitParent,
                         const long int* interactionOffset) {
    CellGroupClass inUpperGroup(ptrsAndSizeCellsUpper);
    const CellGroupClass inLowerGroup(ptrsAndSizeCellsLower);

    using CellMultipoleType = typename std::remove_reference<decltype(inLowerGroup.getCellMultipole(0))>::type;
    std::vector<std::reference_wrapper<const CellMultipoleType>> children;
    long int positionsOfChildren[spaceSystem.getNbChildrenPerCell()];

    for(long int idxParent = inIdxFirstParent+GetThreadId() ;
         idxParent < inIdxLimitParent ; idxParent += GetNbThreads() ){
        const long int nbChildren = interactionOffset[idxParent-inIdxFirstParent+1]-interactionOffset[idxParent-inIdxFirstParent];
        assert(nbChildren >= 1);

        for(long int idxChild = 0 ; idxChild < nbChildren ; ++idxChild){
            children.emplace_back(inLowerGroup.getCellMultipole(idxChild));
            positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(inLowerGroup.getCellSpacialIndex(idxChild));
        }

        inKernel.M2MCuda(inUpperGroup.getCellSymbData(idxParent),
                         inLevel, TbfUtils::make_const(children), inUpperGroup.getCellMultipole(idxParent),
                         positionsOfChildren, nbChildren);

        children.clear();
    }
}


template <class SpaceIndexType, class KernelClass, class CellGroupClass, class IndexClass>
__global__ void M2LInGroup_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass inKernel,
                                std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCells,
                                const IndexClass* inIndexes,
                                const long int inNbInteractionBlocks,
                                const long int* inInteractionBlocks) {
    CellGroupClass inCellGroup(ptrsAndSizeCells);

    using CellMultipoleType = typename std::remove_reference<decltype(inCellGroup.getCellMultipole(0))>::type;
    //using CellLocalType = typename std::remove_reference<decltype(inCellGroup.getCellLocal(0))>::type;

    std::vector<std::reference_wrapper<const CellMultipoleType>> neighbors;
    long int positionsOfNeighbors[spaceSystem.getNbInteractionsPerCell()];


    for(long int idxInteractionBlock = GetThreadId() ; idxInteractionBlock < inNbInteractionBlocks ; idxInteractionBlock += GetNbThreads()){
        const auto interaction = inIndexes[inInteractionBlocks[idxInteractionBlock]];

        auto& targetCell = inCellGroup.getCellLocal(interaction.globalTargetPos);
        const long int nbNeighbors = inInteractionBlocks[idxInteractionBlock+1]-inInteractionBlocks[idxInteractionBlock];

        for(long int idxInteraction = inInteractionBlocks[idxInteractionBlock] ; idxInteraction < inInteractionBlocks[idxInteractionBlock+1] ; ++idxInteraction){
            auto foundSrc = inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexSrc);
            assert(foundSrc);
            assert(inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget)
                   && *inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget) == inIndexes[idxInteraction].globalTargetPos);

            assert(nbNeighbors < spaceSystem.getNbInteractionsPerCell());
            neighbors.emplace_back(inCellGroup.getCellMultipole(*foundSrc));
            positionsOfNeighbors[nbNeighbors] = inIndexes[idxInteraction].arrayIndexSrc;
        }

        inKernel.M2LCuda(inCellGroup.getCellSymbData(interaction.globalTargetPos),
                         inLevel,
                         TbfUtils::make_const(neighbors),
                         positionsOfNeighbors,
                         nbNeighbors,
                         targetCell);
        neighbors.clear();
    }
}

template <class SpaceIndexType, class KernelClass, class CellGroupClassTarget, class CellGroupClassSource, class IndexClass>
__global__ void M2LBetweenGroups_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass inKernel,
                                      std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCells,
                                      std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeOther,
                                      const IndexClass* inIndexes,
                                      const long int inNbInteractionBlocks, const long int* inInteractionBlocksOffset,
                                      const long int* inInteractionBlockIdxs, const long int* inFoundSrcIdxs) {
    CellGroupClassTarget inCellGroup(ptrsAndSizeCells);
    const CellGroupClassSource inOtherCellGroup(ptrsAndSizeOther);

    using CellMultipoleType = typename std::remove_reference<decltype(inOtherCellGroup.getCellMultipole(0))>::type;
    //using CellLocalType = typename std::remove_reference<decltype(inCellGroup.getCellLocal(0))>::type;

    std::vector<std::reference_wrapper<const CellMultipoleType>> neighbors;
    long int positionsOfNeighbors[spaceSystem.getNbInteractionsPerCell()];

    for(long int idxInteractionBlock = GetThreadId() ; idxInteractionBlock < inNbInteractionBlocks ; idxInteractionBlock += GetNbThreads()){
        const auto interaction = inIndexes[inInteractionBlockIdxs[inInteractionBlocksOffset[idxInteractionBlock]]];
        auto& targetCell = inCellGroup.getCellLocal(interaction.globalTargetPos);

        const long int nbNeighbors = inInteractionBlocksOffset[idxInteractionBlock+1]-inInteractionBlocksOffset[idxInteractionBlock];

        for(long int idxNeigh = 0 ; idxNeigh < nbNeighbors ; ++idxNeigh){
            const long int idxInteraction = inInteractionBlockIdxs[inInteractionBlocksOffset[idxInteractionBlock] + idxNeigh];
            assert(inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget)
                   && *inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget) == inIndexes[idxInteraction].globalTargetPos);

            assert(idxNeigh < spaceSystem.getNbInteractionsPerCell());
            neighbors.emplace_back(inOtherCellGroup.getCellMultipole(inFoundSrcIdxs[idxInteraction]));
            positionsOfNeighbors[idxNeigh] = inIndexes[idxInteraction].arrayIndexSrc;
        }

        inKernel.M2LCuda(inCellGroup.getCellSymbData(interaction.globalTargetPos),
                         inLevel,
                         TbfUtils::make_const(neighbors),
                         positionsOfNeighbors,
                         nbNeighbors,
                         targetCell);
        neighbors.clear();
    }
}



template <class SpaceIndexType, class KernelClass, class CellGroupClass>
__global__ void L2L_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass inKernel,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCellsLower,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCellsUpper,
                         const long int inIdxFirstParent, const long int inIdxLimitParent,
                         const long int* interactionOffset) {
    const CellGroupClass inLowerGroup(ptrsAndSizeCellsLower);
    CellGroupClass inUpperGroup(ptrsAndSizeCellsUpper);

    using CellLocalType = typename std::remove_reference<decltype(inLowerGroup.getCellLocal(0))>::type;
    std::vector<std::reference_wrapper<const CellLocalType>> children;
    long int positionsOfChildren[spaceSystem.getNbChildrenPerCell()];

    for(long int idxParent = inIdxFirstParent+GetThreadId() ;
         idxParent < inIdxLimitParent ; idxParent += GetNbThreads() ){
        const long int nbChildren = interactionOffset[idxParent-inIdxFirstParent+1]-interactionOffset[idxParent-inIdxFirstParent];
        assert(nbChildren >= 1);

        for(long int idxChild = 0 ; idxChild < nbChildren ; ++idxChild){
            children.emplace_back(inLowerGroup.getCellLocal(idxChild));
            positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(inLowerGroup.getCellSpacialIndex(idxChild));
        }

        inKernel.L2LCuda(inUpperGroup.getCellSymbData(idxParent),
                         inLevel, children, inUpperGroup.getCellLocal(idxParent),
                         positionsOfChildren, nbChildren);

        children.clear();
    }
}

template <class KernelClass, class LeafGroupClass, class ParticleGroupClass>
__global__ void L2P_core(KernelClass inKernel,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCells,
                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles) {
    const LeafGroupClass inLeafGroup(ptrsAndSizeCells);
    ParticleGroupClass inParticleGroup(ptrsAndSizeParticles);

    assert(inParticleGroup.getNbLeaves() == inLeafGroup.getNbCells());
    for(long int idxLeaf = GetBlockId() ; idxLeaf < inParticleGroup.getNbLeaves() ; idxLeaf += GetNbBlocks()){
        assert(inParticleGroup.getLeafSpacialIndex(idxLeaf) == inLeafGroup.getCellSpacialIndex(idxLeaf));
        const auto& particlesData = TbfUtils::make_const(inParticleGroup).getParticleData(idxLeaf);
        auto&& particlesRhs = inParticleGroup.getParticleRhs(idxLeaf);
        inKernel.L2PCuda(inLeafGroup.getCellSymbData(idxLeaf), inLeafGroup.getCellLocal(idxLeaf),
                     inParticleGroup.getParticleIndexes(idxLeaf),
                     particlesData, particlesRhs,
                     inParticleGroup.getNbParticlesInLeaf(idxLeaf));
    }
}


template <class KernelClass, class ParticleGroupClass, class IndexClass>
__global__ void P2PInGroup_core(KernelClass inKernel,
                                std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles,
                                const IndexClass* inIndexes,
                                const long int* intervalSizes, const std::pair<long int,long int>* inBlockIdxs,
                                const long int inNbBlocks) {
    ParticleGroupClass inParticleGroup(ptrsAndSizeParticles);

    for(long int idxBlock = GetBlockId() ; idxBlock < inNbBlocks ; idxBlock += GetNbBlocks()){
        for(long int idxInteractions = intervalSizes[idxBlock] ; idxInteractions < intervalSizes[idxBlock+1] ; ++idxInteractions){
            const auto interaction = inIndexes[inBlockIdxs[idxInteractions].first];

            auto foundSrc = inBlockIdxs[idxInteractions].second;

            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

            assert(inParticleGroup.getLeafSymbData(foundSrc).spaceIndex == interaction.indexSrc);
            assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

            const auto& srcData = TbfUtils::make_const(inParticleGroup).getParticleData(foundSrc);
            auto&& targetRhs = inParticleGroup.getParticleRhs(interaction.globalTargetPos);
            auto&& srcRhs = inParticleGroup.getParticleRhs(foundSrc);
            const auto& targetData = TbfUtils::make_const(inParticleGroup).getParticleData(interaction.globalTargetPos);

            inKernel.P2PCuda(inParticleGroup.getLeafSymbData(foundSrc),
                         inParticleGroup.getParticleIndexes(foundSrc),
                         srcData, srcRhs,
                         inParticleGroup.getNbParticlesInLeaf(foundSrc),
                         inParticleGroup.getLeafSymbData(interaction.globalTargetPos),
                         inParticleGroup.getParticleIndexes(interaction.globalTargetPos), targetData,
                         targetRhs, inParticleGroup.getNbParticlesInLeaf(interaction.globalTargetPos),
                         interaction.arrayIndexSrc);
        }
    }
}

template <class KernelClass, class ParticleGroupClassTarget, class ParticleGroupClassSource, class IndexClass>
__global__ void P2PBetweenGroupsTsm_core(KernelClass inKernel,
                                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles,
                                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeOthers,
                                         const IndexClass* inIndexes,
                                         const long int* intervalSizes, const std::pair<long int,long int>* inBlockIdxs,
                                         const long int inNbBlocks) {
    ParticleGroupClassTarget inParticleGroup(ptrsAndSizeParticles);
    ParticleGroupClassSource inOtherParticleGroup(ptrsAndSizeOthers);

    for(long int idxBlock = GetBlockId() ; idxBlock < inNbBlocks ; idxBlock += GetNbBlocks()){
        for(long int idxInteractions = intervalSizes[idxBlock] ; idxInteractions < intervalSizes[idxBlock+1] ; ++idxInteractions){
            const auto interaction = inIndexes[inBlockIdxs[idxInteractions].first];

            auto foundSrc = inBlockIdxs[idxInteractions].second;

            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

            assert(inOtherParticleGroup.getLeafSymbData(foundSrc).spaceIndex == interaction.indexSrc);
            assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

            const auto& srcData = TbfUtils::make_const(inOtherParticleGroup).getParticleData(foundSrc);
            auto&& targetRhs = inParticleGroup.getParticleRhs(interaction.globalTargetPos);
            const auto& targetData = TbfUtils::make_const(inParticleGroup).getParticleData(interaction.globalTargetPos);

            inKernel.P2PTsmCuda(inOtherParticleGroup.getLeafSymbData(foundSrc),
                            inOtherParticleGroup.getParticleIndexes(foundSrc),
                            srcData,
                            inOtherParticleGroup.getNbParticlesInLeaf(foundSrc),
                            inParticleGroup.getLeafSymbData(interaction.globalTargetPos),
                            inParticleGroup.getParticleIndexes(interaction.globalTargetPos), targetData,
                            targetRhs, inParticleGroup.getNbParticlesInLeaf(interaction.globalTargetPos),
                            interaction.arrayIndexSrc);
        }
    }
}



template <class KernelClass, class ParticleGroupClass, class IndexClass>
__global__ void P2PBetweenGroups_core(KernelClass inKernel,
                                      std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles,
                                      std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeOthers,
                                      const IndexClass* inIndexes,
                                      const long int* intervalSizes,
                                      const std::pair<long int,long int>* inBlockIdxs,
                                      const long int inNbBlocks) {
//    ParticleGroupClass inParticleGroup(ptrsAndSizeParticles);
//    ParticleGroupClass inOtherParticleGroup(ptrsAndSizeOthers);

//    for(long int idxBlock = GetBlockId() ; idxBlock < inNbBlocks ; idxBlock += GetNbBlocks()){
//        for(long int idxInteractions = intervalSizes[idxBlock] ; idxInteractions < intervalSizes[idxBlock+1] ; ++idxInteractions){
//            const auto interaction = inIndexes[inBlockIdxs[idxInteractions].first];

//            auto foundSrc = inBlockIdxs[idxInteractions].second;
//            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
//                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

//            assert(inOtherParticleGroup.getLeafSymbData(foundSrc).spaceIndex == interaction.indexSrc);
//            assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

//            const auto& srcData = TbfUtils::make_const(inOtherParticleGroup).getParticleData(foundSrc);
//            auto&& srcRhs = inOtherParticleGroup.getParticleRhs(foundSrc);
//            auto&& targetRhs = inParticleGroup.getParticleRhs(interaction.globalTargetPos);
//            const auto& targetData = TbfUtils::make_const(inParticleGroup).getParticleData(interaction.globalTargetPos);

//            inKernel.P2PCuda(inOtherParticleGroup.getLeafSymbData(foundSrc),
//                         inOtherParticleGroup.getParticleIndexes(foundSrc),
//                         srcData, srcRhs,
//                         inOtherParticleGroup.getNbParticlesInLeaf(foundSrc),
//                         inParticleGroup.getLeafSymbData(interaction.globalTargetPos),
//                         inParticleGroup.getParticleIndexes(interaction.globalTargetPos), targetData,
//                         targetRhs, inParticleGroup.getNbParticlesInLeaf(interaction.globalTargetPos),
//                         interaction.arrayIndexSrc);
//        }
//    }
}


template <class KernelClass, class ParticleGroupClass>
__global__ void P2PInner_core(KernelClass inKernel, std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSize) {
    ParticleGroupClass particleGroup(ptrsAndSize);
    for(long int idxLeaf = GetBlockId() ; idxLeaf < static_cast<long int>(particleGroup.getNbLeaves()) ; idxLeaf += GetNbBlocks()){
        const auto& particlesData = TbfUtils::make_const(particleGroup).getParticleData(idxLeaf);
        auto&& particlesRhs = particleGroup.getParticleRhs(idxLeaf);
        inKernel.P2PInnerCuda(particleGroup.getLeafSymbData(idxLeaf),
                          particleGroup.getParticleIndexes(idxLeaf),
                          particlesData, particlesRhs, particleGroup.getNbParticlesInLeaf(idxLeaf));
    }
}

}

template <class ObjectType>
class DeviceUniquePtr{
    ObjectType* data;
    size_t nbElements;
    cudaStream_t cuStream;

public:
    DeviceUniquePtr(const std::vector<ObjectType>& inData, cudaStream_t inCuStream)
        : data(nullptr), nbElements(inData.size()), cuStream(inCuStream) {
        cudaError_t err = cudaMallocAsync(&data, nbElements * sizeof(ObjectType), cuStream);
        if (err != cudaSuccess) {
            throw std::runtime_error("Error allocating CUDA memory");
        }
        err = cudaMemcpyAsync(data, inData.data(), nbElements * sizeof(ObjectType), cudaMemcpyHostToDevice, cuStream);
        if (err != cudaSuccess) {
            cudaFree(data);
            throw std::runtime_error("Error copying data to CUDA device");
        }
    }

    ~DeviceUniquePtr() {
        if (data) {
            cudaFreeAsync(data, cuStream);
        }
    }

    // Move constructor
    DeviceUniquePtr(DeviceUniquePtr&& other) noexcept
        : data(other.data), nbElements(other.nbElements), cuStream(other.cuStream) {
        other.data = nullptr;
        other.nbElements = 0;
    }

    // Move assignment operator
    DeviceUniquePtr& operator=(DeviceUniquePtr&& other) noexcept {
        if (this != &other) {
            if (data) {
                cudaFreeAsync(data, cuStream);
            }

            data = other.data;
            nbElements = other.nbElements;
            cuStream = other.cuStream;

            other.data = nullptr;
            other.nbElements = 0;
        }
        return *this;
    }

    // Delete copy constructor and copy assignment operator
    DeviceUniquePtr(const DeviceUniquePtr&) = delete;
    DeviceUniquePtr& operator=(const DeviceUniquePtr&) = delete;

    // Accessor methods
    ObjectType* device_ptr() const { return data; }
    size_t length() const { return nbElements; }
    size_t size() const { return nbElements; }

    void assign(const ObjectType* inData){
        if (data == nullptr) {
            throw std::runtime_error("Error copying on empty memory block");
        }
        cudaError_t err = cudaMemcpyAsync(data, inData, nbElements * sizeof(ObjectType), cudaMemcpyHostToDevice, cuStream);
        if (err != cudaSuccess) {
            cudaFreeAsync(data, cuStream);
            throw std::runtime_error("Error copying data to CUDA device");
        }
    }

    void copyBack(ObjectType* inData){
        if (data == nullptr) {
            throw std::runtime_error("Error copying on empty memory block");
        }
        cudaError_t err = cudaMemcpyAsync(inData, data, nbElements * sizeof(ObjectType), cudaMemcpyDeviceToHost, cuStream);
        if (err != cudaSuccess) {
            cudaFreeAsync(data, cuStream);
            throw std::runtime_error("Error copying data to CUDA device");
        }
    }
};

template <class ContainerClass>
auto MakeDeviceUniquePtr(const ContainerClass& inContainer, cudaStream_t cuStream){
    using ElementType = std::decay_t<decltype(inContainer[0])>;
    return DeviceUniquePtr<ElementType>(inContainer, cuStream);
}



template <class SpaceIndexType>
class TbfGroupKernelInterfaceCuda{
    const SpaceIndexType spaceSystem;

public:
    TbfGroupKernelInterfaceCuda(SpaceIndexType inSpaceIndex) : spaceSystem(std::move(inSpaceIndex)){}

    template <class KernelClass, class ParticleGroupClass, class LeafGroupClass>
    void P2M(cudaStream_t currentStream,
             KernelClass& inKernel, const ParticleGroupClass& inParticleGroup,
             const LeafGroupClass& inLeafGroup, const ParticleGroupClass& inParticleGroupCuda,
                                    LeafGroupClass& inLeafGroupCuda) {
        TbfGroupKernelInterfaceCuda_core::P2M_core<KernelClass, ParticleGroupClass, LeafGroupClass><<<1,1,0,currentStream>>>(inKernel,
                                                                            inParticleGroupCuda.getDataPtrsAndSizes(),
                                                                            inLeafGroupCuda.getDataPtrsAndSizes());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }

    template <class KernelClass, class CellGroupClass>
    void M2M(cudaStream_t currentStream,
             const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
             const CellGroupClass& inUpperGroup, const CellGroupClass& inLowerGroupCuda,
             CellGroupClass& inUpperGroupCuda) const {
        const auto startingIndex = std::max(spaceSystem.getParentIndex(inLowerGroup.getStartingSpacialIndex()),
                                            inUpperGroup.getStartingSpacialIndex());

        auto foundParent = inUpperGroup.getElementFromSpacialIndex(startingIndex);
        auto foundChild = inLowerGroup.getElementFromParentIndex(spaceSystem, startingIndex);

        assert(foundParent);
        assert(foundChild);

        const long int idxFirstParent = (*foundParent);
        long int idxParent = idxFirstParent;
        const long int idxFirstChidl = (*foundChild);
        long int idxChild = idxFirstChidl;
        long int nbChildren = 0;

        std::vector<long int> interactionOffset(inUpperGroup.getNbCells()+1-idxFirstParent);

        while(idxParent != inUpperGroup.getNbCells()
              && idxChild != inLowerGroup.getNbCells()){
            assert(spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

            assert(nbChildren < spaceSystem.getNbChildrenPerCell());
            nbChildren += 1;

            idxChild += 1;
            if(idxChild != inLowerGroup.getNbCells()
                    && spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) != inUpperGroup.getCellSpacialIndex(idxParent)){

                interactionOffset[idxParent-idxFirstParent+1] = interactionOffset[idxParent-idxFirstParent] + nbChildren;
                assert(interactionOffset[idxParent-idxFirstParent+1] == idxChild);

                idxParent += 1;
                assert(idxParent == inUpperGroup.getNbCells()
                        || spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

                nbChildren = 0;
            }
        }

        if(nbChildren){
            interactionOffset[idxParent-idxFirstParent+1] = interactionOffset[idxParent-idxFirstParent] + nbChildren;
            assert(interactionOffset[idxParent-idxFirstParent+1] == idxChild);
        }

        TbfGroupKernelInterfaceCuda_core::M2M_core<SpaceIndexType, KernelClass, CellGroupClass><<<1,1,0,currentStream>>>(spaceSystem, inLevel, inKernel,
                                                                               inLowerGroupCuda.getDataPtrsAndSizes(),
                                                                               inUpperGroupCuda.getDataPtrsAndSizes(),
                           idxFirstParent, idxParent, interactionOffset.data());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }


    template <class KernelClass, class CellGroupClass, class IndexClass>
    void M2LInGroup(cudaStream_t currentStream,
                    const long int inLevel, KernelClass& inKernel, const CellGroupClass& inCellGroup,
                    CellGroupClass& inCellGroupCuda, const IndexClass& inIndexes) const {
        using CellMultipoleType = typename std::remove_reference<decltype(inCellGroup.getCellMultipole(0))>::type;
        //using CellLocalType = typename std::remove_reference<decltype(inCellGroup.getCellLocal(0))>::type;

        std::vector<long int> interactionBlocks;
        interactionBlocks.emplace_back(0);

        long int idxInteraction = 0;

        while(idxInteraction < static_cast<long int>(inIndexes.size())){
            const auto interaction = inIndexes[idxInteraction];

            long int nbNeighbors = 0;

            do{
                auto foundSrc = inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexSrc);
                assert(foundSrc);
                assert(inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget)
                       && *inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget) == inIndexes[idxInteraction].globalTargetPos);

                assert(nbNeighbors < spaceSystem.getNbInteractionsPerCell());
                nbNeighbors += 1;

                idxInteraction += 1;
            } while(idxInteraction < static_cast<long int>(inIndexes.size())
                    && interaction.indexTarget == inIndexes[idxInteraction].indexTarget);

            assert(idxInteraction == nbNeighbors + interactionBlocks.size());
            interactionBlocks.emplace_back(idxInteraction);
        }

        TbfGroupKernelInterfaceCuda_core::M2LInGroup_core<SpaceIndexType, KernelClass, CellGroupClass, IndexClass><<<1,1,0,currentStream>>>(spaceSystem, inLevel,
                                                                                      inKernel,
                                                                                      inCellGroupCuda.getDataPtrsAndSizes(), inIndexes.data(),
                                  static_cast<long int>(interactionBlocks.size()), interactionBlocks.data());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }


    template <class KernelClass, class CellGroupClassTarget, class CellGroupClassSource, class IndexClass>
    void M2LBetweenGroups(cudaStream_t currentStream,
                          const long int inLevel, KernelClass& inKernel, const CellGroupClassTarget& inCellGroup,
                          const CellGroupClassSource& inOtherCellGroup, CellGroupClassTarget& inCellGroupCuda,
                          const CellGroupClassSource& inOtherCellGroupCuda, const IndexClass& inIndexes) const {
        using CellMultipoleType = typename std::remove_reference<decltype(inOtherCellGroup.getCellMultipole(0))>::type;
        //using CellLocalType = typename std::remove_reference<decltype(inCellGroup.getCellLocal(0))>::type;

        std::vector<long int> offsetInteractionIdxs;
        offsetInteractionIdxs.emplace_back(0);
        std::vector<long int> interactionIdxs;
        std::vector<long int> foundSrcIdxs;

        long int idxInteraction = 0;

        while(idxInteraction < static_cast<long int>(inIndexes.size())){
            const auto interaction = inIndexes[idxInteraction];

            long int nbNeighbors = 0;

            do{
                auto foundSrc = inOtherCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexSrc);
                if(foundSrc != -1){
                    assert(inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget)
                          && *inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget) == inIndexes[idxInteraction].globalTargetPos);

                    assert(nbNeighbors < spaceSystem.getNbInteractionsPerCell());
                    nbNeighbors += 1;

                    interactionIdxs.emplace_back(idxInteraction);
                    foundSrcIdxs.emplace_back(foundSrc);
                }

                idxInteraction += 1;
            } while(idxInteraction < static_cast<long int>(inIndexes.size())
                    && interaction.indexTarget == inIndexes[idxInteraction].indexTarget);

            if(nbNeighbors){
                offsetInteractionIdxs.emplace_back(interactionIdxs.size());
            }
        }

        TbfGroupKernelInterfaceCuda_core::M2LBetweenGroups_core<SpaceIndexType, KernelClass, CellGroupClassTarget, CellGroupClassSource, IndexClass><<<1,1,0,currentStream>>>(spaceSystem, inLevel,
                                                                                         inKernel,
                                                                                         inCellGroupCuda.getDataPtrsAndSizes(),
                                                                                         inOtherCellGroup.getDataPtrsAndSizes(),
                                                                                         inIndexes.data(),
                              offsetInteractionIdxs.size(), offsetInteractionIdxs.data(),
                              interactionIdxs.data(), foundSrcIdxs.data());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }


    template <class KernelClass, class CellGroupClass>
    void L2L(cudaStream_t currentStream,
             const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
             const CellGroupClass& inUpperGroup, const CellGroupClass& inLowerGroupCuda,
             CellGroupClass& inUpperGroupCuda) const {
        const auto startingIndex = std::max(spaceSystem.getParentIndex(inLowerGroup.getStartingSpacialIndex()),
                                            inUpperGroup.getStartingSpacialIndex());

        auto foundParent = inUpperGroup.getElementFromSpacialIndex(startingIndex);
        auto foundChild = inLowerGroup.getElementFromParentIndex(spaceSystem, startingIndex);

        assert(foundParent);
        assert(foundChild);

        const long int idxFirstParent = (*foundParent);
        long int idxParent = idxFirstParent;
        const long int idxFirstChidl = (*foundChild);
        long int idxChild = idxFirstChidl;
        long int nbChildren = 0;

        std::vector<long int> interactionOffset(inUpperGroup.getNbCells()+1-idxFirstParent);

        while(idxParent != inUpperGroup.getNbCells()
               && idxChild != inLowerGroup.getNbCells()){
            assert(spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

            assert(nbChildren < spaceSystem.getNbChildrenPerCell());
            nbChildren += 1;

            idxChild += 1;
            if(idxChild != inLowerGroup.getNbCells()
                && spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) != inUpperGroup.getCellSpacialIndex(idxParent)){

                interactionOffset[idxParent-idxFirstParent+1] = interactionOffset[idxParent-idxFirstParent] + nbChildren;
                assert(interactionOffset[idxParent-idxFirstParent+1] == idxChild);

                idxParent += 1;
                assert(idxParent == inUpperGroup.getNbCells()
                       || spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

                nbChildren = 0;
            }
        }

        if(nbChildren){
            interactionOffset[idxParent-idxFirstParent+1] = interactionOffset[idxParent-idxFirstParent] + nbChildren;
            assert(interactionOffset[idxParent-idxFirstParent+1] == idxChild);
        }

        TbfGroupKernelInterfaceCuda_core::L2L_core<SpaceIndexType, KernelClass, CellGroupClass><<<1,1,0,currentStream>>>(spaceSystem, inLevel,
                                                                               inKernel,
                                                                               inLowerGroupCuda.getDataPtrsAndSizes(),
                                                                               inUpperGroupCuda.getDataPtrsAndSizes(),
                           idxFirstParent, idxParent, interactionOffset.data());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }

    template <class KernelClass, class LeafGroupClass, class ParticleGroupClass>
    void L2P(cudaStream_t currentStream,
             KernelClass& inKernel, const LeafGroupClass& /*inLeafGroup*/,
             const ParticleGroupClass& /*inParticleGroup*/, const LeafGroupClass& inLeafGroupCuda,
             ParticleGroupClass& inParticleGroupCuda) const {
        TbfGroupKernelInterfaceCuda_core::L2P_core<KernelClass,LeafGroupClass,ParticleGroupClass><<<1,1,0,currentStream>>>(inKernel,
                                                                            inLeafGroupCuda.getDataPtrsAndSizes(),
                                                                            inParticleGroupCuda.getDataPtrsAndSizes());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }


    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PInGroup(cudaStream_t currentStream,
                    KernelClass& inKernel,
                    const ParticleGroupClass& inParticleGroup,
                    ParticleGroupClass& inParticleGroupCuda,
                    const IndexClass& inIndexes) const {
        std::vector<std::pair<long int,long int>> interactionBlocks[spaceSystem.getNbNeighborsPerLeaf()];
        std::vector<long int> interactionBlockIntervals[spaceSystem.getNbNeighborsPerLeaf()];

        for(long int idxColor = 0 ; idxColor < spaceSystem.getNbNeighborsPerLeaf() ; ++idxColor){
            interactionBlockIntervals[idxColor].emplace_back(0);
        }

        typename SpaceIndexType::IndexType previousTarget = -1;

        for(long int idxInteraction = 0 ; idxInteraction < static_cast<long int>(inIndexes.size()) ; ++idxInteraction){
            const auto interaction = inIndexes[idxInteraction];

            auto foundSrc = inParticleGroup.getElementFromSpacialIndex(interaction.indexSrc);
            assert(foundSrc);
            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

            assert(inParticleGroup.getLeafSymbData(*foundSrc).spaceIndex == interaction.indexSrc);
            assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

            const auto colorTgt = spaceSystem.getColorsIdxAtLeafLevel(interaction.indexTarget);
            if(previousTarget != interaction.indexTarget){
                interactionBlockIntervals[colorTgt].emplace_back(interactionBlocks[colorTgt].size());
                previousTarget = interaction.indexTarget;
            }

            interactionBlockIntervals[colorTgt].back() += 1;
            interactionBlocks[colorTgt].emplace_back(std::make_pair(idxInteraction, *foundSrc));
        }

        {
            auto inIndexesCuda = MakeDeviceUniquePtr(inIndexes,currentStream);
            for(long int idxColor = 0 ; idxColor < spaceSystem.getNbNeighborsPerLeaf() ; ++idxColor){
                auto interactionBlockIntervalsCuda = MakeDeviceUniquePtr(interactionBlockIntervals[idxColor],currentStream);
                auto interactionBlocksCuda = MakeDeviceUniquePtr(interactionBlocks[idxColor],currentStream);
                TbfGroupKernelInterfaceCuda_core::P2PInGroup_core<KernelClass, ParticleGroupClass><<<1,1,0,currentStream>>>(
                        inKernel,
                        inParticleGroupCuda.getDataPtrsAndSizes(),
                        inIndexesCuda.device_ptr(),
                        interactionBlockIntervalsCuda.device_ptr(),
                        interactionBlocksCuda.device_ptr(),
                        interactionBlockIntervalsCuda.size()-1);
                [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
                CUDA_ASSERT(cudaRes);
            }
        }
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }

    template <class KernelClass, class ParticleGroupClass>
    void P2PInner(cudaStream_t currentStream,
                  KernelClass& inKernel, const ParticleGroupClass& /*inParticleGroup*/,
                  ParticleGroupClass& inParticleGroupCuda) const {
        TbfGroupKernelInterfaceCuda_core::P2PInner_core<KernelClass,ParticleGroupClass><<<1,1,0,currentStream>>>(inKernel, inParticleGroupCuda.getDataPtrsAndSizes());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }

    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PBetweenGroups(cudaStream_t currentStream,
                          KernelClass& inKernel, const ParticleGroupClass& inParticleGroup,
                          const ParticleGroupClass& inOtherParticleGroup,
                          ParticleGroupClass& inParticleGroupCuda,
                          ParticleGroupClass& inOtherParticleGroupCuda, const IndexClass& inIndexes) const {
        std::vector<std::pair<long int,long int>> interactionBlocks[spaceSystem.getNbNeighborsPerLeaf()];
        std::vector<long int> interactionBlockIntervals[spaceSystem.getNbNeighborsPerLeaf()];

        for(long int idxColor = 0 ; idxColor < spaceSystem.getNbNeighborsPerLeaf() ; ++idxColor){
            interactionBlockIntervals[idxColor].emplace_back(0);
        }

        typename SpaceIndexType::IndexType previousTarget = -1;

        for(long int idxInteraction = 0 ; idxInteraction < static_cast<long int>(inIndexes.size()) ; ++idxInteraction){
            const auto interaction = inIndexes[idxInteraction];

            auto foundSrc = inOtherParticleGroup.getElementFromSpacialIndex(interaction.indexSrc);
            if(foundSrc){
                assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                       && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

                assert(inOtherParticleGroup.getLeafSymbData(*foundSrc).spaceIndex == interaction.indexSrc);
                assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

                const auto colorTgt = spaceSystem.getColorsIdxAtLeafLevel(interaction.indexTarget);
                if(previousTarget != interaction.indexTarget){
                    interactionBlockIntervals[colorTgt].emplace_back(interactionBlocks[colorTgt].size());
                    previousTarget = interaction.indexTarget;
                }

                interactionBlockIntervals[colorTgt].back() += 1;
                interactionBlocks[colorTgt].emplace_back(std::make_pair(idxInteraction, *foundSrc));
            }
        }

        {
            auto inIndexesCuda = MakeDeviceUniquePtr(inIndexes,currentStream);

            for(long int idxColor = 0 ; idxColor < spaceSystem.getNbNeighborsPerLeaf() ; ++idxColor){
                {
                    auto interactionBlockIntervalsCuda = MakeDeviceUniquePtr(interactionBlockIntervals[idxColor],currentStream);
                    auto interactionBlocksCuda = MakeDeviceUniquePtr(interactionBlocks[idxColor],currentStream);
                    TbfGroupKernelInterfaceCuda_core::P2PBetweenGroups_core<KernelClass, ParticleGroupClass><<<1,1,0,currentStream>>>(
                                                                                      inKernel,
                                                                                      inParticleGroupCuda.getDataPtrsAndSizes(),
                                                                                      inOtherParticleGroupCuda.getDataPtrsAndSizes(),
                                                                                      inIndexesCuda.device_ptr(),
                                                                                      interactionBlockIntervalsCuda.device_ptr(),
                                                                                      interactionBlocksCuda.device_ptr(),
                                                                                      interactionBlockIntervalsCuda.size()-1);
                }
                [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
                 CUDA_ASSERT(cudaRes);
            }
        }
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }



    template <class KernelClass, class ParticleGroupClassTarget, class ParticleGroupClassSource, class IndexClass>
    void P2PBetweenGroupsTsm(cudaStream_t currentStream,
                             KernelClass& inKernel, const ParticleGroupClassTarget& inParticleGroup,
                             const ParticleGroupClassSource& inOtherParticleGroup, ParticleGroupClassTarget& inParticleGroupCuda,
                             ParticleGroupClassSource& inOtherParticleGroupCuda, const IndexClass& inIndexes) const {

        std::vector<std::pair<long int,long int>> interactionBlocks;
        std::vector<long int> interactionBlockIntervals;
        interactionBlockIntervals.emplace_back(0);

        typename SpaceIndexType::IndexType previousTarget = -1;

        for(long int idxInteraction = 0 ; idxInteraction < static_cast<long int>(inIndexes.size()) ; ++idxInteraction){
            const auto interaction = inIndexes[idxInteraction];

            auto foundSrc = inOtherParticleGroup.getElementFromSpacialIndex(interaction.indexSrc);
            if(foundSrc){
                assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                       && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

                assert(inOtherParticleGroup.getLeafSymbData(*foundSrc).spaceIndex == interaction.indexSrc);
                assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

                if(previousTarget != interaction.indexTarget){
                    interactionBlockIntervals.emplace_back(interactionBlocks.size());
                    previousTarget = interaction.indexTarget;
                }

                interactionBlockIntervals.back() += 1;
                interactionBlocks.emplace_back(std::make_pair(idxInteraction, *foundSrc));
            }
        }

        {
            auto inIndexesCuda = MakeDeviceUniquePtr(inIndexes,currentStream);
            auto interactionBlocksCuda = MakeDeviceUniquePtr(interactionBlocks,currentStream);
            auto interactionBlockIntervalsCuda = MakeDeviceUniquePtr(interactionBlockIntervals,currentStream);

            TbfGroupKernelInterfaceCuda_core::P2PBetweenGroupsTsm_core<KernelClass, ParticleGroupClassTarget, ParticleGroupClassSource, IndexClass><<<1,1,0,currentStream>>>(inKernel, inParticleGroupCuda.getDataPtrsAndSizes(),
                                                                                                   inOtherParticleGroupCuda.getDataPtrsAndSizes(),
                                                                                 inIndexesCuda.device_ptr(), interactionBlockIntervalsCuda.device_ptr(),
                                                                                 interactionBlocksCuda.device_ptr(),  interactionBlockIntervalsCuda.size()-1);
        }
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }
};

#endif
