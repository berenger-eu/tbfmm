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
__global__ void P2M_core(typename KernelClass::CudaKernelData inKernel,
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
        KernelClass::CudaP2M(inKernel, symbData, inParticleGroup.getParticleIndexes(idxLeaf), particlesData, inParticleGroup.getNbParticlesInLeaf(idxLeaf),
                         leafData);
    }
}


template <class SpaceIndexType, class KernelClass, class CellGroupClass>
__global__ void M2M_core(const SpaceIndexType& spaceSystem, const long int inLevel,
                         typename KernelClass::CudaKernelData inKernel,
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

        KernelClass::M2MCuda(inKernel, inUpperGroup.getCellSymbData(idxParent),
                         inLevel, TbfUtils::make_const(children), inUpperGroup.getCellMultipole(idxParent),
                         positionsOfChildren, nbChildren);

        children.clear();
    }
}


template <class SpaceIndexType, class KernelClass, class CellGroupClass, class IndexClass>
__global__ void M2LInGroup_core(const SpaceIndexType& spaceSystem, const long int inLevel,
                                typename KernelClass::CudaKernelData inKernel,
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

        KernelClass::M2LCuda(inKernel, inCellGroup.getCellSymbData(interaction.globalTargetPos),
                         inLevel,
                         TbfUtils::make_const(neighbors),
                         positionsOfNeighbors,
                         nbNeighbors,
                         targetCell);
        neighbors.clear();
    }
}

template <class SpaceIndexType, class KernelClass, class CellGroupClassTarget, class CellGroupClassSource, class IndexClass>
__global__ void M2LBetweenGroups_core(const SpaceIndexType& spaceSystem, const long int inLevel,
                                      typename KernelClass::CudaKernelData inKernel,
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

        KernelClass::M2LCuda(inKernel, inCellGroup.getCellSymbData(interaction.globalTargetPos),
                         inLevel,
                         TbfUtils::make_const(neighbors),
                         positionsOfNeighbors,
                         nbNeighbors,
                         targetCell);
        neighbors.clear();
    }
}



template <class SpaceIndexType, class KernelClass, class CellGroupClass>
__global__ void L2L_core(const SpaceIndexType& spaceSystem, const long int inLevel,
                         typename KernelClass::CudaKernelData inKernel,
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

        KernelClass::L2LCuda(inKernel, inUpperGroup.getCellSymbData(idxParent),
                         inLevel, children, inUpperGroup.getCellLocal(idxParent),
                         positionsOfChildren, nbChildren);

        children.clear();
    }
}

template <class KernelClass, class LeafGroupClass, class ParticleGroupClass>
__global__ void L2P_core(typename KernelClass::CudaKernelData inKernel,
                         std::array<std::pair<unsigned char*,size_t>,3> ptrsAndSizeCells,
                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles) {
    const LeafGroupClass inLeafGroup(ptrsAndSizeCells);
    ParticleGroupClass inParticleGroup(ptrsAndSizeParticles);

    assert(inParticleGroup.getNbLeaves() == inLeafGroup.getNbCells());
    for(long int idxLeaf = GetBlockId() ; idxLeaf < inParticleGroup.getNbLeaves() ; idxLeaf += GetNbBlocks()){
        assert(inParticleGroup.getLeafSpacialIndex(idxLeaf) == inLeafGroup.getCellSpacialIndex(idxLeaf));
        const auto& particlesData = TbfUtils::make_const(inParticleGroup).getParticleData(idxLeaf);
        auto&& particlesRhs = inParticleGroup.getParticleRhs(idxLeaf);
        KernelClass::L2PCuda(inKernel, inLeafGroup.getCellSymbData(idxLeaf), inLeafGroup.getCellLocal(idxLeaf),
                     inParticleGroup.getParticleIndexes(idxLeaf),
                     particlesData, particlesRhs,
                     inParticleGroup.getNbParticlesInLeaf(idxLeaf));
    }
}


template <class KernelClass, class ParticleGroupClassTarget, class ParticleGroupClassSource, class IndexClass>
__global__ void P2PBetweenGroupsTsm_core(typename KernelClass::CudaKernelData inKernel,
                                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeOthers,
                                         std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSizeParticles,
                                         const IndexClass* inIndexes,
                                         const long int* intervalSizes,
                                         const long int inNbBlocks) {
    ParticleGroupClassTarget inParticleGroup(ptrsAndSizeParticles);
    ParticleGroupClassSource inOtherParticleGroup(ptrsAndSizeOthers);

    for(long int idxBlock = GetBlockId() ; idxBlock < inNbBlocks ; idxBlock += GetNbBlocks()){
        for(long int idxInteractions = intervalSizes[idxBlock] ; idxInteractions < intervalSizes[idxBlock+1] ; ++idxInteractions){
            const auto interaction = inIndexes[idxInteractions];

            auto foundSrc = interaction.globalSrcPos;

            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

            assert(inOtherParticleGroup.getLeafSymbData(foundSrc).spaceIndex == interaction.indexSrc);
            assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

            const auto& srcData = TbfUtils::make_const(inOtherParticleGroup).getParticleData(foundSrc);
            auto&& targetRhs = inParticleGroup.getParticleRhs(interaction.globalTargetPos);
            const auto& targetData = TbfUtils::make_const(inParticleGroup).getParticleData(interaction.globalTargetPos);

            KernelClass::P2PTsmCuda(inKernel,
                                    inOtherParticleGroup.getLeafSymbData(foundSrc),
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


template <class KernelClass, class ParticleGroupClass>
__global__ void P2PInner_core(typename KernelClass::CudaKernelData inKernel,
                              std::array<std::pair<unsigned char*,size_t>,2> ptrsAndSize) {
    ParticleGroupClass particleGroup(ptrsAndSize);
    for(long int idxLeaf = GetBlockId() ; idxLeaf < static_cast<long int>(particleGroup.getNbLeaves()) ; idxLeaf += GetNbBlocks()){
        const auto& particlesData = TbfUtils::make_const(particleGroup).getParticleData(idxLeaf);
        auto&& particlesRhs = particleGroup.getParticleRhs(idxLeaf);
        KernelClass::P2PInnerCuda(inKernel, particleGroup.getLeafSymbData(idxLeaf),
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

    void release(){
        if (data) {
            cudaFreeAsync(data, cuStream);
        }
    }

    ~DeviceUniquePtr() {
        release();
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
        TbfGroupKernelInterfaceCuda_core::P2M_core<KernelClass, ParticleGroupClass, LeafGroupClass><<<64,256,0,currentStream>>>(inKernel.getCudaKernelData(),
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

        TbfGroupKernelInterfaceCuda_core::M2M_core<SpaceIndexType, KernelClass, CellGroupClass><<<64,256,0,currentStream>>>(spaceSystem, inLevel, inKernel.getCudaKernelData(),
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

        TbfGroupKernelInterfaceCuda_core::M2LInGroup_core<SpaceIndexType, KernelClass, CellGroupClass, IndexClass><<<64,256,0,currentStream>>>(spaceSystem, inLevel,
                                                                                      inKernel.getCudaKernelData(),
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

        TbfGroupKernelInterfaceCuda_core::M2LBetweenGroups_core<SpaceIndexType, KernelClass, CellGroupClassTarget, CellGroupClassSource, IndexClass><<<64,256,0,currentStream>>>(spaceSystem, inLevel,
                                                                                         inKernel.getCudaKernelData(),
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

        TbfGroupKernelInterfaceCuda_core::L2L_core<SpaceIndexType, KernelClass, CellGroupClass><<<64,256,0,currentStream>>>(spaceSystem, inLevel,
                                                                               inKernel.getCudaKernelData(),
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
        TbfGroupKernelInterfaceCuda_core::L2P_core<KernelClass,LeafGroupClass,ParticleGroupClass><<<64,256,0,currentStream>>>(inKernel.getCudaKernelData(),
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
        P2PBetweenGroupsGeneric(currentStream,
                                inKernel, inParticleGroup,
                                inParticleGroup, inParticleGroupCuda,
                                inParticleGroupCuda, inIndexes,
                                true, true);
    }

    template <class KernelClass, class ParticleGroupClass>
    void P2PInner(cudaStream_t currentStream,
                  KernelClass& inKernel, const ParticleGroupClass& /*inParticleGroup*/,
                  ParticleGroupClass& inParticleGroupCuda) const {
        TbfGroupKernelInterfaceCuda_core::P2PInner_core<KernelClass,ParticleGroupClass><<<64,256,0,currentStream>>>(inKernel.getCudaKernelData(),
                                                                                                                     inParticleGroupCuda.getDataPtrsAndSizes());
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }

    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PBetweenGroups(cudaStream_t currentStream,
                          KernelClass& inKernel,
                          const ParticleGroupClass& inOtherParticleGroup,
                           const ParticleGroupClass& inParticleGroup,
                          ParticleGroupClass& inOtherParticleGroupCuda,
                          ParticleGroupClass& inParticleGroupCuda,
                          const IndexClass& inIndexes) const {
        P2PBetweenGroupsGeneric(currentStream,
                                inKernel,
                                inOtherParticleGroup,
                                inParticleGroup,
                                inOtherParticleGroupCuda,
                                inParticleGroupCuda,
                                inIndexes,
                                true, false);
    }

    template <class KernelClass, class ParticleGroupClassTarget, class ParticleGroupClassSource, class IndexClass>
    void P2PBetweenGroupsTsm(cudaStream_t currentStream,
                             KernelClass& inKernel,
                             const ParticleGroupClassSource& inOtherParticleGroup,
                             const ParticleGroupClassTarget& inParticleGroup,
                             ParticleGroupClassSource& inOtherParticleGroupCuda,
                             ParticleGroupClassTarget& inParticleGroupCuda,
                             const IndexClass& inIndexes) const {
        P2PBetweenGroupsGeneric(currentStream,
                                inKernel,
                                inOtherParticleGroup,
                                inParticleGroup,
                                inOtherParticleGroupCuda,
                                inParticleGroupCuda,
                                inIndexes,
                                false, false);
    }


    struct IndexTypeFull{
        long int globalTargetPos;
        long int indexTarget;
        long int indexSrc;
        long int globalSrcPos;
        long int arrayIndexSrc;
    };

    template <class KernelClass, class ParticleGroupClassTarget, class ParticleGroupClassSource, class IndexClass>
    void P2PBetweenGroupsGeneric(cudaStream_t currentStream,
                             KernelClass& inKernel,
                                 const ParticleGroupClassSource& inOtherParticleGroup,
                             const ParticleGroupClassTarget& inParticleGroup,
                                 ParticleGroupClassSource& inOtherParticleGroupCuda,
                             ParticleGroupClassTarget& inParticleGroupCuda,
                                 const IndexClass& inIndexes,
                                 const bool inBothDirection, const bool scrMustBeThere) const {
        std::vector<IndexTypeFull> indexesFull;
        std::vector<long int> indexesIntervals;
        indexesIntervals.push_back(0);

        typename SpaceIndexType::IndexType previousTarget = -1;

        for(long int idxInteraction = 0 ; idxInteraction < static_cast<long int>(inIndexes.size()) ; ++idxInteraction){
            const auto interaction = inIndexes[idxInteraction];

            auto foundSrc = inOtherParticleGroup.getElementFromSpacialIndex(interaction.indexSrc);
            if(scrMustBeThere || foundSrc){
                assert(foundSrc);
                assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                       && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

                assert(inOtherParticleGroup.getLeafSymbData(*foundSrc).spaceIndex == interaction.indexSrc);
                assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

                if(previousTarget != interaction.indexTarget){
                    indexesIntervals.emplace_back(indexesFull.size());
                    previousTarget = interaction.indexTarget;
                }

                indexesIntervals.back() += 1;
                indexesFull.emplace_back(IndexTypeFull{interaction.globalTargetPos,
                                                       interaction.indexTarget, interaction.indexSrc, *foundSrc,
                                                       interaction.arrayIndexSrc});
            }
        }

        if(indexesFull.empty()){
            return;
        }
        assert(indexesIntervals.size() >= 2);

        auto indexesFullCuda = MakeDeviceUniquePtr(indexesFull,currentStream);
        auto indexesIntervalsCuda = MakeDeviceUniquePtr(indexesIntervals,currentStream);
        TbfGroupKernelInterfaceCuda_core::P2PBetweenGroupsTsm_core<KernelClass, ParticleGroupClassTarget, ParticleGroupClassSource><<<64,256,0,currentStream>>>(
            inKernel.getCudaKernelData(),
            inOtherParticleGroupCuda.getDataPtrsAndSizes(),
            inParticleGroupCuda.getDataPtrsAndSizes(),
            indexesFullCuda.device_ptr(),
            indexesIntervalsCuda.device_ptr(),
            indexesIntervalsCuda.size()-1);

        if(inBothDirection){
            // Reverse indexes
            for(auto& index: indexesFull){
                std::swap(index.globalTargetPos, index.globalSrcPos);
                std::swap(index.indexTarget, index.indexSrc);
                // TODO reverse this index.arrayIndexSrc;
            }

            std::sort(indexesFull.begin(), indexesFull.end(), [](const auto& i1, const auto& i2){
                return i1.indexTarget < i2.indexTarget;
            });

            std::vector<long int> indexesIntervalsReverse;
            previousTarget = -1;
            for(long int idxInteractionRev = 0 ; idxInteractionRev < static_cast<long int>(indexesFull.size()) ; ++idxInteractionRev){
                const auto& interaction = indexesFull[idxInteractionRev];
                if(previousTarget != interaction.indexTarget){
                    indexesIntervalsReverse.emplace_back(idxInteractionRev);
                    previousTarget = interaction.indexTarget;
                }
            }
            indexesIntervalsReverse.emplace_back(static_cast<long int>(indexesFull.size()));

            auto indexesFullCudaReverse = MakeDeviceUniquePtr(indexesFull,currentStream);
            auto indexesIntervalsCudaReverse = MakeDeviceUniquePtr(indexesIntervalsReverse,currentStream);
            TbfGroupKernelInterfaceCuda_core::P2PBetweenGroupsTsm_core<KernelClass, ParticleGroupClassTarget, ParticleGroupClassSource><<<64,256,0,currentStream>>>(
                inKernel.getCudaKernelData(),
                inParticleGroupCuda.getDataPtrsAndSizes(),
                inOtherParticleGroupCuda.getDataPtrsAndSizes(),
                indexesFullCudaReverse.device_ptr(),
                indexesIntervalsCudaReverse.device_ptr(),
                indexesIntervalsCudaReverse.size()-1);

            [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
            CUDA_ASSERT(cudaRes);
        }
        else{
            [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
            CUDA_ASSERT(cudaRes);
        }

        indexesFullCuda.release();
        indexesIntervalsCuda.release();
        [[maybe_unused]] auto cudaRes = cudaStreamSynchronize(currentStream);
        CUDA_ASSERT(cudaRes);
    }
};

#endif
