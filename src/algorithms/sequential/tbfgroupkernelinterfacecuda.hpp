#ifndef TBFGROUPKERNELINTERFACECUDA_HPP
#define TBFGROUPKERNELINTERFACECUDA_HPP

#include "tbfglobal.hpp"
#include "utils/tbfutils.hpp"

#include <cassert>
#include <cuda_runtime.h>

#include <vector>
#include <utility>

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
__global__ void P2M_core(KernelClass& inKernel, const ParticleGroupClass& inParticleGroup,
                         LeafGroupClass& inLeafGroup) {
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
__global__ void M2M_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
                         CellGroupClass& inUpperGroup, const long int inIdxFirstParent, const long int inIdxLimitParent,
                         const long int* interactionOffset) {
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
__global__ void M2LInGroup_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass& inKernel, CellGroupClass& inCellGroup,
                                const IndexClass& inIndexes,
                                const long int inNbInteractionBlocks,
                                const long int* inInteractionBlocks) {
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
__global__ void M2LBetweenGroups_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass& inKernel, CellGroupClassTarget& inCellGroup,
                                      const CellGroupClassSource& inOtherCellGroup, const IndexClass& inIndexes,
                                      const long int inNbInteractionBlocks, const long int* inInteractionBlocksOffset,
                                      const long int* inInteractionBlockIdxs, const long int* inFoundSrcIdxs) {
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
__global__ void L2L_core(const SpaceIndexType& spaceSystem, const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
                         CellGroupClass& inUpperGroup, const long int inIdxFirstParent, const long int inIdxLimitParent,
                         const long int* interactionOffset) {
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
__global__ void L2P_core(KernelClass& inKernel, const LeafGroupClass& inLeafGroup,
                         ParticleGroupClass& inParticleGroup) {
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
__global__ void P2PInGroup_core(KernelClass& inKernel, ParticleGroupClass& inParticleGroup, const IndexClass& inIndexes,
                                const long int* intervalSizes, const std::pair<long int,long int>* inBlockIdxs,
                                const long int inNbBlocks) {
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
__global__ void P2PBetweenGroupsTsm_core(KernelClass& inKernel, ParticleGroupClassTarget& inParticleGroup,
                                         ParticleGroupClassSource& inOtherParticleGroup, const IndexClass& inIndexes,
                                         const long int* intervalSizes, const std::pair<long int,long int>* inBlockIdxs,
                                         const long int inNbBlocks) {
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
__global__ void P2PBetweenGroups_core(KernelClass& inKernel, ParticleGroupClass& inParticleGroup,
                                      ParticleGroupClass& inOtherParticleGroup, const IndexClass& inIndexes,
                                      const long int* intervalSizes, const std::pair<long int,long int>* inBlockIdxs,
                                      const long int inNbBlocks) {
    for(long int idxBlock = GetBlockId() ; idxBlock < inNbBlocks ; idxBlock += GetNbBlocks()){
        for(long int idxInteractions = intervalSizes[idxBlock] ; idxInteractions < intervalSizes[idxBlock+1] ; ++idxInteractions){
            const auto interaction = inIndexes[inBlockIdxs[idxInteractions].first];

            auto foundSrc = inBlockIdxs[idxInteractions].second;
            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

            assert(inOtherParticleGroup.getLeafSymbData(foundSrc).spaceIndex == interaction.indexSrc);
            assert(inParticleGroup.getLeafSymbData(interaction.globalTargetPos).spaceIndex == interaction.indexTarget);

            const auto& srcData = TbfUtils::make_const(inOtherParticleGroup).getParticleData(foundSrc);
            auto&& srcRhs = inOtherParticleGroup.getParticleRhs(foundSrc);
            auto&& targetRhs = inParticleGroup.getParticleRhs(interaction.globalTargetPos);
            const auto& targetData = TbfUtils::make_const(inParticleGroup).getParticleData(interaction.globalTargetPos);

            inKernel.P2PCuda(inOtherParticleGroup.getLeafSymbData(foundSrc),
                         inOtherParticleGroup.getParticleIndexes(foundSrc),
                         srcData, srcRhs,
                         inOtherParticleGroup.getNbParticlesInLeaf(foundSrc),
                         inParticleGroup.getLeafSymbData(interaction.globalTargetPos),
                         inParticleGroup.getParticleIndexes(interaction.globalTargetPos), targetData,
                         targetRhs, inParticleGroup.getNbParticlesInLeaf(interaction.globalTargetPos),
                         interaction.arrayIndexSrc);
        }
    }
}


template <class KernelClass, class ParticleGroupClass>
__global__ void P2PInner_core(KernelClass& inKernel, ParticleGroupClass& inParticleGroup) {
    for(long int idxLeaf = GetBlockId() ; idxLeaf < static_cast<long int>(inParticleGroup.getNbLeaves()) ; idxLeaf += GetNbBlocks()){
        const auto& particlesData = TbfUtils::make_const(inParticleGroup).getParticleData(idxLeaf);
        auto&& particlesRhs = inParticleGroup.getParticleRhs(idxLeaf);
        inKernel.P2PInnerCuda(inParticleGroup.getLeafSymbData(idxLeaf),
                          inParticleGroup.getParticleIndexes(idxLeaf),
                          particlesData, particlesRhs, inParticleGroup.getNbParticlesInLeaf(idxLeaf));
    }
}

}


template <class SpaceIndexType>
class TbfGroupKernelInterfaceCuda{
    const SpaceIndexType spaceSystem;

public:
    TbfGroupKernelInterfaceCuda(SpaceIndexType inSpaceIndex) : spaceSystem(std::move(inSpaceIndex)){}

    template <class KernelClass, class ParticleGroupClass, class LeafGroupClass>
    void P2M(KernelClass& inKernel, const ParticleGroupClass& inParticleGroup,
                                    LeafGroupClass& inLeafGroup) {
        TbfGroupKernelInterfaceCuda_core::P2M_core<<<1,1>>>(inKernel, inParticleGroup, inLeafGroup);
    }

    template <class KernelClass, class CellGroupClass>
    void M2M(const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
             CellGroupClass& inUpperGroup) const {
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

        TbfGroupKernelInterfaceCuda_core::M2M_core<<<1,1>>>(spaceSystem, inLevel, inKernel, inLowerGroup, inUpperGroup,
                           idxFirstParent, idxParent, interactionOffset.data());
    }


    template <class KernelClass, class CellGroupClass, class IndexClass>
    void M2LInGroup(const long int inLevel, KernelClass& inKernel, CellGroupClass& inCellGroup, const IndexClass& inIndexes) const {
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

        TbfGroupKernelInterfaceCuda_core::M2LInGroup_core<<<1,1>>>(spaceSystem, inLevel, inKernel, inCellGroup, inIndexes,
                                  static_cast<long int>(interactionBlocks.size()), interactionBlocks.data());
    }


    template <class KernelClass, class CellGroupClassTarget, class CellGroupClassSource, class IndexClass>
    void M2LBetweenGroups(const long int inLevel, KernelClass& inKernel, CellGroupClassTarget& inCellGroup,
                          const CellGroupClassSource& inOtherCellGroup, const IndexClass& inIndexes) const {
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

        TbfGroupKernelInterfaceCuda_core::M2LBetweenGroups_core(spaceSystem, inLevel, inKernel, inCellGroup, inOtherCellGroup, inIndexes,
                              offsetInteractionIdxs.size(), offsetInteractionIdxs.data(),
                              interactionIdxs.data(), foundSrcIdxs.data());
    }


    template <class KernelClass, class CellGroupClass>
    void L2L(const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
             CellGroupClass& inUpperGroup) const {
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

        TbfGroupKernelInterfaceCuda_core::L2L_core<<<1,1>>>(spaceSystem, inLevel, inKernel, inLowerGroup, inUpperGroup,
                           idxFirstParent, idxParent, interactionOffset.data());
    }

    template <class KernelClass, class LeafGroupClass, class ParticleGroupClass>
    void L2P(KernelClass& inKernel, const LeafGroupClass& inLeafGroup,
             ParticleGroupClass& inParticleGroup) const {
        TbfGroupKernelInterfaceCuda_core::L2P_core<<<1,1>>>(inKernel, inLeafGroup, inParticleGroup);
    }


    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PInGroup(KernelClass& inKernel, ParticleGroupClass& inParticleGroup, const IndexClass& inIndexes) const {
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

        for(long int idxColor = 0 ; idxColor < spaceSystem.getNbNeighborsPerLeaf() ; ++idxColor){
            TbfGroupKernelInterfaceCuda_core::P2PInGroup_core<<<1,1>>>(inKernel, inParticleGroup, inIndexes,
                                                                              interactionBlockIntervals[idxColor].data(),
                                                                              interactionBlocks[idxColor].data(),
                                                                              interactionBlockIntervals[idxColor].size()-1);
        }
    }

    template <class KernelClass, class ParticleGroupClass>
    void P2PInner(KernelClass& inKernel, ParticleGroupClass& inParticleGroup) const {
        TbfGroupKernelInterfaceCuda_core::P2PInner_core<<<1,1>>>(inKernel, inParticleGroup);
    }

    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PBetweenGroups(KernelClass& inKernel, ParticleGroupClass& inParticleGroup,
                          ParticleGroupClass& inOtherParticleGroup, const IndexClass& inIndexes) const {
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

        for(long int idxColor = 0 ; idxColor < spaceSystem.getNbNeighborsPerLeaf() ; ++idxColor){
            TbfGroupKernelInterfaceCuda_core::P2PBetweenGroups_core<<<1,1>>>(inKernel, inParticleGroup, inOtherParticleGroup, inIndexes,
                                                                              interactionBlockIntervals[idxColor].data(),
                                                                              interactionBlocks[idxColor].data(),
                                                                              interactionBlockIntervals[idxColor].size()-1);
        }
    }



    template <class KernelClass, class ParticleGroupClassTarget, class ParticleGroupClassSource, class IndexClass>
    void P2PBetweenGroupsTsm(KernelClass& inKernel, ParticleGroupClassTarget& inParticleGroup,
                             ParticleGroupClassSource& inOtherParticleGroup, const IndexClass& inIndexes) const {

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

        TbfGroupKernelInterfaceCuda_core::P2PBetweenGroupsTsm_core<<<1,1>>>(inKernel, inParticleGroup, inOtherParticleGroup,
                                                                             inIndexes, interactionBlockIntervals.data(),
                                                                             interactionBlocks.data(),  interactionBlockIntervals.size()-1);
    }
};

#endif
