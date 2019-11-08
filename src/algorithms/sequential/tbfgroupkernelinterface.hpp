#ifndef TBFGROUPKERNELINTERFACE_HPP
#define TBFGROUPKERNELINTERFACE_HPP

#include "tbfglobal.hpp"

#include <cassert>

template <class SpaceIndexType>
class TbfGroupKernelInterface{
    const SpaceIndexType spaceSystem;

public:
    TbfGroupKernelInterface(SpaceIndexType inSpaceIndex) : spaceSystem(std::move(inSpaceIndex)){}

    template <class KernelClass, class ParticleGroupClass, class LeafGroupClass>
    void P2M(KernelClass& inKernel, const ParticleGroupClass& inParticleGroup,
             LeafGroupClass& inLeafGroup) const {
        assert(inParticleGroup.getNbLeaves() == inLeafGroup.getNbCells());
        for(long int idxLeaf = 0 ; idxLeaf < inParticleGroup.getNbLeaves() ; ++idxLeaf){
            assert(inParticleGroup.getLeafSpacialIndex(idxLeaf) == inLeafGroup.getCellSpacialIndex(idxLeaf));
            inKernel.P2M(inParticleGroup.getParticleData(idxLeaf), inParticleGroup.getNbParticlesInLeaf(idxLeaf),
                         inLeafGroup.getCellMultipole(idxLeaf));
        }
    }

    template <class KernelClass, class CellGroupClass>
    void M2M(const long int inLevel, KernelClass& inKernel, const CellGroupClass& inLowerGroup,
             CellGroupClass& inUpperGroup) const {
        using CellMultipoleType = typename std::remove_reference<decltype(inLowerGroup.getCellMultipole(0))>::type;
        std::vector<std::reference_wrapper<const CellMultipoleType>> children;
        long int positionsOfChildren[spaceSystem.getNbChildrenPerCell()];
        long int nbChildren = 0;

        const auto startingIndex = std::max(spaceSystem.getParentIndex(inLowerGroup.getStartingSpacialIndex()),
                                            inUpperGroup.getStartingSpacialIndex());

        auto foundParent = inUpperGroup.getElementFromSpacialIndex(startingIndex);
        auto foundChild = inLowerGroup.getElementFromParentIndex(spaceSystem, startingIndex);

        assert(foundParent);
        assert(foundChild);

        long int idxParent = (*foundParent);
        long int idxChild = (*foundChild);

        while(idxParent != inUpperGroup.getNbCells()
              && idxChild != inLowerGroup.getNbCells()){
            assert(spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

            assert(nbChildren < spaceSystem.getNbChildrenPerCell());
            children.emplace_back(inLowerGroup.getCellMultipole(idxChild));
            positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(inLowerGroup.getCellSpacialIndex(idxChild));
            nbChildren += 1;

            idxChild += 1;
            if(idxChild != inLowerGroup.getNbCells()
                    && spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) != inUpperGroup.getCellSpacialIndex(idxParent)){

                inKernel.M2M(inLevel, children, inUpperGroup.getCellMultipole(idxParent),
                             positionsOfChildren, nbChildren);

                idxParent += 1;
                assert(idxParent == inUpperGroup.getNbCells()
                        || spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

                children.clear();
                nbChildren = 0;
            }
        }

        if(nbChildren){
            inKernel.M2M(inLevel, children, inUpperGroup.getCellMultipole(idxParent),
                     positionsOfChildren, nbChildren);
        }
    }

    template <class KernelClass, class CellGroupClass, class IndexClass>
    void M2LInGroup(const long int inLevel, KernelClass& inKernel, CellGroupClass& inCellGroup, const IndexClass& inIndexes) const {
        using CellMultipoleType = typename std::remove_reference<decltype(inCellGroup.getCellMultipole(0))>::type;
        using CellLocalType = typename std::remove_reference<decltype(inCellGroup.getCellLocal(0))>::type;

        std::vector<std::reference_wrapper<const CellMultipoleType>> neighbors;
        long int positionsOfNeighbors[spaceSystem.getNbNeighborsPerCell()];
        long int nbNeighbors = 0;

        long int idxInteraction = 0;

        while(idxInteraction < static_cast<long int>(inIndexes.size())){
            const auto interaction = inIndexes[idxInteraction];

            auto& targetCell = inCellGroup.getCellLocal(interaction.globalTargetPos);

            do{
                auto foundSrc = inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexSrc);
                assert(foundSrc);
                assert(inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget)
                       && *inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget) == inIndexes[idxInteraction].globalTargetPos);

                assert(nbNeighbors < spaceSystem.getNbNeighborsPerCell());
                neighbors.emplace_back(inCellGroup.getCellMultipole(*foundSrc));
                positionsOfNeighbors[nbNeighbors] = inIndexes[idxInteraction].arrayIndexSrc;
                nbNeighbors += 1;

                idxInteraction += 1;
            } while(idxInteraction < static_cast<long int>(inIndexes.size())
                    && interaction.indexTarget == inIndexes[idxInteraction].indexTarget);

            inKernel.M2L(inLevel,
                         neighbors,
                         positionsOfNeighbors,
                         nbNeighbors,
                         targetCell);
            neighbors.clear();
            nbNeighbors = 0;
        }
    }

    template <class KernelClass, class CellGroupClass, class IndexClass>
    void M2LBetweenGroups(const long int inLevel, KernelClass& inKernel, CellGroupClass& inCellGroup,
                          const CellGroupClass& inOtherCellGroup, const IndexClass& inIndexes) const {
        using CellMultipoleType = typename std::remove_reference<decltype(inOtherCellGroup.getCellMultipole(0))>::type;
        using CellLocalType = typename std::remove_reference<decltype(inCellGroup.getCellLocal(0))>::type;

        std::vector<std::reference_wrapper<const CellMultipoleType>> neighbors;
        long int positionsOfNeighbors[spaceSystem.getNbNeighborsPerCell()];
        long int nbNeighbors = 0;

        long int idxInteraction = 0;

        while(idxInteraction < static_cast<long int>(inIndexes.size())){
            const auto interaction = inIndexes[idxInteraction];

            auto& targetCell = inCellGroup.getCellLocal(interaction.globalTargetPos);

            do{
                auto foundSrc = inOtherCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexSrc);
                if(foundSrc){
                    assert(inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget)
                          && *inCellGroup.getElementFromSpacialIndex(inIndexes[idxInteraction].indexTarget) == inIndexes[idxInteraction].globalTargetPos);

                    assert(nbNeighbors < spaceSystem.getNbNeighborsPerCell());
                    neighbors.emplace_back(inOtherCellGroup.getCellMultipole(*foundSrc));
                    positionsOfNeighbors[nbNeighbors] = inIndexes[idxInteraction].arrayIndexSrc;
                    nbNeighbors += 1;
                }

                idxInteraction += 1;
            } while(idxInteraction < static_cast<long int>(inIndexes.size())
                    && interaction.indexTarget == inIndexes[idxInteraction].indexTarget);

            if(nbNeighbors){
                inKernel.M2L(inLevel,
                         neighbors,
                         positionsOfNeighbors,
                         nbNeighbors,
                         targetCell);
                neighbors.clear();
                nbNeighbors = 0;
            }
        }
    }

    template <class KernelClass, class CellGroupClass>
    void L2L(const long int inLevel, KernelClass& inKernel, const CellGroupClass& inUpperGroup,
             CellGroupClass& inLowerGroup) const {
        using CellLocalType = typename std::remove_reference<decltype(inLowerGroup.getCellMultipole(0))>::type;
        std::vector<std::reference_wrapper<CellLocalType>> children;
        long int positionsOfChildren[spaceSystem.getNbChildrenPerCell()];
        long int nbChildren = 0;

        const auto startingIndex = std::max(spaceSystem.getParentIndex(inLowerGroup.getStartingSpacialIndex()),
                                            inUpperGroup.getStartingSpacialIndex());

        auto foundParent = inUpperGroup.getElementFromSpacialIndex(startingIndex);
        auto foundChild = inLowerGroup.getElementFromParentIndex(spaceSystem, startingIndex);

        assert(foundParent);
        assert(foundChild);

        long int idxParent = (*foundParent);
        long int idxChild = (*foundChild);

        while(idxParent != inUpperGroup.getNbCells()
              && idxChild != inLowerGroup.getNbCells()){
            assert(spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

            assert(nbChildren < spaceSystem.getNbChildrenPerCell());
            children.emplace_back(inLowerGroup.getCellLocal(idxChild));
            positionsOfChildren[nbChildren] = spaceSystem.childPositionFromParent(inLowerGroup.getCellSpacialIndex(idxChild));
            nbChildren += 1;

            idxChild += 1;
            if(idxChild != inLowerGroup.getNbCells()
                    && spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) != inUpperGroup.getCellSpacialIndex(idxParent)){

                inKernel.L2L(inLevel, inUpperGroup.getCellLocal(idxParent), children,
                             positionsOfChildren, nbChildren);

                idxParent += 1;
                assert(idxParent == inUpperGroup.getNbCells()
                        || spaceSystem.getParentIndex(inLowerGroup.getCellSpacialIndex(idxChild)) == inUpperGroup.getCellSpacialIndex(idxParent));

                children.clear();
                nbChildren = 0;
            }
        }

        if(nbChildren){
            inKernel.L2L(inLevel, inUpperGroup.getCellLocal(idxParent), children,
                         positionsOfChildren, nbChildren);
        }
    }

    template <class KernelClass, class LeafGroupClass, class ParticleGroupClass>
    void L2P(KernelClass& inKernel, const LeafGroupClass& inLeafGroup,
             ParticleGroupClass& inParticleGroup) const {
        assert(inParticleGroup.getNbLeaves() == inLeafGroup.getNbCells());
        for(long int idxLeaf = 0 ; idxLeaf < inParticleGroup.getNbLeaves() ; ++idxLeaf){
            assert(inParticleGroup.getLeafSpacialIndex(idxLeaf) == inLeafGroup.getCellSpacialIndex(idxLeaf));
            inKernel.L2P(inLeafGroup.getCellLocal(idxLeaf), inParticleGroup.getParticleRhs(idxLeaf),
                          inParticleGroup.getNbParticlesInLeaf(idxLeaf));
        }
    }

    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PInGroup(KernelClass& inKernel, ParticleGroupClass& inParticleGroup, const IndexClass& inIndexes) const {
        for(long int idxInteraction = 0 ; idxInteraction < static_cast<long int>(inIndexes.size()) ; ++idxInteraction){
            const auto interaction = inIndexes[idxInteraction];

            auto foundSrc = inParticleGroup.getElementFromSpacialIndex(interaction.indexSrc);
            assert(foundSrc);
            assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                   && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

            inKernel.P2P(inParticleGroup.getParticleData(*foundSrc), inParticleGroup.getNbParticlesInLeaf(*foundSrc),
                         interaction.arrayIndexSrc,
                         inParticleGroup.getParticleRhs(interaction.globalTargetPos), inParticleGroup.getNbParticlesInLeaf(interaction.globalTargetPos));
        }
    }

    template <class KernelClass, class ParticleGroupClass>
    void P2PInner(KernelClass& inKernel, ParticleGroupClass& inParticleGroup) const {
        for(long int idxLeaf = 0 ; idxLeaf < static_cast<long int>(inParticleGroup.getNbLeaves()) ; ++idxLeaf){
            inKernel.P2PInner(inParticleGroup.getParticleData(idxLeaf),
                         inParticleGroup.getParticleRhs(idxLeaf), inParticleGroup.getNbParticlesInLeaf(idxLeaf));
        }
    }

    template <class KernelClass, class ParticleGroupClass, class IndexClass>
    void P2PBetweenGroups(KernelClass& inKernel, ParticleGroupClass& inParticleGroup,
                          const ParticleGroupClass& inOtherParticleGroup, const IndexClass& inIndexes) const {
        for(long int idxInteraction = 0 ; idxInteraction < static_cast<long int>(inIndexes.size()) ; ++idxInteraction){
            const auto interaction = inIndexes[idxInteraction];

            auto foundSrc = inOtherParticleGroup.getElementFromSpacialIndex(interaction.indexSrc);
            if(foundSrc){
                assert(inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget)
                       && *inParticleGroup.getElementFromSpacialIndex(interaction.indexTarget) == interaction.globalTargetPos);

                inKernel.P2P(inOtherParticleGroup.getParticleData(*foundSrc), inOtherParticleGroup.getNbParticlesInLeaf(*foundSrc),
                             interaction.arrayIndexSrc,
                             inParticleGroup.getParticleRhs(interaction.globalTargetPos), inParticleGroup.getNbParticlesInLeaf(interaction.globalTargetPos));
            }
        }
    }
};

#endif
