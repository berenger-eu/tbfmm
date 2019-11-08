#ifndef TBFTESTKERNEL_HPP
#define TBFTESTKERNEL_HPP

#include "tbfglobal.hpp"

template <class RealType_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class TbfTestKernel{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;
public:
    explicit TbfTestKernel(const SpacialConfiguration& /*inConfiguration*/){}
    explicit TbfTestKernel(const TbfTestKernel&){}

    template <class ParticlesClass, class LeafClass>
    void P2M(const ParticlesClass&& /*inParticles*/, const long int inNbParticles, LeafClass& inOutLeaf) const {
        inOutLeaf[0] += inNbParticles;
    }

    template <class CellClassContainer, class CellClass>
    void M2M(const long int /*inLevel*/, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int /*childrenPos*/[], const int inNbChildren) const {
        for(long int idxChild = 0 ; idxChild < inNbChildren ; ++idxChild){
            const CellClass& child = inLowerCell[idxChild];
            inOutUpperCell[0] += child[0];
        }
    }

    template <class CellClassContainer, class CellClass>
    void M2L(const long int /*inLevel*/, const CellClassContainer& inInteractingCells, const long int /*neighPos*/[], const long int inNbNeighbors,
             CellClass& inOutCell) const {
        for(long int idxNeigh = 0 ; idxNeigh < inNbNeighbors ; ++idxNeigh){
            const CellClass& neighbor = inInteractingCells[idxNeigh];
            inOutCell[0] += neighbor[0];
        }
    }

    template <class CellClass, class CellClassContainer>
    void L2L(const long int /*inLevel*/, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int /*childrednPos*/[], const long int inNbChildren) const {
        for(long int idxChild = 0 ; idxChild < inNbChildren ; ++idxChild){
            CellClass& child = inOutLowerCell[idxChild];
            child[0] += inUpperCell[0];
        }
    }

    template <class LeafClass, class ParticlesClass>
    void L2P(const LeafClass& inLeaf, ParticlesClass&& inOutParticles, const long int inNbParticles) const {
        for(int idxPart = 0 ; idxPart < inNbParticles ; ++idxPart){
            inOutParticles[0][idxPart] += inLeaf[0];
        }
    }

    template <class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const ParticlesClassValues&& /*inParticlesNeighbors*/, const long int inNbParticlesNeighbors,
             const long int /*inNeighborPos*/, ParticlesClassRhs&& inOutParticles, const long int inNbOutParticles) const {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticles[0][idxPart] += inNbParticlesNeighbors;
        }
    }

    template <class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const ParticlesClassValues&& /*inParticlesNeighbors*/,
                  ParticlesClassRhs&& inOutParticles, const long int inNbOutParticles) const {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticles[0][idxPart] += inNbOutParticles - 1;
        }
    }
};

#endif
