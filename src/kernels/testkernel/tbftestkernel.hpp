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

    template <class CellClass>
    void M2M(const long int /*inLevel*/, const CellClass& inLowerCell, CellClass& inOutUpperCell, const long int /*childPos*/) const {
        inOutUpperCell[0] += inLowerCell[0];
    }

    template <class CellClass>
    void M2L(const long int /*inLevel*/, const CellClass& inInteractingCell, const long int /*neighPos*/, CellClass& inOutCell) const {
        inOutCell[0] += inInteractingCell[0];
    }

    template <class CellClass>
    void L2L(const long int /*inLevel*/, const CellClass& inUpperCell, CellClass& inOutLowerCell, const long int /*childPos*/) const {
        inOutLowerCell[0] += inUpperCell[0];
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
