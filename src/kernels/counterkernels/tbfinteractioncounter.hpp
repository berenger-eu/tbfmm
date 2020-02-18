#ifndef TBFINTERACTIONCOUNTER_HPP
#define TBFINTERACTIONCOUNTER_HPP

#include "tbfglobal.hpp"

template <class RealKernel>
class TbfInteractionCounter : public RealKernel {
public:
    struct Counters{
        long int P2M = 0;
        long int M2M = 0;
        long int M2L = 0;
        long int L2L = 0;
        long int L2P = 0;
        long int P2P = 0;
        long int P2PInner = 0;

        static Counters Reduce(const Counters& inOther1, const Counters& inOther2){
            Counters result;
            result.P2M = inOther1.P2M + inOther2.P2M;
            result.M2M = inOther1.M2M + inOther2.M2M;
            result.M2L = inOther1.M2L + inOther2.M2L;
            result.L2L = inOther1.L2L + inOther2.L2L;
            result.L2P = inOther1.L2P + inOther2.L2P;
            result.P2P = inOther1.P2P + inOther2.P2P;
            result.P2PInner = inOther1.P2PInner + inOther2.P2PInner;
            return result;
        }

        template <class StreamClass>
        friend StreamClass& operator<<(StreamClass& output, const Counters& inCounters){
            output << "Counters (" << &inCounters << ") :\n";
            output << " - P2M : " << inCounters.P2M << " (the number of leaves)\n";
            output << " - M2M : " << inCounters.M2M << " (the number of parent-child relations)\n";
            output << " - M2L : " << inCounters.M2L << " (the number of target-neighbor relations)\n";
            output << " - L2L : " << inCounters.L2L << " (the number of parent-child relations)\n";
            output << " - L2P : " << inCounters.L2P << " (the number of leaves)\n";
            output << " - P2P : " << inCounters.P2P << " (the number of \"particle with particle\" interactions)\n";
            output << " - P2PInner : " << inCounters.P2PInner << " (the number of \"particle with particle\" interactions)\n";
            return output;
        }
    };

    using ReduceType = Counters;

private:
    Counters counters;

public:
    using RealKernel::RealKernel;

    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData& inLeafIndex,
             const long int particlesIndexes[], const ParticlesClass& inParticles, const long int inNbParticles, LeafClass& inOutLeaf) {
        counters.P2M += 1;
        RealKernel::P2M(inLeafIndex, particlesIndexes, inParticles, inNbParticles, inOutLeaf);
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData& inCellIndex,
             const long int inLevel, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) {
        counters.M2M += inNbChildren;
        RealKernel::M2M(inCellIndex, inLevel, inLowerCell, inOutUpperCell, childrenPos, inNbChildren);
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData& inTargetIndex,
             const long int inLevel, const CellClassContainer& inInteractingCells, const long int neighPos[], const long int inNbNeighbors,
             CellClass& inOutCell) {
        counters.M2L += inNbNeighbors;
        RealKernel::M2L(inTargetIndex, inLevel, inInteractingCells, neighPos, inNbNeighbors, inOutCell);
    }

    template <class CellSymbolicData,class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData& inParentIndex,
             const long int inLevel, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int childrednPos[], const long int inNbChildren) {
        counters.L2L += inNbChildren;
        RealKernel::L2L(inParentIndex, inLevel, inUpperCell, inOutLowerCell, childrednPos, inNbChildren);
    }

    template <class CellSymbolicData,class LeafClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& inLeafIndex,
             const LeafClass& inLeaf, const long int particlesIndexes[],
             const ParticlesClassValues& inOutParticles, ParticlesClassRhs& inOutParticlesRhs,
             const long int inNbParticles) {
        counters.L2P += 1;
        RealKernel::L2P(inLeafIndex, inLeaf, particlesIndexes, inOutParticles, inOutParticlesRhs, inNbParticles);
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData& inNeighborIndex, const long int neighborsIndexes[],
             const ParticlesClassValues& inParticlesNeighbors, ParticlesClassRhs& inParticlesNeighborsRhs,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicData& inParticlesIndex, const long int targetIndexes[],
             const ParticlesClassValues& inOutParticles,
             ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
             const long arrayIndexSrc) {
        counters.P2P += inNbParticlesNeighbors * inNbOutParticles;
        RealKernel::P2P(inNeighborIndex,neighborsIndexes, inParticlesNeighbors, inParticlesNeighborsRhs, inNbParticlesNeighbors, inParticlesIndex,
                        targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles, arrayIndexSrc);
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& inNeighborIndex, const long int neighborsIndexes[],
             const ParticlesClassValuesSource& inParticlesNeighbors,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicDataTarget& inParticlesIndex, const long int targetIndexes[],
             const ParticlesClassValuesTarget& inOutParticles,
             ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
             const long arrayIndexSrc) const {
        counters.P2P += inNbParticlesNeighbors * inNbOutParticles;
        RealKernel::P2P(inNeighborIndex, inParticlesNeighbors, neighborsIndexes, inNbParticlesNeighbors, inParticlesIndex,
                        targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles, arrayIndexSrc);
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& inLeafIndex, const long int targetIndexes[],
                  const ParticlesClassValues& inOutParticles,
                  ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles) {
        counters.P2PInner += inNbOutParticles * inNbOutParticles - inNbOutParticles;
        RealKernel::P2PInner(inLeafIndex, targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles);
    }

    void reset(){
        counters = Counters();
    }

    const Counters& getReduceData() const{
        return counters;
    }

    static Counters Reduce(const ReduceType& inOther1, const ReduceType& inOther2){
        return Counters::Reduce(inOther1.counters, inOther2.counters);
    }
};

#endif
