#ifndef TBFINTERACTIONTIMER_HPP
#define TBFINTERACTIONTIMER_HPP

#include "tbfglobal.hpp"

#include "utils/tbftimer.hpp"

template <class RealKernel>
class TbfInteractionTimer : public RealKernel {
public:
    struct Timers{
        TbfTimer P2M;
        TbfTimer M2M;
        TbfTimer M2L;
        TbfTimer L2L;
        TbfTimer L2P;
        TbfTimer P2P;
        TbfTimer P2PInner;

        static Timers Reduce(const Timers& inOther1, const Timers& inOther2){
            Timers result = inOther1;
            result.P2M.merge(inOther2.P2M);
            result.M2M.merge(inOther2.M2M);
            result.M2L.merge(inOther2.M2L);
            result.L2L.merge(inOther2.L2L);
            result.L2P.merge(inOther2.L2P);
            result.P2P.merge(inOther2.P2P);
            result.P2PInner.merge(inOther2.P2PInner);
            return result;
        }

        template <class StreamClass>
        friend StreamClass& operator<<(StreamClass& output, const Timers& inTimers){
            output << "Timers (" << &inTimers << ") :\n";
            output << " - P2M : " << inTimers.P2M.getCumulated() << "s\n";
            output << " - M2M : " << inTimers.M2M.getCumulated() << "s\n";
            output << " - M2L : " << inTimers.M2L.getCumulated() << "s\n";
            output << " - L2L : " << inTimers.L2L.getCumulated() << "s\n";
            output << " - L2P : " << inTimers.L2P.getCumulated() << "s\n";
            output << " - P2P : " << inTimers.P2P.getCumulated() << "s\n";
            output << " - P2PInner : " << inTimers.P2PInner.getCumulated() << "s\n";
            return output;
        }
    };

    using ReduceType = Timers;

private:
    Timers counters;

public:
    using RealKernel::RealKernel;

    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData& inLeafIndex, const long int particlesIndexes[],
             const ParticlesClass& inParticles, const long int inNbParticles, LeafClass& inOutLeaf) {
        counters.P2M.start();
        RealKernel::P2M(inLeafIndex, particlesIndexes, inParticles, inNbParticles, inOutLeaf);
        counters.P2M.stop();
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData& inCellIndex,
             const long int inLevel, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) {
        counters.M2M.start();
        RealKernel::M2M(inCellIndex, inLevel, inLowerCell, inOutUpperCell, childrenPos, inNbChildren);
        counters.M2M.stop();
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData& inTargetIndex,
             const long int inLevel, const CellClassContainer& inInteractingCells, const long int neighPos[], const long int inNbNeighbors,
             CellClass& inOutCell) {
        counters.M2L.start();
        RealKernel::M2L(inTargetIndex, inLevel, inInteractingCells, neighPos, inNbNeighbors, inOutCell);
        counters.M2L.stop();
    }

    template <class CellSymbolicData,class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData& inParentIndex,
             const long int inLevel, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int childrednPos[], const long int inNbChildren) {
        counters.L2L.start();
        RealKernel::L2L(inParentIndex, inLevel, inUpperCell, inOutLowerCell, childrednPos, inNbChildren);
        counters.L2L.stop();
    }

    template <class CellSymbolicData,class LeafClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& inLeafIndex,
             const LeafClass& inLeaf, const long int particlesIndexes[],
             const ParticlesClassValues& inOutParticles, ParticlesClassRhs& inOutParticlesRhs,
             const long int inNbParticles) {
        counters.L2P.start();
        RealKernel::L2P(inLeafIndex, inLeaf, particlesIndexes, inOutParticles, inOutParticlesRhs, inNbParticles);
        counters.L2P.stop();
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData& inNeighborIndex, const long int neighborsIndexes[],
             const ParticlesClassValues& inParticlesNeighbors, ParticlesClassRhs& inParticlesNeighborsRhs,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicData& inParticlesIndex, const long int targetIndexes[], const ParticlesClassValues& inOutParticles,
             ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
             const long arrayIndexSrc) {
        counters.P2P.start();
        RealKernel::P2P(inNeighborIndex, neighborsIndexes, inParticlesNeighbors, inParticlesNeighborsRhs, inNbParticlesNeighbors, inParticlesIndex,
                        targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles, arrayIndexSrc);
        counters.P2P.stop();
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& inNeighborIndex, const long int neighborsIndexes[],
             const ParticlesClassValuesSource& inParticlesNeighbors,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicDataTarget& inParticlesIndex, const long int targetIndexes[],
             const ParticlesClassValuesTarget& inOutParticles,
             ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
             const long arrayIndexSrc) const {
        counters.P2P.start();
        RealKernel::P2P(inNeighborIndex, inParticlesNeighbors, neighborsIndexes, inNbParticlesNeighbors, inParticlesIndex,
                        targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles, arrayIndexSrc);
        counters.P2P.stop();
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& inLeafIndex, const long int targetIndexes[],
                  const ParticlesClassValues& inOutParticles,
                  ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles) {
        counters.P2PInner.start();
        RealKernel::P2PInner(inLeafIndex, targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles);
        counters.P2PInner.stop();
    }

    void reset(){
        counters = Timers();
    }

    const Timers& getReduceData() const{
        return counters;
    }

    static Timers Reduce(const ReduceType& inOther1, const ReduceType& inOther2){
        return Timers::Reduce(inOther1.counters, inOther2.counters);
    }
};

#endif
