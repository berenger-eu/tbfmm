#ifndef TBFINTERACTIONPRINTER_HPP
#define TBFINTERACTIONPRINTER_HPP

#include "tbfglobal.hpp"
#include "utils/tbfutils.hpp"
#include "utils/tbfperiodicshifter.hpp"

#include <iostream>

template <class RealKernel>
class TbfInteractionPrinter : public RealKernel {
    using RealType = typename RealKernel::RealType;
    constexpr static long int Dim = RealKernel::SpaceIndexType::Dim;

    auto getBoxWidthsAtLevel(const long int inLevel) const{
        std::array<RealType, Dim> widths = spaceSystem.getConfiguration().getBoxWidths();
        for(RealType& width : widths){
            width /= RealType(1<<inLevel);
        }
        return widths;
    }

    typename RealKernel::SpaceIndexType spaceSystem;

public:
    using ReduceType = void;

    template <class ... Params>
    explicit TbfInteractionPrinter(const typename RealKernel::SpacialConfiguration& inConfiguration, Params ... params)
        : RealKernel(inConfiguration, std::forward<Params>(params)...), spaceSystem(inConfiguration) {}

    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData& inLeafIndex,
             const long int particlesIndexes[], const ParticlesClass& inParticles, const long int inNbParticles, LeafClass& inOutLeaf) {
        std::cout << "[INTERACTION] P2M:" << std::endl;
        std::cout << "[INTERACTION]  - Leaf index: " << inLeafIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Leaf coord: " << TbfUtils::ArrayPrinter(inLeafIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles: " << inNbParticles << std::endl;
        std::cout << "[INTERACTION]  - Leaf widths: " << TbfUtils::ArrayPrinter(spaceSystem.getConfiguration().getLeafWidths()) << std::endl;
        RealKernel::P2M(inLeafIndex, particlesIndexes, inParticles, inNbParticles, inOutLeaf);
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData& inCellIndex,
             const long int inLevel, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) {
        std::cout << "[INTERACTION] M2M:" << std::endl;
        std::cout << "[INTERACTION]  - Parent index: " << inCellIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Level: " << inLevel << std::endl;
        std::cout << "[INTERACTION]  - Cell widths: " << TbfUtils::ArrayPrinter(getBoxWidthsAtLevel(inLevel)) << std::endl;
        std::cout << "[INTERACTION]  - Number of children: " << inNbChildren << std::endl;
        RealKernel::M2M(inCellIndex, inLevel, inLowerCell, inOutUpperCell, childrenPos, inNbChildren);
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData& inTargetIndex,
             const long int inLevel, const CellClassContainer& inInteractingCells, const long int neighPos[], const long int inNbNeighbors,
             CellClass& inOutCell) {
        std::cout << "[INTERACTION] M2L:" << std::endl;
        std::cout << "[INTERACTION]  - Target index: " << inTargetIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Level: " << inLevel << std::endl;
        std::cout << "[INTERACTION]  - Cell widths: " << TbfUtils::ArrayPrinter(getBoxWidthsAtLevel(inLevel)) << std::endl;
        std::cout << "[INTERACTION]  - Number of neighbors: " << inNbNeighbors << std::endl;

        for(long int idxInteraction = 0 ; idxInteraction < inNbNeighbors ; ++idxInteraction){
            std::cout << "[INTERACTION]  - Idx : " << idxInteraction << std::endl;
            std::cout << "[INTERACTION]  - Interaction index : " << neighPos[idxInteraction] << std::endl;
            std::cout << "[INTERACTION]  - Interaction relative position : " << TbfUtils::ArrayPrinter(spaceSystem.getRelativePosFromInteractionIndex(neighPos[idxInteraction])) << std::endl;
        }

        RealKernel::M2L(inTargetIndex, inLevel, inInteractingCells, neighPos, inNbNeighbors, inOutCell);
    }

    template <class CellSymbolicData,class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData& inParentIndex,
             const long int inLevel, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int childrednPos[], const long int inNbChildren) {
        std::cout << "[INTERACTION] L2L:" << std::endl;
        std::cout << "[INTERACTION]  - Parent index: " << inParentIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Level: " << inLevel << std::endl;
        std::cout << "[INTERACTION]  - Cell widths: " << TbfUtils::ArrayPrinter(getBoxWidthsAtLevel(inLevel)) << std::endl;
        std::cout << "[INTERACTION]  - Number of children: " << inNbChildren << std::endl;
        RealKernel::L2L(inParentIndex, inLevel, inUpperCell, inOutLowerCell, childrednPos, inNbChildren);
    }

    template <class CellSymbolicData,class LeafClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& inLeafIndex,
             const LeafClass& inLeaf, const long int particlesIndexes[],
             const ParticlesClassValues& inOutParticles, ParticlesClassRhs& inOutParticlesRhs,
             const long int inNbParticles) {
        std::cout << "[INTERACTION] L2P:" << std::endl;
        std::cout << "[INTERACTION]  - Leaf index: " << inLeafIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Leaf coord: " << TbfUtils::ArrayPrinter(inLeafIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles: " << inNbParticles << std::endl;
        std::cout << "[INTERACTION]  - Leaf widths: " << TbfUtils::ArrayPrinter(spaceSystem.getConfiguration().getLeafWidths()) << std::endl;
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
        std::cout << "[INTERACTION] P2P:" << std::endl;
        std::cout << "[INTERACTION]  - Neighbor index: " << inNeighborIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Neighbor pos: " << TbfUtils::ArrayPrinter(inNeighborIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles in neighbors: " << inNbParticlesNeighbors << std::endl;
        std::cout << "[INTERACTION]  - Target index: " << inParticlesIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Target pos: " << TbfUtils::ArrayPrinter(inParticlesIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles in target: " << inNbOutParticles << std::endl;
        std::cout << "[INTERACTION]  - Array index: " << arrayIndexSrc << std::endl;
        std::cout << "[INTERACTION]  - Leaf widths: " << TbfUtils::ArrayPrinter(spaceSystem.getConfiguration().getLeafWidths()) << std::endl;

        if constexpr(RealKernel::SpaceIndexType::IsPeriodic){
            using PeriodicShifter = typename TbfPeriodicShifter<typename RealKernel::RealType, typename RealKernel::SpaceIndexType>::Neighbor;
            if(PeriodicShifter::NeedToShift(inNeighborIndex, inParticlesIndex, spaceSystem, arrayIndexSrc)){
                std::cout << "[INTERACTION]  - This is a periodic interaction" << std::endl;
                std::cout << "[INTERACTION]  - Periodic shift coef to apply to source: " << TbfUtils::ArrayPrinter(PeriodicShifter::GetShiftCoef(inNeighborIndex, inParticlesIndex, spaceSystem, arrayIndexSrc)) << std::endl;
            }
        }

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
        std::cout << "[INTERACTION] P2PTsm:" << std::endl;
        std::cout << "[INTERACTION]  - Neighbor indboxLimitex: " << inNeighborIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Neighbor pos: " << TbfUtils::ArrayPrinter(inNeighborIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles in neighbors: " << inNbParticlesNeighbors << std::endl;
        std::cout << "[INTERACTION]  - Target index: " << inParticlesIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Target pos: " << TbfUtils::ArrayPrinter(inParticlesIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles in target: " << inNbOutParticles << std::endl;
        std::cout << "[INTERACTION]  - Array index: " << arrayIndexSrc << std::endl;
        std::cout << "[INTERACTION]  - Leaf widths: " << TbfUtils::ArrayPrinter(spaceSystem.getConfiguration().getLeafWidths()) << std::endl;

        if constexpr(RealKernel::SpaceIndexType::IsPeriodic){
            using PeriodicShifter = typename TbfPeriodicShifter<typename RealKernel::RealType, typename RealKernel::SpaceIndexType>::Neighbor;
            if(PeriodicShifter::NeedToShift(inNeighborIndex, inParticlesIndex, spaceSystem, arrayIndexSrc)){
                std::cout << "[INTERACTION]  - This is a periodic interaction" << std::endl;
                std::cout << "[INTERACTION]  - Periodic shift coef to apply to source: " << TbfUtils::ArrayPrinter(PeriodicShifter::GetShiftCoef(inNeighborIndex, inParticlesIndex, spaceSystem, arrayIndexSrc)) << std::endl;
            }
        }

        RealKernel::P2P(inNeighborIndex, inParticlesNeighbors, neighborsIndexes, inNbParticlesNeighbors, inParticlesIndex,
                        targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles, arrayIndexSrc);
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& inLeafIndex, const long int targetIndexes[],
                  const ParticlesClassValues& inOutParticles,
                  ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles) {
        std::cout << "[INTERACTION] P2PInner:" << std::endl;
        std::cout << "[INTERACTION]  - Target index: " << inLeafIndex.spaceIndex << std::endl;
        std::cout << "[INTERACTION]  - Target pos: " << TbfUtils::ArrayPrinter(inLeafIndex.boxCoord) << std::endl;
        std::cout << "[INTERACTION]  - Number of particles in target: " << inNbOutParticles << std::endl;
        std::cout << "[INTERACTION]  - Leaf widths: " << TbfUtils::ArrayPrinter(spaceSystem.getConfiguration().getLeafWidths()) << std::endl;
        RealKernel::P2PInner(inLeafIndex, targetIndexes, inOutParticles, inOutParticlesRhs, inNbOutParticles);
    }
};

#endif
