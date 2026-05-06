#ifndef TBFGAUSSKERNEL_HPP
#define TBFGAUSSKERNEL_HPP

#include <array>
#include <cmath>

#include "tbfglobal.hpp"
#include "kernels/P2P/FP2PGauss.hpp"
#include "utils/tbfperiodicshifter.hpp"

template <
    class RealType_T,
    int P,
    class SpaceIndexType_T = TbfDefaultSpaceIndexType2D<RealType_T>>
class TbfGaussKernel
{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration =
        TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

private:
    static_assert(SpaceIndexType::Dim == 2,
                  "TbfGaussKernel is implemented only for 2D");

    static constexpr int OrderSize = P + 1;
    static constexpr int ExpansionSize = OrderSize * OrderSize;

    const SpaceIndexType spaceIndexSystem;

    const RealType boxWidth;
    const int treeHeight;
    const RealType widthAtLeafLevel;
    const RealType widthAtLeafLevelDiv2;
    const std::array<RealType, SpaceIndexType::Dim> boxCorner;

    /**
     * Gaussian kernel:
     *
     * K(x,y) = coefficient * exp(-|x-y|^2 / delta)
     */
    const RealType delta;
    const RealType coefficient;

    const RealType sqrtDelta;
    const RealType invSqrtDelta;

    RealType factorials[OrderSize];

    template <class Container>
    static auto GetPtr(Container& ptr)
    {
        return &ptr[0];
    }

    template <class Container>
    static auto GetPtr(const Container& ptr)
    {
        return &ptr[0];
    }

    static constexpr int expansionIndex(const int ax, const int ay)
    {
        return ax * OrderSize + ay;
    }

    void precomputeFactorials()
    {
        factorials[0] = RealType(1);

        for (int i = 1; i < OrderSize; ++i)
        {
            factorials[i] = RealType(i) * factorials[i - 1];
        }
    }

    std::array<RealType, SpaceIndexType::Dim>
    getLeafCenter(const std::array<long int, SpaceIndexType::Dim>& coordinate) const
    {
        std::array<RealType, SpaceIndexType::Dim> center{};

        for (int d = 0; d < SpaceIndexType::Dim; ++d)
        {
            center[d] =
                boxCorner[d]
                + (RealType(coordinate[d]) + RealType(0.5)) * widthAtLeafLevel;
        }

        return center;
    }

    std::array<RealType, SpaceIndexType::Dim>
    getLeafCenter(const typename SpaceIndexType::IndexType& inIndex) const
    {
        return getLeafCenter(spaceIndexSystem.getBoxPosFromIndex(inIndex));
    }

    std::array<RealType, SpaceIndexType::Dim>
    getBoxCenter(const std::array<long int, SpaceIndexType::Dim>& coord,
                 const long int level) const
    {
        const RealType widthAtLevel =
            boxWidth * std::pow(RealType(0.5), RealType(level));

        std::array<RealType, SpaceIndexType::Dim> center{};

        for (int d = 0; d < SpaceIndexType::Dim; ++d)
        {
            center[d] =
                boxCorner[d]
                + (RealType(coord[d]) + RealType(0.5)) * widthAtLevel;
        }

        return center;
    }

public:
    explicit TbfGaussKernel(
        const SpacialConfiguration& inConfiguration,
        const RealType inDelta = RealType(1),
        const RealType inCoefficient = RealType(1))
        : spaceIndexSystem(inConfiguration),
          boxWidth(inConfiguration.getBoxWidths()[0]),
          treeHeight(int(inConfiguration.getTreeHeight())),
          widthAtLeafLevel(inConfiguration.getLeafWidths()[0]),
          widthAtLeafLevelDiv2(widthAtLeafLevel / RealType(2)),
          boxCorner(inConfiguration.getBoxCorner()),
          delta(inDelta),
          coefficient(inCoefficient),
          sqrtDelta(std::sqrt(inDelta)),
          invSqrtDelta(RealType(1) / std::sqrt(inDelta))
    {
        precomputeFactorials();
    }

    TbfGaussKernel(const TbfGaussKernel& other)
        : spaceIndexSystem(other.spaceIndexSystem),
          boxWidth(other.boxWidth),
          treeHeight(other.treeHeight),
          widthAtLeafLevel(other.widthAtLeafLevel),
          widthAtLeafLevelDiv2(other.widthAtLeafLevelDiv2),
          boxCorner(other.boxCorner),
          delta(other.delta),
          coefficient(other.coefficient),
          sqrtDelta(other.sqrtDelta),
          invSqrtDelta(other.invSqrtDelta)
    {
        precomputeFactorials();
    }

    TbfGaussKernel(TbfGaussKernel&&) = default;

    TbfGaussKernel& operator=(const TbfGaussKernel&) = default;
    TbfGaussKernel& operator=(TbfGaussKernel&&) = default;

    template <
        class LeafSymbolicData,
        class ParticlesClassValues,
        class ParticlesClassRhs>
    void P2P(
        const LeafSymbolicData& inNeighborIndex,
        const long int /*neighborsIndexes*/[],
        const ParticlesClassValues& inNeighbors,
        ParticlesClassRhs& inNeighborsRhs,
        const long int inNbParticlesNeighbors,
        const LeafSymbolicData& inTargetIndex,
        const long int /*targetIndexes*/[],
        const ParticlesClassValues& inTargets,
        ParticlesClassRhs& inTargetsRhs,
        const long int inNbOutParticles,
        [[maybe_unused]] const long arrayIndexSrc) const
    {
        if constexpr (SpaceIndexType::IsPeriodic)
        {
            using PeriodicShifter =
                typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;

            if (PeriodicShifter::NeedToShift(
                    inNeighborIndex,
                    inTargetIndex,
                    spaceIndexSystem,
                    arrayIndexSrc))
            {
                const auto duplicateSources =
                    PeriodicShifter::DuplicatePositionsAndApplyShift(
                        inNeighborIndex,
                        inTargetIndex,
                        spaceIndexSystem,
                        arrayIndexSrc,
                        inNeighbors,
                        inNbParticlesNeighbors);

                FP2PGauss::template FullMutual<RealType>(
                    duplicateSources,
                    inNeighborsRhs,
                    inNbParticlesNeighbors,
                    inTargets,
                    inTargetsRhs,
                    inNbOutParticles,
                    delta,
                    coefficient);

                PeriodicShifter::FreePositions(duplicateSources);
            }
            else
            {
                FP2PGauss::template FullMutual<RealType>(
                    inNeighbors,
                    inNeighborsRhs,
                    inNbParticlesNeighbors,
                    inTargets,
                    inTargetsRhs,
                    inNbOutParticles,
                    delta,
                    coefficient);
            }
        }
        else
        {
            FP2PGauss::template FullMutual<RealType>(
                inNeighbors,
                inNeighborsRhs,
                inNbParticlesNeighbors,
                inTargets,
                inTargetsRhs,
                inNbOutParticles,
                delta,
                coefficient);
        }
    }

    template <
        class LeafSymbolicDataSource,
        class ParticlesClassValuesSource,
        class LeafSymbolicDataTarget,
        class ParticlesClassValuesTarget,
        class ParticlesClassRhs>
    void P2PTsm(
        const LeafSymbolicDataSource& inNeighborIndex,
        const long int /*neighborsIndexes*/[],
        const ParticlesClassValuesSource& inNeighbors,
        const long int inNbParticlesNeighbors,
        const LeafSymbolicDataTarget& inTargetIndex,
        const long int /*targetIndexes*/[],
        const ParticlesClassValuesTarget& inTargets,
        ParticlesClassRhs& inTargetsRhs,
        const long int inNbOutParticles,
        [[maybe_unused]] const long arrayIndexSrc) const
    {
        if constexpr (SpaceIndexType::IsPeriodic)
        {
            using PeriodicShifter =
                typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;

            if (PeriodicShifter::NeedToShift(
                    inNeighborIndex,
                    inTargetIndex,
                    spaceIndexSystem,
                    arrayIndexSrc))
            {
                const auto duplicateSources =
                    PeriodicShifter::DuplicatePositionsAndApplyShift(
                        inNeighborIndex,
                        inTargetIndex,
                        spaceIndexSystem,
                        arrayIndexSrc,
                        inNeighbors,
                        inNbParticlesNeighbors);

                FP2PGauss::template GenericFullRemote<RealType>(
                    duplicateSources,
                    inNbParticlesNeighbors,
                    inTargets,
                    inTargetsRhs,
                    inNbOutParticles,
                    delta,
                    coefficient);

                PeriodicShifter::FreePositions(duplicateSources);
            }
            else
            {
                FP2PGauss::template GenericFullRemote<RealType>(
                    inNeighbors,
                    inNbParticlesNeighbors,
                    inTargets,
                    inTargetsRhs,
                    inNbOutParticles,
                    delta,
                    coefficient);
            }
        }
        else
        {
            FP2PGauss::template GenericFullRemote<RealType>(
                inNeighbors,
                inNbParticlesNeighbors,
                inTargets,
                inTargetsRhs,
                inNbOutParticles,
                delta,
                coefficient);
        }
    }

    template <
        class LeafSymbolicData,
        class ParticlesClassValues,
        class ParticlesClassRhs>
    void P2PInner(
        const LeafSymbolicData& /*inIndex*/,
        const long int /*indexes*/[],
        const ParticlesClassValues& inTargets,
        ParticlesClassRhs& inTargetsRhs,
        const long int inNbOutParticles) const
    {
        FP2PGauss::template GenericInner<RealType>(
            inTargets,
            inTargetsRhs,
            inNbOutParticles,
            delta,
            coefficient);
    }

    /**
     * @brief Particle to Multipole.
     *
     * Build Gaussian multipole expansion coefficients
     * from particles inside one leaf.
     */
    template <
        class CellSymbolicData,
        class ParticlesClass,
        class LeafClass>
    void P2M(
        const CellSymbolicData& LeafIndex,
        const long int /*particlesIndexes*/[],
        const ParticlesClass& SourceParticles,
        const unsigned long long inNbParticles,
        LeafClass& LeafCell) const
    {
        RealType* const multipole = &LeafCell[0];

        const std::array<RealType, SpaceIndexType::Dim> cellPosition =
            getLeafCenter(LeafIndex.boxCoord);

        const RealType* const physicalValues = GetPtr(SourceParticles[2]);
        const RealType* const positionsX = GetPtr(SourceParticles[0]);
        const RealType* const positionsY = GetPtr(SourceParticles[1]);

        for (unsigned long long idxPart = 0;
             idxPart < inNbParticles;
             ++idxPart)
        {
            const RealType q =
                coefficient * physicalValues[idxPart];

            const RealType dx =
                (positionsX[idxPart] - cellPosition[0])
                * invSqrtDelta;

            const RealType dy =
                (positionsY[idxPart] - cellPosition[1])
                * invSqrtDelta;

            RealType powersX[OrderSize];
            RealType powersY[OrderSize];

            powersX[0] = RealType(1);
            powersY[0] = RealType(1);

            for (int k = 1; k < OrderSize; ++k)
            {
                powersX[k] = powersX[k - 1] * dx;
                powersY[k] = powersY[k - 1] * dy;
            }

            for (int ax = 0; ax < OrderSize; ++ax)
            {
                for (int ay = 0; ay < OrderSize; ++ay)
                {
                    const int idx =
                        expansionIndex(ax, ay);

                    multipole[idx] +=
                        q
                        * powersX[ax]
                        * powersY[ay]
                        / (factorials[ax]
                           * factorials[ay]);
                }
            }
        }
    }

};

#endif // TBFGAUSSKERNEL_HPP