#ifndef TBFTESTKERNEL_HPP
#define TBFTESTKERNEL_HPP
#include "tbfglobal.hpp"
#include "kernels/P2P/FP2PLog.hpp"
#include "kernels/rotationkernel/FSmartPointer.hpp"
#include "kernels/rotationkernel/FMemUtils.hpp"

#include "utils/tbfperiodicshifter.hpp"

template <class RealType_T, int P, class SpaceIndexType_T = TbfDefaultSpaceIndexType2D<RealType_T>>
class TbfLogKernel
{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

private:
    const RealType PI = RealType(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L);
    const RealType PIDiv2 = RealType(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L / 2.0);
    const RealType PI2 = RealType(3.14159265358979323846264338327950288419716939937510582097494459230781640628620899863L * 2.0);

    //< Size of the data array computed using a suite relation
    static const int SizeArray = P + 1;
    // //< To have P*2 where needed
    static const int P2 = P * 2;

    ///////////////////////////////////////////////////////
    // Object attributes
    ///////////////////////////////////////////////////////

    const SpaceIndexType spaceIndexSystem;

    const RealType boxWidth;                 //< the box width at leaf level
    const int treeHeight;                    //< The height of the tree
    const RealType widthAtLeafLevel;         //< width of box at leaf level
    const RealType widthAtLeafLevelDiv2;     //< width of box at leaf leve div 2
    const std::array<RealType, 2> boxCorner; //< position of the box corner

    RealType factorials[P2 + 1]; //< This contains the factorial until 2*P+1

    ///////////////////////////////////////////////////////
    // Precomputation
    ///////////////////////////////////////////////////////

    /** Compute the factorial from 0 to P*2
     * Then the data is accessible in factorials array:
     * factorials[n] = n! with n <= 2*P
     */
    void precomputeFactorials()
    {
        factorials[0] = 1;
        RealType fidx = 1;
        for (std::size_t idx = 1; idx <= P2; ++idx, ++fidx)
        {
            factorials[idx] = fidx * factorials[idx - 1];
        }
    }

    /** Return the position of a leaf from its tree coordinate
     * This is used only for the leaf
     */
    std::array<RealType, 2> getLeafCenter(const std::array<long int, 2> &coordinate) const
    {
        return std::array<RealType, 2>{boxCorner[0] + (RealType(coordinate[0]) + RealType(.5)) * widthAtLeafLevel,
                                       boxCorner[1] + (RealType(coordinate[1]) + RealType(.5)) * widthAtLeafLevel};
    }

    std::array<RealType, 2> getLeafCenter(const typename SpaceIndexType::IndexType &inIndex) const
    {
        return getLeafCenter(spaceIndexSystem.getBoxPosFromIndex(inIndex));
    }

    std::array<RealType, 2> getBoxCenter(const std::array<long int, 2> &coord, long int level) const
    {
        RealType widthAtLevel = boxWidth * std::pow(RealType(0.5), RealType(level));
        return {
            boxCorner[0] + (RealType(coord[0]) + RealType(.5)) * widthAtLevel,
            boxCorner[1] + (RealType(coord[1]) + RealType(.5)) * widthAtLevel};
    }

public:
    explicit TbfLogKernel(const SpacialConfiguration &inConfiguration) : spaceIndexSystem(inConfiguration),
                                                                         boxWidth(inConfiguration.getBoxWidths()[0]),
                                                                         treeHeight(int(inConfiguration.getTreeHeight())),
                                                                         widthAtLeafLevel(inConfiguration.getLeafWidths()[0]),
                                                                         widthAtLeafLevelDiv2(widthAtLeafLevel / 2),
                                                                         boxCorner(inConfiguration.getBoxCorner())
    {
        precomputeFactorials();
    }

    TbfLogKernel(const TbfLogKernel &other) : spaceIndexSystem(other.spaceIndexSystem),
                                              boxWidth(other.boxWidth),
                                              treeHeight(other.treeHeight),
                                              widthAtLeafLevel(other.widthAtLeafLevel),
                                              widthAtLeafLevelDiv2(other.widthAtLeafLevelDiv2),
                                              boxCorner(other.boxCorner)
    {
        precomputeFactorials();
    }
    TbfLogKernel(TbfLogKernel &&) = default;

    TbfLogKernel &operator=(const TbfLogKernel &) = default;
    TbfLogKernel &operator=(TbfLogKernel &&) = default;

    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData &LeafIndex, const long int /*particlesIndexes*/[],
             const ParticlesClass &SourceParticles, const unsigned long long inNbParticles, LeafClass &LeafCell) const
    {
        // w is the multipole moment
        std::complex<RealType> *const w = &LeafCell[0];

        // Copying the position is faster than using cell position
        const std::array<RealType, SpaceIndexType_T::Dim> cellPosition = getLeafCenter(LeafIndex.boxCoord);

        const RealType *const physicalValues = SourceParticles[2]; // pointer to contiguously allocated physical parameters (other than coordinates)
        const RealType *const positionsX = SourceParticles[0];     // x-coordinate
        const RealType *const positionsY = SourceParticles[1];     // y-coordinate
        for (unsigned long long idxPart = 0; idxPart < inNbParticles; ++idxPart)
        {
            // The physical value (charge)
            const RealType q = physicalValues[idxPart];
            std::complex<RealType> diff{1.0};
            std::complex<RealType> relativePosition{positionsX[idxPart] - cellPosition[0],
                                                    positionsY[idxPart] - cellPosition[1]};
            for (int l = 0; l <= P; ++l)
            {
                const RealType potential = q / factorials[l];
                w[l].real(w[l].real() + potential * diff.real());
                w[l].imag(w[l].imag() + potential * diff.imag());
                diff *= relativePosition;
            }
        }
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData &inParentIndex,
             const long int inLevel, const CellClassContainer &inLowerCell, CellClass &inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) const
    {
        // A buffer to copy the source w allocated once
        std::complex<RealType> source_w[SizeArray];
        const std::array<RealType, SpaceIndexType_T::Dim> parent_center = getBoxCenter(inParentIndex.boxCoord, inLevel);
        // For all children
        for (long int idxChild = 0; idxChild < inNbChildren; ++idxChild)
        {
            const std::array<long int, SpaceIndexType_T::Dim> child_coord =
                spaceIndexSystem.getBoxPosFromIndex(spaceIndexSystem.getChildIndexFromParent(inParentIndex.spaceIndex, childrenPos[idxChild]));

            auto child_center = getBoxCenter(child_coord, inLevel + 1);


            std::complex<RealType> diff{child_center[0] - parent_center[0],
                                        child_center[1] - parent_center[1]};
            FMemUtils::copyall(source_w, inLowerCell[idxChild].get(), SizeArray);

            std::array<std::complex<RealType>, 2 * P + 1> diff_powers;
            diff_powers[0] = 1.0; // diff^{0}
            for (int k = 1; k <= 2 * P; ++k)
            {
                diff_powers[k] = diff_powers[k - 1] * diff;
            }

            std::complex<RealType> target_w[SizeArray];
            for (int k = 0; k <= P; ++k)
            {
                std::complex<RealType> acc(0.0, 0.0);

                for (int s = 0; s <= k; ++s)
                {
                    acc += diff_powers[k - s] * source_w[s] / factorials[k - s];
                }
                target_w[k] = acc;
            }

            // Sum the result
            FMemUtils::addall(inOutUpperCell, target_w, SizeArray);
        }
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData &inTargetIndex,
             const long int inLevel, const CellClassContainer &inInteractingCells, const long int neighPos[], const long int inNbNeighbors,
             CellClass &inOutCell) const
    {
        // To copy the multipole data allocated once
        std::complex<RealType> source_w[SizeArray];
        const std::array<RealType, SpaceIndexType_T::Dim> inTargetIndex_center = getBoxCenter(inTargetIndex.boxCoord, inLevel);
        // For all children
        for (int idxNeigh = 0; idxNeigh < inNbNeighbors; ++idxNeigh)
        {
            const std::array<long int, SpaceIndexType_T::Dim> relative_pos = spaceIndexSystem.getRelativePosFromInteractionIndex(neighPos[idxNeigh]);

            std::array<long int, SpaceIndexType_T::Dim> neigh_coord = {
                inTargetIndex.boxCoord[0] + relative_pos[0],
                inTargetIndex.boxCoord[1] + relative_pos[1]};

            auto neigh_center = getBoxCenter(neigh_coord, inLevel);

            std::complex<RealType> diff{inTargetIndex_center[0] - neigh_center[0],
                                        inTargetIndex_center[1] - neigh_center[1]};
            // Copy multipole data into buffer
            FMemUtils::copyall(source_w, inInteractingCells[idxNeigh].get(), SizeArray);

            std::complex<RealType> diff_inv = 1.0 / diff;

            // Precompute diff powers: diff^{-(1..2P)}
            std::array<std::complex<RealType>, 2 * P + 1> diff_inv_powers;
            diff_inv_powers[0] = 1.0; // diff^{0}
            for (int k = 1; k <= 2 * P; ++k)
            {
                diff_inv_powers[k] = diff_inv_powers[k - 1] * diff_inv;
            }

            // Transfer to u
            std::complex<RealType> target_u[SizeArray];

            if (diff.imag() == 0.0 && diff.real() < 0.0)
            {
                RealType theta = -PI;
                target_u[0] = -source_w[0] * (std::log(std::abs(diff)) + std::complex<RealType>(0.0, theta));
            }
            else
            {
                target_u[0] = -std::log(diff) * source_w[0];
            }

            RealType minus_1_pow_m = -1.0;

            for (int k = 1; k <= P; k++)
            {
                target_u[0] += factorials[k - 1] * diff_inv_powers[k] * source_w[k];
            }

            for (int l = 1; l <= P; ++l)
            {
                std::complex<RealType> sum{0.0};
                for (int n = 0; n <= P; ++n)
                {
                    sum += factorials[l + n - 1] * diff_inv_powers[l + n] * source_w[n];
                }
                target_u[l] = sum * minus_1_pow_m;
                minus_1_pow_m = -minus_1_pow_m;
            }

            // Sum
            FMemUtils::addall(inOutCell, target_u, SizeArray);
        }
    }

    template <class CellSymbolicData, class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData &inParentIndex,
             const long int inLevel, const CellClass &inUpperCell, CellClassContainer &inOutLowerCell,
             const long int childrenPos[], const long int inNbChildren) const
    {
        // To copy the source local allocated once
        std::complex<RealType> source_u[SizeArray];

        const std::array<RealType, SpaceIndexType_T::Dim> parent_center = getBoxCenter(inParentIndex.boxCoord, inLevel);
        // For all children
        for (int idxChild = 0; idxChild < inNbChildren; ++idxChild)
        {

            std::array<long int, SpaceIndexType_T::Dim> child_coord = spaceIndexSystem.getBoxPosFromIndex(spaceIndexSystem.getChildIndexFromParent(inParentIndex.spaceIndex, childrenPos[idxChild]));

            auto child_center = getBoxCenter(child_coord, inLevel + 1);

            std::complex<RealType> diff{child_center[0] - parent_center[0],
                                        child_center[1] - parent_center[1]};
            // Copy the local data into the buffer
            FMemUtils::copyall(source_u, inUpperCell, SizeArray);

            // Precompute diff powers: diff^{-(1..2P)}
            std::array<std::complex<RealType>, 2 * P + 1> diff_powers;
            diff_powers[0] = 1.0; // diff^{0}
            for (int k = 1; k <= 2 * P; ++k)
            {
                diff_powers[k] = diff_powers[k - 1] * diff;
            }

            // Translate
            std::complex<RealType> target_u[SizeArray];

            for (int l = 0; l <= P; ++l)
            {
                std::complex<RealType> sum{0.0};
                for (int m = l; m <= P; ++m)
                {
                    sum += source_u[m] * diff_powers[m - l] / factorials[m - l];
                }
                target_u[l] = sum;
            }

            // Sum in child
            FMemUtils::addall(inOutLowerCell[idxChild].get(), target_u, SizeArray);
        }
    }

    template <class CellSymbolicData, class LeafClass, class ParticlesClass, class ParticlesClassRhs>
    void L2P(const CellSymbolicData &LeafIndex,
             const LeafClass &LeafCell, const long int /*particlesIndexes*/[],
             const ParticlesClass &inOutParticles, ParticlesClassRhs &inOutParticlesRhs,
             const long int inNbParticles) const
    {
        const std::complex<RealType> *const u = &LeafCell[0];
        const std::array<RealType, 2> cellPosition = getLeafCenter(LeafIndex.boxCoord);

        // For all particles in the leaf box
        const RealType *const positionsX = inOutParticles[0];
        const RealType *const positionsY = inOutParticles[1];
        RealType *const potentials = inOutParticlesRhs[0];
        for (int idxPart = 0; idxPart < inNbParticles; ++idxPart)
        {
            std::complex<RealType> diff{positionsX[idxPart] - cellPosition[0],
                                        positionsY[idxPart] - cellPosition[1]};

            std::array<std::complex<RealType>, 2 * P + 1> diff_power;
            diff_power[0] = 1.0;
            for (int i = 1; i <= 2 * P; i++)
            {
                diff_power[i] = diff_power[i - 1] * diff;
            }

            std::complex<RealType> potential = 0.0;
            for (int m = 0; m <= P; ++m)
            {
                potential += diff_power[m] * u[m] / factorials[m];
            }
            // inc potential
            potentials[idxPart] += (potential.real() / PI2);
        }
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData &inNeighborIndex, const long int /*neighborsIndexes*/[],
             const ParticlesClassValues &inNeighbors, ParticlesClassRhs &inNeighborsRhs,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicData &inTargetIndex, const long int /*targetIndexes*/[],
             const ParticlesClassValues &inTargets,
             ParticlesClassRhs &inTargetsRhs, const long int inNbOutParticles,
             [[maybe_unused]] const long arrayIndexSrc) const
    {
        if constexpr (SpaceIndexType::IsPeriodic)
        {
            using PeriodicShifter = typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;
            if (PeriodicShifter::NeedToShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc))
            {
                const auto duplicateSources = PeriodicShifter::DuplicatePositionsAndApplyShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc,
                                                                                               inNeighbors, inNbParticlesNeighbors);
                FP2PLog::template FullMutual<RealType>((duplicateSources), (inNeighborsRhs), inNbParticlesNeighbors,
                                                       (inTargets), (inTargetsRhs), inNbOutParticles);
                PeriodicShifter::FreePositions(duplicateSources);
            }
            else
            {
                FP2PLog::template FullMutual<RealType>((inNeighbors), (inNeighborsRhs), inNbParticlesNeighbors,
                                                       (inTargets), (inTargetsRhs), inNbOutParticles);
            }
        }
        else
        {
            FP2PLog::template FullMutual<RealType>((inNeighbors), (inNeighborsRhs), inNbParticlesNeighbors,
                                                   (inTargets), (inTargetsRhs), inNbOutParticles);
        }
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource &inNeighborIndex, const long int /*neighborsIndexes*/[],
                const ParticlesClassValuesSource &inNeighbors,
                const long int inNbParticlesNeighbors,
                const LeafSymbolicDataTarget &inTargetIndex, const long int /*targetIndexes*/[],
                const ParticlesClassValuesTarget &inTargets,
                ParticlesClassRhs &inTargetsRhs, const long int inNbOutParticles,
                [[maybe_unused]] const long arrayIndexSrc) const
    {
        if constexpr (SpaceIndexType::IsPeriodic)
        {
            using PeriodicShifter = typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;
            if (PeriodicShifter::NeedToShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc))
            {
                const auto duplicateSources = PeriodicShifter::DuplicatePositionsAndApplyShift(inNeighborIndex, inTargetIndex, spaceIndexSystem, arrayIndexSrc,
                                                                                               inNeighbors, inNbParticlesNeighbors);
                FP2PLog::template GenericFullRemote<RealType>((duplicateSources), inNbParticlesNeighbors,
                                                              (inTargets), (inTargetsRhs), inNbOutParticles);
                PeriodicShifter::FreePositions(duplicateSources);
            }
            else
            {
                FP2PLog::template GenericFullRemote<RealType>((inNeighbors), inNbParticlesNeighbors,
                                                              (inTargets), (inTargetsRhs), inNbOutParticles);
            }
        }
        else
        {
            FP2PLog::template GenericFullRemote<RealType>((inNeighbors), inNbParticlesNeighbors,
                                                          (inTargets), (inTargetsRhs), inNbOutParticles);
        }
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData & /*inIndex*/, const long int /*indexes*/[],
                  const ParticlesClassValues &inTargets,
                  ParticlesClassRhs &inTargetsRhs, const long int inNbOutParticles) const
    {
        FP2PLog::template GenericInner<RealType>((inTargets), (inTargetsRhs), inNbOutParticles);
    }
};

#endif
