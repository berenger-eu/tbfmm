#ifndef TBFTESTKERNEL_HPP
#define TBFTESTKERNEL_HPP

#include "tbfglobal.hpp"
#include "kernels/P2P/FP2PLog.hpp"
#include "FSmartPointer.hpp"
#include "FMemUtils.hpp"

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
    // static const int SizeArray = ((P + 2) * (P + 1)) / 2;
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
    }

    TbfLogKernel(const TbfLogKernel &) = default;
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

        const RealType *const physicalValues = SourceParticles[1]; // pointer to contiguously allocated physical parameters (other than coordinates)
        const RealType *const positionsX = SourceParticles[0];     // x-coordinate
        const RealType *const positionsY = SourceParticles[1];     // y-coordinate
        for (unsigned long long idxPart = 0; idxPart < inNbParticles; ++idxPart)
        {
            // const std::array<RealType, SpaceIndexType_T::Dim> relativePosition{
            //     {positionsX[idxPart] - cellPosition[0],
            //      positionsY[idxPart] - cellPosition[1]}};
            // The physical value (charge)
            const RealType q = physicalValues[idxPart];
            int index_l_m = 0; // To construct the index of (l,m) continously
            std::complex<RealType> diff{1.0};
            std::complex<RealType> relativePosition{positionsX[idxPart] - cellPosition[0],
                                                    positionsY[idxPart] - cellPosition[1]};
            for (int l = 0; l <= P; ++l)
            {
                const RealType potential = q / factorials[l];
                for (int m = 0; m <= l; ++m, ++index_l_m)
                {
                    w[index_l_m].real(w[index_l_m].real() + potential * diff.real());
                    w[index_l_m].imag(w[index_l_m].imag() + potential * diff.imag());
                }
                diff *= relativePosition;
            }
        }
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData &inParentIndex,
             const long int inLevel, const CellClassContainer &inLowerCell, CellClass &inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) const
    {
        // Get the translation coef for this level (same for all child)
        // const RealType *const coef = M2MTranslationCoef[inLevel];
        // A buffer to copy the source w allocated once
        std::complex<RealType> source_w[SizeArray];
        const std::array<RealType, SpaceIndexType_T::Dim> parent_center = getBoxCenter(inParentIndex.boxCoord, inLevel);
        // For all children
        for (long int idxChild = 0; idxChild < inNbChildren; ++idxChild)
        {
            const std::array<RealType, SpaceIndexType_T::Dim> child_center = getBoxCenter(spaceIndexSystem.getBoxPosFromIndex(childrenPos[idxChild]), inLevel + 1);
            std::complex<RealType> diff{child_center[0] - parent_center[0],
                                        child_center[1] - parent_center[1]};
            FMemUtils::copyall(source_w, inLowerCell[idxChild].get(), SizeArray);

            std::complex<RealType> target_w[SizeArray];
            // int index_lm = 0;
            // for (int l = 0; l <= P; ++l)
            // {
            //     std::complex<RealType> coef{1.0};
            //     RealType w_lm_real = 0.0;
            //     RealType w_lm_imag = 0.0;
            //     for (int s = l; s >= 0; --s, ++index_lm)
            //     {
            //         w_lm_real += coef.real() / factorials[s] * source_w[s].real();
            //         w_lm_imag += coef.imag() / factorials[s] * source_w[s].imag();
            //         coef *= diff;
            //     }
            //     target_w[index_lm] = std::complex<RealType>(w_lm_real, w_lm_imag);
            // }
            for (int k = 0; k <= P; ++k)
            {
                std::complex<RealType> acc(0.0, 0.0);
                std::complex<RealType> diff_power(1.0, 0.0); // (z_c - z_cp)^0

                for (int s = k; s >= 0; --s)
                {
                    acc += diff_power * source_w[s] / factorials[k - s];
                    diff_power *= diff; // increment power (no std::pow)
                }

                target_w[k] = acc;
            }

            // Sum the result
            FMemUtils::addall(inOutUpperCell, target_w, SizeArray);
        }
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData & /*inTargetIndex*/,
             const long int /*inLevel*/, const CellClassContainer &inInteractingCells, const long int /*neighPos*/[], const long int inNbNeighbors,
             CellClass &inOutCell) const
    {
        for (long int idxNeigh = 0; idxNeigh < inNbNeighbors; ++idxNeigh)
        {
            const auto &neighbor = inInteractingCells[idxNeigh].get();
            inOutCell[0] += neighbor[0];
        }
    }

    template <class CellSymbolicData, class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData & /*inParentIndex*/,
             const long int /*inLevel*/, const CellClass &inUpperCell, CellClassContainer &inOutLowerCell,
             const long int /*childrednPos*/[], const long int inNbChildren) const
    {
        for (long int idxChild = 0; idxChild < inNbChildren; ++idxChild)
        {
            auto &child = inOutLowerCell[idxChild].get();
            child[0] += inUpperCell[0];
        }
    }

    template <class CellSymbolicData, class LeafClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData & /*inLeafIndex*/,
             [[maybe_unused]] const LeafClass &inLeaf, const long int /*particlesIndexes*/[],
             const ParticlesClassValues & /*inOutParticles*/, [[maybe_unused]] ParticlesClassRhs &inOutParticlesRhs,
             const long int inNbParticles) const
    {
        for (int idxPart = 0; idxPart < inNbParticles; ++idxPart)
        {
            // inOutParticlesRhs[0][idxPart] += inLeaf[0];
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
