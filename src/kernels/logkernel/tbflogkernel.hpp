#ifndef TBFTESTKERNEL_HPP
#define TBFTESTKERNEL_HPP

#include "tbfglobal.hpp"
#include "kernels/P2P/FP2PLog.hpp"

#include "utils/tbfperiodicshifter.hpp"

template <class RealType_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType2D<RealType_T>>
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
    // //< To have P*2 where needed
    // static const int P2 = P * 2;

    ///////////////////////////////////////////////////////
    // Object attributes
    ///////////////////////////////////////////////////////

    const SpaceIndexType spaceIndexSystem;

    const RealType boxWidth;                 //< the box width at leaf level
    const int treeHeight;                    //< The height of the tree
    const RealType widthAtLeafLevel;         //< width of box at leaf level
    const RealType widthAtLeafLevelDiv2;     //< width of box at leaf leve div 2
    const std::array<RealType, 2> boxCorner; //< position of the box corner
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
    void P2M(const CellSymbolicData & /*inLeafIndex*/, const long int /*particlesIndexes*/[],
             const ParticlesClass & /*inParticles*/, const long int inNbParticles, LeafClass &inOutLeaf) const
    {
        inOutLeaf[0] += inNbParticles;
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData & /*inCellIndex*/,
             const long int /*inLevel*/, const CellClassContainer &inLowerCell, CellClass &inOutUpperCell,
             const long int /*childrenPos*/[], const long int inNbChildren) const
    {
        for (long int idxChild = 0; idxChild < inNbChildren; ++idxChild)
        {
            const auto &child = inLowerCell[idxChild].get();
            inOutUpperCell[0] += child[0];
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
             const LeafClass &inLeaf, const long int /*particlesIndexes*/[],
             const ParticlesClassValues & /*inOutParticles*/, ParticlesClassRhs &inOutParticlesRhs,
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

#ifdef __NVCC__
    static constexpr bool CpuP2P = true;
    static constexpr bool CudaP2P = true;

    struct CudaKernelData
    {
        bool notUsed;
    };

    void initCudaKernelData(const cudaStream_t & /*inStream*/)
    {
    }

    auto getCudaKernelData()
    {
        return CudaKernelData();
    }

    void releaseCudaKernelData(const cudaStream_t & /*inStream*/)
    {
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    __device__ static void P2PTsmCuda(const CudaKernelData & /*cudaKernelData*/,
                                      const LeafSymbolicDataSource &inNeighborIndex, const long int /*neighborsIndexes*/[],
                                      const ParticlesClassValuesSource &inNeighbors,
                                      const long int inNbParticlesNeighbors,
                                      const LeafSymbolicDataTarget &inTargetIndex, const long int /*targetIndexes*/[],
                                      const ParticlesClassValuesTarget &inTargets,
                                      ParticlesClassRhs &inTargetsRhs, const long int inNbOutParticles,
                                      [[maybe_unused]] const long arrayIndexSrc) /*const*/
    {
        static_assert(SpaceIndexType::IsPeriodic == false);
        TbfP2PCuda::template GenericFullRemote<RealType>((inNeighbors), inNbParticlesNeighbors,
                                                         (inTargets), (inTargetsRhs), inNbOutParticles);
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    __device__ static void P2PInnerCuda(const CudaKernelData & /*cudaKernelData*/,
                                        const LeafSymbolicData & /*inIndex*/, const long int /*indexes*/[],
                                        const ParticlesClassValues &inTargets,
                                        ParticlesClassRhs &inTargetsRhs, const long int inNbOutParticles) /*const*/
    {
        TbfP2PCuda::template GenericInner<RealType>((inTargets), (inTargetsRhs), inNbOutParticles);
    }
#endif
};

#endif
