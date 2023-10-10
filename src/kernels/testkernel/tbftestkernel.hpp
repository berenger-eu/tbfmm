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

    TbfTestKernel(const TbfTestKernel&) = default;
    TbfTestKernel(TbfTestKernel&&) = default;

    TbfTestKernel& operator=(const TbfTestKernel&) = default;
    TbfTestKernel& operator=(TbfTestKernel&&) = default;

    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData& /*inLeafIndex*/,  const long int /*particlesIndexes*/[],
             const ParticlesClass& /*inParticles*/, const long int inNbParticles, LeafClass& inOutLeaf) const {
        inOutLeaf[0] += inNbParticles;
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int /*childrenPos*/[], const long int inNbChildren) const {
        for(long int idxChild = 0 ; idxChild < inNbChildren ; ++idxChild){
            const auto& child = inLowerCell[idxChild].get();
            inOutUpperCell[0] += child[0];
        }
    }

    template <class CellSymbolicData,class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData& /*inTargetIndex*/,
             const long int /*inLevel*/, const CellClassContainer& inInteractingCells, const long int /*neighPos*/[], const long int inNbNeighbors,
             CellClass& inOutCell) const {
        for(long int idxNeigh = 0 ; idxNeigh < inNbNeighbors ; ++idxNeigh){
            const auto& neighbor = inInteractingCells[idxNeigh].get();
            inOutCell[0] += neighbor[0];
        }
    }

    template <class CellSymbolicData,class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData& /*inParentIndex*/,
             const long int /*inLevel*/, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int /*childrednPos*/[], const long int inNbChildren) const {
        for(long int idxChild = 0 ; idxChild < inNbChildren ; ++idxChild){
            auto& child = inOutLowerCell[idxChild].get();
            child[0] += inUpperCell[0];
        }
    }

    template <class CellSymbolicData,class LeafClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& /*inLeafIndex*/,
             const LeafClass& inLeaf,  const long int /*particlesIndexes*/[],
             const ParticlesClassValues& /*inOutParticles*/, ParticlesClassRhs& inOutParticlesRhs,
             const long int inNbParticles) const {
        for(int idxPart = 0 ; idxPart < inNbParticles ; ++idxPart){
            inOutParticlesRhs[0][idxPart] += inLeaf[0];
        }
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData& /*inNeighborIndex*/, const long int /*neighborsIndexes*/[],
             const ParticlesClassValues& /*inParticlesNeighbors*/, ParticlesClassRhs& inParticlesNeighborsRhs,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicData& /*inParticlesIndex*/, const long int /*targetIndexes*/[],
             const ParticlesClassValues& /*inOutParticles*/,
             ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
             const long /*arrayIndexSrc*/) const {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticlesRhs[0][idxPart] += inNbParticlesNeighbors;
        }
        for(int idxPart = 0 ; idxPart < inNbParticlesNeighbors ; ++idxPart){
            inParticlesNeighborsRhs[0][idxPart] += inNbOutParticles;
        }
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& /*inNeighborIndex*/, const long int /*neighborsIndexes*/[],
             const ParticlesClassValuesSource& /*inParticlesNeighbors*/,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicDataTarget& /*inParticlesIndex*/, const long int /*targetIndexes*/[],
             const ParticlesClassValuesTarget& /*inOutParticles*/,
             ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
             const long /*arrayIndexSrc*/) const {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticlesRhs[0][idxPart] += inNbParticlesNeighbors;
        }
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& /*inLeafIndex*/, const long int /*targetIndexes*/[],
                  const ParticlesClassValues& /*inOutParticles*/,
                  ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles) const {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticlesRhs[0][idxPart] += inNbOutParticles - 1;
        }
    }

    #ifdef __NVCC__
    static constexpr bool CpuP2P = true;
    static constexpr bool CudaP2P = true;

    struct CudaKernelData{ bool notUsed; };

    void initCudaKernelData(const cudaStream_t& /*inStream*/){
    }

    auto getCudaKernelData(){
        return CudaKernelData();
    }

    void releaseCudaKernelData(const cudaStream_t& /*inStream*/){
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    __device__ static void P2PTsmCuda(const CudaKernelData& /*cudaKernelData*/,
                                      const LeafSymbolicDataSource& /*inNeighborIndex*/, const long int /*neighborsIndexes*/[],
                const ParticlesClassValuesSource& /*inParticlesNeighbors*/,
                const long int inNbParticlesNeighbors,
                const LeafSymbolicDataTarget& /*inParticlesIndex*/, const long int /*targetIndexes*/[],
                const ParticlesClassValuesTarget& /*inOutParticles*/,
                ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles,
                const long /*arrayIndexSrc*/) /*const*/ {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticlesRhs[0][idxPart] += inNbParticlesNeighbors;
        }
    }

    template <class LeafSymbolicData,class ParticlesClassValues, class ParticlesClassRhs>
    __device__ static void P2PInnerCuda(const CudaKernelData& /*cudaKernelData*/,
                                        const LeafSymbolicData& /*inLeafIndex*/, const long int /*targetIndexes*/[],
                  const ParticlesClassValues& /*inOutParticles*/,
                  ParticlesClassRhs& inOutParticlesRhs, const long int inNbOutParticles) /*const*/ {
        for(int idxPart = 0 ; idxPart < inNbOutParticles ; ++idxPart){
            inOutParticlesRhs[0][idxPart] += inNbOutParticles - 1;
        }
    }
    #endif
};

#endif
