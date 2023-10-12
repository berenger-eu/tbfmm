// ===================================================================================
// Copyright ScalFmm 2011 INRIA, Olivier Coulaud, Berenger Bramas, Matthias Messner
// olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
// Keep in private GIT
// @SCALFMM_PRIVATE

#ifndef FUNIFKERNEL_HPP
#define FUNIFKERNEL_HPP

#include "FUnifM2LHandler.hpp"
#include "FAbstractUnifKernel.hpp"
#include "kernels/P2P/FP2PR.hpp"

#include "utils/tbfperiodicshifter.hpp"

#include "tbfglobal.hpp"


/**
 * @author Pierre Blanchard (pierre.blanchard@inria.fr)
 * @class FUnifKernel
 * @brief
 * Please read the license
 *
 * This kernels implement the Lagrange interpolation based FMM operators. It
 * implements all interfaces (P2P,P2M,M2M,M2L,L2L,L2P) which are required by
 * the FFmmAlgorithm and FFmmAlgorithmThread.
 *
 * @tparam CellClass Type of cell
 * @tparam ContainerClass Type of container to store particles
 * @tparam MatrixKernelClass Type of matrix kernel function
 * @tparam ORDER Lagrange interpolation order
 */
template < class RealType_T, class MatrixKernelClass, int ORDER, int Dim = 3,
           class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class FUnifKernel
  : public FAbstractUnifKernel<RealType_T, MatrixKernelClass, ORDER, Dim, SpaceIndexType_T>
{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

private:
    // private types
    using M2LHandlerClass = FUnifM2LHandler<RealType, ORDER,MatrixKernelClass::Type>;

    // using from
    using AbstractBaseClass = FAbstractUnifKernel< RealType, MatrixKernelClass, ORDER, Dim, SpaceIndexType>;

    /// Needed for P2P and M2L operators
    const MatrixKernelClass *const MatrixKernel;

    /// Needed for M2L operator
    const M2LHandlerClass M2LHandler;

    /// Leaf level separation criterion
    const int LeafLevelSeparationCriterion;

public:
    /**
    * The constructor initializes all constant attributes and it reads the
    * precomputed and compressed M2L operators from a binary file (an
    * runtime_error is thrown if the required file is not valid).
    */
    FUnifKernel(const SpacialConfiguration& inConfiguration,
                const MatrixKernelClass *const inMatrixKernel,
                const int inLeafLevelSeparationCriterion = 1)
    : FAbstractUnifKernel< RealType, MatrixKernelClass, ORDER, Dim, SpaceIndexType>(inConfiguration),
      MatrixKernel(inMatrixKernel),
      M2LHandler(MatrixKernel,
                 int(inConfiguration.getTreeHeight()),
                 inConfiguration.getBoxWidths()[0],
                 inLeafLevelSeparationCriterion),
      LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
    { }


    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(const CellSymbolicData& LeafIndex,  const long int /*particlesIndexes*/[],
             const ParticlesClass& SourceParticles, const long int inNbParticles, LeafClass& LeafCell) const {
        const auto LeafCellCenter = AbstractBaseClass::getLeafCellCenter(LeafIndex.boxCoord);
        // 1) apply Sy
        AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  LeafCell.multipole_exp, std::forward<const ParticlesClass>(SourceParticles), inNbParticles);
        // 2) apply Discrete Fourier Transform
        M2LHandler.applyZeroPaddingAndDFT(LeafCell.multipole_exp,
                                          LeafCell.transformed_multipole_exp);
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2M(const CellSymbolicData& /*inParentIndex*/,
             const long int /*inLevel*/, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
             const long int childrenPos[], const long int inNbChildren) const {
        // 1) apply Sy
        //FBlas::scal(AbstractBaseClass::nnodes, RealType(0.), ParentCell->getMultipole(idxRhs));
        for (unsigned int idxChild=0 ; idxChild < inNbChildren ; ++idxChild){
            AbstractBaseClass::Interpolator->applyM2M(int(childrenPos[idxChild]), inLowerCell[idxChild].get().multipole_exp,
                                                      inOutUpperCell.multipole_exp);
        }
        // 2) Apply Discete Fourier Transform
        M2LHandler.applyZeroPaddingAndDFT(inOutUpperCell.multipole_exp,
                                          inOutUpperCell.transformed_multipole_exp);
    }


    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2L(const CellSymbolicData& /*inTargetIndex*/,
             const long int inLevel, const CellClassContainer& inInteractingCells, const long int neighPos[], const long int inNbNeighbors,
             CellClass& inOutCell) {
        const RealType CellWidth(AbstractBaseClass::BoxWidth / RealType(FMath::pow(2, int(inLevel))));
        const RealType scale(MatrixKernel->getScaleFactor(CellWidth));

        assert(inNbNeighbors == int(inInteractingCells.size()));

        for(long int idxExistingNeigh = 0 ; idxExistingNeigh < inNbNeighbors ; ++idxExistingNeigh){
            const int idxNeigh = int(neighPos[idxExistingNeigh]);
            M2LHandler.applyFC(idxNeigh, int(inLevel), scale,
                               inInteractingCells[idxExistingNeigh].get().transformed_multipole_exp,
                               inOutCell.transformed_local_exp);
        }
    }


    template <class CellSymbolicData, class CellClass, class CellClassContainer>
    void L2L(const CellSymbolicData& /*inParentIndex*/,
             const long int /*inLevel*/, const CellClass& inUpperCell, CellClassContainer& inOutLowerCell,
             const long int childrenPos[], const long int inNbChildren) {
        // 1) Apply Inverse Discete Fourier Transform
        RealType localExp[AbstractBaseClass::nnodes] = {0};
        M2LHandler.unapplyZeroPaddingAndDFT(inUpperCell.transformed_local_exp,
                                            localExp);
        FBlas::add(AbstractBaseClass::nnodes,const_cast<RealType*>(inUpperCell.local_exp),localExp);

        // 2) apply Sx
        for (unsigned int idxChild=0; idxChild < inNbChildren; ++idxChild){
            AbstractBaseClass::Interpolator->applyL2L(int(childrenPos[idxChild]), localExp,
                                                      inOutLowerCell[idxChild].get().local_exp);
        }
    }

    template <class CellSymbolicData, class LeafClass, class ParticlesClass, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& LeafIndex,
             const LeafClass& LeafCell,  const long int /*particlesIndexes*/[],
             const ParticlesClass& inOutParticles, ParticlesClassRhs& inOutParticlesRhs,
             const long int inNbParticles) {
        const std::array<RealType, Dim> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafIndex.boxCoord));

        RealType localExp[AbstractBaseClass::nnodes] = {0};

        // 1)  Apply Inverse Discete Fourier Transform
        M2LHandler.unapplyZeroPaddingAndDFT(LeafCell.transformed_local_exp,
                                            localExp);
        FBlas::add(AbstractBaseClass::nnodes,const_cast<RealType*>(LeafCell.local_exp),localExp);

        // 2.a) apply Sx
        AbstractBaseClass::Interpolator->applyL2P(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  localExp, std::forward<const ParticlesClass>(inOutParticles),
                                                  std::forward<ParticlesClassRhs>(inOutParticlesRhs), inNbParticles);

        // 2.b) apply Px (grad Sx)
        AbstractBaseClass::Interpolator->applyL2PGradient(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                          localExp, std::forward<const ParticlesClass>(inOutParticles),
                                                          std::forward<ParticlesClassRhs>(inOutParticlesRhs), inNbParticles);
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData& inNeighborIndex, const long int /*neighborsIndexes*/[],
             const ParticlesClassValues& inNeighbors, ParticlesClassRhs& inNeighborsRhs, const long int inNbParticlesNeighbors,
             const LeafSymbolicData& inTargetIndex,  const long int /*targetIndexes*/[],
             const ParticlesClassValues& inTargets,
             ParticlesClassRhs& inTargetsRhs, const long int inNbOutParticles,
             [[maybe_unused]] const long arrayIndexSrc) const {
        if constexpr(SpaceIndexType::IsPeriodic){
            using PeriodicShifter = typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;
            if(PeriodicShifter::NeedToShift(inNeighborIndex, inTargetIndex, AbstractBaseClass::spaceIndexSystem, arrayIndexSrc)){
                const auto duplicateSources = PeriodicShifter::DuplicatePositionsAndApplyShift(inNeighborIndex, inTargetIndex, AbstractBaseClass::spaceIndexSystem, arrayIndexSrc,
                                                                            inNeighbors, inNbParticlesNeighbors);
                FP2PR::template FullMutual<RealType> ((duplicateSources),(inNeighborsRhs), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
                PeriodicShifter::FreePositions(duplicateSources);
            }
            else{
                FP2PR::template FullMutual<RealType> ((inNeighbors),(inNeighborsRhs), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
            }
        }
        else{
            FP2PR::template FullMutual<RealType> ((inNeighbors),(inNeighborsRhs), inNbParticlesNeighbors,
                                                                                       (inTargets), (inTargetsRhs), inNbOutParticles);
        }
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& inNeighborIndex, const long int /*neighborsIndexes*/[],
             const ParticlesClassValuesSource& inNeighbors,
             const long int inNbParticlesNeighbors,
             const LeafSymbolicDataTarget& inTargetIndex, const long int /*targetIndexes*/[],
             const ParticlesClassValuesTarget& inTargets,
             ParticlesClassRhs& inTargetsRhs, const long int inNbOutParticles,
             [[maybe_unused]] const long arrayIndexSrc) const {
        if constexpr(SpaceIndexType::IsPeriodic){
            using PeriodicShifter = typename TbfPeriodicShifter<RealType, SpaceIndexType>::Neighbor;
            if(PeriodicShifter::NeedToShift(inNeighborIndex, inTargetIndex, AbstractBaseClass::spaceIndexSystem, arrayIndexSrc)){
                const auto duplicateSources = PeriodicShifter::DuplicatePositionsAndApplyShift(inNeighborIndex, inTargetIndex, AbstractBaseClass::spaceIndexSystem, arrayIndexSrc,
                                                                            inNeighbors, inNbParticlesNeighbors);
                FP2PR::template GenericFullRemote<RealType> ((duplicateSources), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
                PeriodicShifter::FreePositions(duplicateSources);
            }
            else{
                FP2PR::template GenericFullRemote<RealType> ((inNeighbors), inNbParticlesNeighbors,
                                                             (inTargets), (inTargetsRhs), inNbOutParticles);
            }
        }
        else{
            FP2PR::template GenericFullRemote<RealType> ((inNeighbors), inNbParticlesNeighbors,
                                                                                       (inTargets), (inTargetsRhs), inNbOutParticles);
        }
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& /*inIndex*/, const long int /*targetIndexes*/[],
                  const ParticlesClassValues& inTargets,
                  ParticlesClassRhs& inTargetsRhs, const long int inNbOutParticles) const {
        FP2PR::template GenericInner<RealType>((inTargets),(inTargetsRhs), inNbOutParticles);
    }
};


#endif //FUNIFKERNEL_HPP

// [--END--]
