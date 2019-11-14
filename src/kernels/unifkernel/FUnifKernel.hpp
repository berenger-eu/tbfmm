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
template < class RealType_T, class MatrixKernelClass, int ORDER, int Dim = 3, int NVALS = 1,
           class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class FUnifKernel
  : public FAbstractUnifKernel<RealType_T, MatrixKernelClass, ORDER, Dim, NVALS>
{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

private:
    // private types
    using M2LHandlerClass = FUnifM2LHandler<RealType, ORDER,MatrixKernelClass::Type>;

    // using from
    using AbstractBaseClass = FAbstractUnifKernel< RealType, MatrixKernelClass, ORDER, Dim, NVALS>;

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
    : FAbstractUnifKernel< RealType, MatrixKernelClass, ORDER, Dim,  NVALS>(inConfiguration),
      MatrixKernel(inMatrixKernel),
      M2LHandler(MatrixKernel,
                 inConfiguration.getTreeHeight(),
                 inConfiguration.getBoxWidth(),
                 inLeafLevelSeparationCriterion),
      LeafLevelSeparationCriterion(inLeafLevelSeparationCriterion)
    { }

#ifdef ZERAZERAZERAZER
    template <class ParticlesClass, class LeafClass>
    void P2M(const ParticlesClass&& SourceParticles, const long int inNbParticles, LeafClass& LeafCell) const {
    {
        const auto LeafCellCenter = AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate());
        // 1) apply Sy
        AbstractBaseClass::Interpolator->applyP2M(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  LeafCell->getMultipole(0), SourceParticles, inNbParticles);

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            // 2) apply Discrete Fourier Transform
            M2LHandler.applyZeroPaddingAndDFT(LeafCell.getMultipole(idxRhs),
                                              LeafCell.getTransformedMultipole(idxRhs));

        }
    }


        template <class CellClassContainer, class CellClass>
        void M2M(const long int /*inLevel*/, const CellClassContainer& inLowerCell, CellClass& inOutUpperCell,
                 const long int childrenPos[], const int inNbChildren) const {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            // 1) apply Sy
            //FBlas::scal(AbstractBaseClass::nnodes, RealType(0.), ParentCell->getMultipole(idxRhs));
            for (unsigned int idxChild=0 ; idxChild < inNbChildren ; ++idxChild){
                AbstractBaseClass::Interpolator->applyM2M(childrenPos[idxChild], inLowerCell->getMultipole(idxRhs),
                                                          ParentCell->getMultipole(idxRhs));
            }
            // 2) Apply Discete Fourier Transform
            M2LHandler.applyZeroPaddingAndDFT(ParentCell->getMultipole(idxRhs), 
                                              ParentCell->getTransformedMultipole(idxRhs));
        }
    }


    void M2L(CellClass* const  TargetCell, const CellClass* SourceCells[],
             const int neighborPositions[], const int inSize, const int TreeLevel)  override {
        const RealType CellWidth(AbstractBaseClass::BoxWidth / RealType(FMath::pow(2, TreeLevel)));
        const RealType scale(MatrixKernel->getScaleFactor(CellWidth));

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){
            FComplex<RealType> *const TransformedLocalExpansion = TargetCell->getTransformedLocal(idxRhs);

            for(int idxExistingNeigh = 0 ; idxExistingNeigh < inSize ; ++idxExistingNeigh){
                const int idxNeigh = neighborPositions[idxExistingNeigh];
                M2LHandler.applyFC(idxNeigh, TreeLevel, scale,
                                   SourceCells[idxExistingNeigh]->getTransformedMultipole(idxRhs),
                                   TransformedLocalExpansion);
            }
        }
    }


    void L2L(const CellClass* const  ParentCell,
             CellClass*  *const  ChildCells,
             const int /*TreeLevel*/)
    {
        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            // 1) Apply Inverse Discete Fourier Transform
            RealType localExp[AbstractBaseClass::nnodes];
            M2LHandler.unapplyZeroPaddingAndDFT(ParentCell->getTransformedLocal(idxRhs),
                                                localExp);
            FBlas::add(AbstractBaseClass::nnodes,const_cast<RealType*>(ParentCell->getLocal(idxRhs)),localExp);

            // 2) apply Sx
            for (unsigned int ChildIndex=0; ChildIndex < 8; ++ChildIndex){
                if (ChildCells[ChildIndex]){
                    AbstractBaseClass::Interpolator->applyL2L(ChildIndex, localExp, ChildCells[ChildIndex]->getLocal(idxRhs));
                }
            }
        }
    }

    template <class LeafClass, class ParticlesClass>
    void L2P(const LeafClass& LeafCell,
             ParticlesClass&& TargetParticles, const long int inNbParticles)
    {
        const std::array<RealType, Dim> LeafCellCenter(AbstractBaseClass::getLeafCellCenter(LeafCell->getCoordinate()));

        RealType localExp[NVALS*AbstractBaseClass::nnodes];

        for(int idxRhs = 0 ; idxRhs < NVALS ; ++idxRhs){

            // 1)  Apply Inverse Discete Fourier Transform
            M2LHandler.unapplyZeroPaddingAndDFT(LeafCell->getTransformedLocal(idxRhs), 
                                                localExp + idxRhs*AbstractBaseClass::nnodes);
            FBlas::add(AbstractBaseClass::nnodes,const_cast<RealType*>(LeafCell->getLocal(idxRhs)),localExp + idxRhs*AbstractBaseClass::nnodes);

        }

        // 2.a) apply Sx
        AbstractBaseClass::Interpolator->applyL2P(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                  localExp, TargetParticles, inNbParticles);

        // 2.b) apply Px (grad Sx)
        AbstractBaseClass::Interpolator->applyL2PGradient(LeafCellCenter, AbstractBaseClass::BoxWidthLeaf,
                                                          localExp, TargetParticles, inNbParticles);


    }

    template <class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const ParticlesClassValues&& inNeighbors, const long int inNbParticlesNeighbors,
             const long int inNeighborPos, ParticlesClassRhs&& inTargets, const long int inNbOutParticles) const {
        DirectInteractionComputer<RealType, MatrixKernelClass::NCMP, NVALS>::P2PRemote(inTargets, MatrixKernel, inNbOutParticles);
    }

    template <class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const ParticlesClassValues&& inNeighbors,
                  ParticlesClassRhs&& inTargets, const long int inNbOutParticles) const {
        DirectInteractionComputer<RealType, MatrixKernelClass::NCMP, NVALS>::P2PInner(inTargets, MatrixKernel, inNbOutParticles);
    }
#endif
};


#endif //FUNIFKERNEL_HPP

// [--END--]
