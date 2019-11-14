// ===================================================================================
// Copyright ScalFmm 2011 INRIA, Olivier Coulaud, Bérenger Bramas, Matthias Messner
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
#ifndef FP2PMULTIRHS_HPP
#define FP2PMULTIRHS_HPP

namespace FP2P {

  /*
   * FullMutualMultiRhs (generic version)
   */
  template <class FReal, class ContainerClass, typename MatrixKernelClass>
  inline void FullMutualMultiRhs(ContainerClass* const  inTargets, ContainerClass* const inNeighbors[],
                                 const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
    const int Dim = 3;
    const long int nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValuesArray();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesXArray();
    FReal*const targetsForcesY = inTargets->getForcesYArray();
    FReal*const targetsForcesZ = inTargets->getForcesZArray();
    FReal*const targetsPotentials = inTargets->getPotentialsArray();
    const int NVALS = inTargets->getNVALS();
    const long int targetsLD  = inTargets->getLeadingDimension();

    for(long int idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        for(long int idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            if( inNeighbors[idxNeighbors] ){
                const long int nbParticlesSources = inNeighbors[idxNeighbors]->getNbParticles();
                const FReal*const sourcesPhysicalValues = inNeighbors[idxNeighbors]->getPhysicalValuesArray();
                const FReal*const sourcesX = inNeighbors[idxNeighbors]->getPositions()[0];
                const FReal*const sourcesY = inNeighbors[idxNeighbors]->getPositions()[1];
                const FReal*const sourcesZ = inNeighbors[idxNeighbors]->getPositions()[2];
                FReal*const sourcesForcesX = inNeighbors[idxNeighbors]->getForcesXArray();
                FReal*const sourcesForcesY = inNeighbors[idxNeighbors]->getForcesYArray();
                FReal*const sourcesForcesZ = inNeighbors[idxNeighbors]->getForcesZArray();
                FReal*const sourcesPotentials = inNeighbors[idxNeighbors]->getPotentialsArray();
                const long int sourcesLD  = inNeighbors[idxNeighbors]->getLeadingDimension();

                for(long int idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){

                    // Compute kernel of interaction and its derivative
                    const std::array<FReal, Dim> sourcePoint(sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource]);
                    const std::array<FReal, Dim> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
                    FReal Kxy[1];
                    FReal dKxy[3];
                    MatrixKernel->evaluateBlockAndDerivative(sourcePoint[0],sourcePoint[1],sourcePoint[2],
                                                             targetPoint[0],targetPoint[1],targetPoint[2],Kxy,dKxy);

                    for(int idxVals = 0 ; idxVals < NVALS ; ++idxVals){
                      
                        const long int idxTargetValue = idxVals*targetsLD+idxTarget;
                        const long int idxSourceValue = idxVals*sourcesLD+idxSource;
                        
                        FReal coef = (targetsPhysicalValues[idxTargetValue] * sourcesPhysicalValues[idxSourceValue]);
                        
                        targetsForcesX[idxTargetValue] += dKxy[0] * coef;
                        targetsForcesY[idxTargetValue] += dKxy[1] * coef;
                        targetsForcesZ[idxTargetValue] += dKxy[2] * coef;
                        targetsPotentials[idxTargetValue] += ( Kxy[0] * sourcesPhysicalValues[idxSourceValue] );
                        
                        sourcesForcesX[idxSourceValue] -= dKxy[0] * coef;
                        sourcesForcesY[idxSourceValue] -= dKxy[1] * coef;
                        sourcesForcesZ[idxSourceValue] -= dKxy[2] * coef;
                        sourcesPotentials[idxSourceValue] += ( Kxy[0] * targetsPhysicalValues[idxTargetValue] );

                    } // NVALS

                }
            }
        }
    }
  }

  template <class FReal, class ContainerClass, typename MatrixKernelClass>
  inline void InnerMultiRhs(ContainerClass* const  inTargets, const MatrixKernelClass *const MatrixKernel){
    const int Dim = 3;
    const long int nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValuesArray();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesXArray();
    FReal*const targetsForcesY = inTargets->getForcesYArray();
    FReal*const targetsForcesZ = inTargets->getForcesZArray();
    FReal*const targetsPotentials = inTargets->getPotentialsArray();
    const int NVALS = inTargets->getNVALS();
    const long int targetsLD  = inTargets->getLeadingDimension();

    for(long int idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        for(long int idxSource = idxTarget + 1 ; idxSource < nbParticlesTargets ; ++idxSource){

            // Compute kernel of interaction...
            const std::array<FReal, Dim> sourcePoint(targetsX[idxSource],targetsY[idxSource],targetsZ[idxSource]);
            const std::array<FReal, Dim> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
            FReal Kxy[1];
            FReal dKxy[3];
            MatrixKernel->evaluateBlockAndDerivative(sourcePoint[0],sourcePoint[1],sourcePoint[2],
                                                     targetPoint[0],targetPoint[1],targetPoint[2],Kxy,dKxy);

            for(int idxVals = 0 ; idxVals < NVALS ; ++idxVals){
                      
                const long int idxTargetValue = idxVals*targetsLD+idxTarget;
                const long int idxSourceValue = idxVals*targetsLD+idxSource;
                
                FReal coef = (targetsPhysicalValues[idxTargetValue] * targetsPhysicalValues[idxSourceValue]);
                
                targetsForcesX[idxTargetValue] += dKxy[0] * coef;
                targetsForcesY[idxTargetValue] += dKxy[1] * coef;
                targetsForcesZ[idxTargetValue] += dKxy[2] * coef;
                targetsPotentials[idxTargetValue] += ( Kxy[0] * targetsPhysicalValues[idxSourceValue] );
                
                targetsForcesX[idxSourceValue] -= dKxy[0] * coef;
                targetsForcesY[idxSourceValue] -= dKxy[1] * coef;
                targetsForcesZ[idxSourceValue] -= dKxy[2] * coef;
                targetsPotentials[idxSourceValue] += ( Kxy[0] * targetsPhysicalValues[idxTargetValue] );

            }// NVALS

        }
    }
}


/**
   * FullRemoteMultiRhs (generic version)
   */
template <class FReal, class ContainerClass, typename MatrixKernelClass>
inline void FullRemoteMultiRhs(ContainerClass* const  inTargets, ContainerClass* const inNeighbors[],
                       const int limiteNeighbors, const MatrixKernelClass *const MatrixKernel){
    const int Dim = 3;
    const long int nbParticlesTargets = inTargets->getNbParticles();
    const FReal*const targetsPhysicalValues = inTargets->getPhysicalValuesArray();
    const FReal*const targetsX = inTargets->getPositions()[0];
    const FReal*const targetsY = inTargets->getPositions()[1];
    const FReal*const targetsZ = inTargets->getPositions()[2];
    FReal*const targetsForcesX = inTargets->getForcesXArray();
    FReal*const targetsForcesY = inTargets->getForcesYArray();
    FReal*const targetsForcesZ = inTargets->getForcesZArray();
    FReal*const targetsPotentials = inTargets->getPotentialsArray();
    const int NVALS = inTargets->getNVALS();
    const long int targetsLD  = inTargets->getLeadingDimension();

    for(long int idxNeighbors = 0 ; idxNeighbors < limiteNeighbors ; ++idxNeighbors){
        for(long int idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
            if( inNeighbors[idxNeighbors] ){
                const long int nbParticlesSources = inNeighbors[idxNeighbors]->getNbParticles();
                const FReal*const sourcesPhysicalValues = inNeighbors[idxNeighbors]->getPhysicalValuesArray();
                const FReal*const sourcesX = inNeighbors[idxNeighbors]->getPositions()[0];
                const FReal*const sourcesY = inNeighbors[idxNeighbors]->getPositions()[1];
                const FReal*const sourcesZ = inNeighbors[idxNeighbors]->getPositions()[2];
                const long int sourcesLD  = inNeighbors[idxNeighbors]->getLeadingDimension();

                for(long int idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){

                    // Compute kernel of interaction...
                    const std::array<FReal, Dim> sourcePoint(sourcesX[idxSource],sourcesY[idxSource],sourcesZ[idxSource]);
                    const std::array<FReal, Dim> targetPoint(targetsX[idxTarget],targetsY[idxTarget],targetsZ[idxTarget]);
                    FReal Kxy[1];
                    FReal dKxy[3];
                    MatrixKernel->evaluateBlockAndDerivative(sourcePoint[0],sourcePoint[1],sourcePoint[2],
                                                             targetPoint[0],targetPoint[1],targetPoint[2],Kxy,dKxy);

                    for(int idxVals = 0 ; idxVals < NVALS ; ++idxVals){

                        const long int idxTargetValue = idxVals*targetsLD+idxTarget;
                        const long int idxSourceValue = idxVals*sourcesLD+idxSource;
                        
                        FReal coef = (targetsPhysicalValues[idxTargetValue] * sourcesPhysicalValues[idxSourceValue]);
                        
                        targetsForcesX[idxTargetValue] += dKxy[0] * coef;
                        targetsForcesY[idxTargetValue] += dKxy[1] * coef;
                        targetsForcesZ[idxTargetValue] += dKxy[2] * coef;
                        targetsPotentials[idxTargetValue] += ( Kxy[0] * sourcesPhysicalValues[idxSourceValue] );

                    } // NVALS

                }
            }
        }
    }
}

}

#endif // FP2PMULTIRHS_HPP
