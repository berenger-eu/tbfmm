#ifndef FINTERPP2PKERNELS_HPP
#define FINTERPP2PKERNELS_HPP


#include "FP2P.hpp"
#include "FP2PR.hpp"

///////////////////////////////////////////////////////
// P2P Wrappers
///////////////////////////////////////////////////////

template <class FReal, int NCMP>
struct DirectInteractionComputer
{
    template <typename ContainerClass, typename ContainerClassRhs, typename MatrixKernelClass>
    static void P2P( const ContainerClass& inNeigh, const long int inNeighNbParticles,
                     const MatrixKernelClass *const MatrixKernel,
                     const ContainerClass& inTarget, ContainerClassRhs& inTargetRhs,
                     const long int inTargetNbParticles){
        FP2P::FullMutualKIJ<FReal, ContainerClass, MatrixKernelClass>(inTarget, inTargetRhs, inTargetNbParticles,
                                                                      inNeigh, inNeighNbParticles,
                                                                      MatrixKernel);
    }

    template <typename ContainerClass, typename ContainerClassRhs, typename MatrixKernelClass>
    static void P2PInner( const MatrixKernelClass *const MatrixKernel,
                          const ContainerClass& inTarget, ContainerClassRhs& inTargetRhs,
                          const long int inTargetNbParticles){
        FP2P::InnerKIJ<FReal, ContainerClass, MatrixKernelClass>(inTarget, inTargetRhs, inTargetNbParticles,
                                                                 MatrixKernel);
    }
};


/*! Specialization for scalar kernels and single rhs*/
template <class FReal>
struct DirectInteractionComputer<FReal, 1>
{
    template <typename ContainerClass, typename ContainerClassRhs, typename MatrixKernelClass>
    static void P2P(const ContainerClass& inNeigh, const long int inNeighNbParticles,
                    const MatrixKernelClass *const MatrixKernel,
                    const ContainerClass& inTarget, ContainerClassRhs& inTargetRhs,
                    const long int inTargetNbParticles){
        FP2PT<FReal>::template FullMutual<ContainerClass,MatrixKernelClass> (inTarget, inTargetRhs, inTargetNbParticles,
                                                                             inNeigh, inNeighNbParticles,
                                                                             MatrixKernel);
    }

    template <typename ContainerClass, typename ContainerClassRhs, typename MatrixKernelClass>
    static void P2PInner( const MatrixKernelClass *const MatrixKernel,
                          const ContainerClass& inTarget, ContainerClassRhs& inTargetRhs,
                          const long int inTargetNbParticles){
        FP2PT<FReal>::template Inner<ContainerClass, MatrixKernelClass>(inTarget, inTargetRhs, inTargetNbParticles,
                                                                        MatrixKernel);
    }
};

#endif // FINTERPP2PKERNELS_HPP
