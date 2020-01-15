#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#include "utils/tbftimer.hpp"

#include "kernels/unifkernel/FUnifKernel.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

#include <iostream>


int main(){
    using RealType = double;
    const int Dim = 3;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
    const long int TreeHeight = 8;
    const std::array<RealType, Dim> inBoxCenter{{0.5, 0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, inBoxCenter);

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = 1000;

    TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

    std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);

    for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        particlePositions[idxPart] = randomGenerator.getNewItem();
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    const unsigned int ORDER = 5;
    constexpr long int NbDataValuesPerParticle = Dim+1;
    constexpr long int NbRhsValuesPerParticle = 4;

    constexpr long int VectorSize = TensorTraits<ORDER>::nnodes;
    constexpr long int TransformedVectorSize = (2*ORDER-1)*(2*ORDER-1)*(2*ORDER-1);

    struct MultipoleData{
        RealType multipole_exp[NbRhsValuesPerParticle * 1 * VectorSize];
        std::complex<RealType> transformed_multipole_exp[NbRhsValuesPerParticle * 1 * TransformedVectorSize];
    };

    struct LocalData{
        RealType     local_exp[NbRhsValuesPerParticle * 1 * VectorSize];
        std::complex<RealType>     transformed_local_exp[NbRhsValuesPerParticle * 1 * TransformedVectorSize];
    };

    using MultipoleClass = MultipoleData;
    using LocalClass = LocalData;
    typedef FUnifKernel<RealType, FInterpMatrixKernelR<RealType>, ORDER> KernelClass;
    const long int inNbElementsPerBlock = 50;
    const bool inOneGroupPerParent = false;

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    TbfTree<RealType, RealType, NbDataValuesPerParticle, RealType, NbRhsValuesPerParticle, MultipoleClass, LocalClass> tree(configuration, inNbElementsPerBlock,
                                                                                particlePositions, inOneGroupPerParent);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

    FInterpMatrixKernelR<RealType> interpolator;
    TbfAlgorithm<RealType, KernelClass> algorithm(configuration, KernelClass(configuration, &interpolator));
    //TbfSmSpetabaruAlgorithm<RealType, KernelClass> algorithm(configuration, KernelClass(configuration, &interpolator));

    TbfTimer timerExecute;

    algorithm.execute(tree);

    timerExecute.stop();
    std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

    return 0;
}

