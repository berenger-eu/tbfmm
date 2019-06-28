#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "kernels/testkernel/tbftestkernel.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "utils/tbftimer.hpp"

/// - add uniform kernels
/// - add GPU kernels
/// - do unit tests
/// - periodicity
/// - ensure code quality with visilibity of Template

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

    constexpr long int NbDataValuesPerParticle = Dim;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;
    const long int inNbElementsPerBlock = 50;
    const bool inOneGroupPerParent = false;

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    TbfTree<RealType, RealType, NbDataValuesPerParticle, long int, NbRhsValuesPerParticle, MultipoleClass, LocalClass> tree(configuration, inNbElementsPerBlock,
                                                                                particlePositions, inOneGroupPerParent);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

    TbfAlgorithm<RealType, TbfTestKernel<RealType>> algorithm(configuration);

    TbfTimer timerExecute;

    algorithm.execute(tree);

    timerExecute.stop();
    std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

    return 0;
}

