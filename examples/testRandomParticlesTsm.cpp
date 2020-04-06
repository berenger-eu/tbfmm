#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftreetsm.hpp"
#include "kernels/testkernel/tbftestkernel.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbftimer.hpp"
#include "kernels/counterkernels/tbfinteractioncounter.hpp"
#include "kernels/counterkernels/tbfinteractiontimer.hpp"

#include "utils/tbfparams.hpp"


#include <iostream>


int main(int argc, char** argv){
    if(TbfParams::ExistParameter(argc, argv, {"-h", "--help"})){
        std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        std::cout << "[HELP] where params are:" << std::endl;
        std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        return 1;
    }

    using RealType = double;
    const int Dim = 3;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
    const long int TreeHeight = TbfParams::GetValue<long int>(argc, argv, {"-th", "--tree-height"}, 6);
    const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = TbfParams::GetValue<long int>(argc, argv, {"-nb", "--nb-particles"}, 1000);

    std::cout << "Particles info" << std::endl;
    std::cout << " - Tree height = " << TreeHeight << std::endl;
    std::cout << " - Number of particles = " << NbParticles << std::endl;

    TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

    std::vector<std::array<RealType, Dim>> particlePositionsSource(NbParticles);
    std::vector<std::array<RealType, Dim>> particlePositionsTarget(NbParticles);

    for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        particlePositionsSource[idxPart] = randomGenerator.getNewItem();
        particlePositionsTarget[idxPart] = randomGenerator.getNewItem();
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim;
    using ParticleRhsType = long int;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;
    using AlgorithmClass = TbfAlgorithmSelecterTsm::type<RealType, TbfTestKernel<RealType>>;
    using TreeClass = TbfTreeTsm<RealType,
                                 ParticleDataType,
                                 NbDataValuesPerParticle,
                                 ParticleRhsType,
                                 NbRhsValuesPerParticle,
                                 MultipoleClass,
                                 LocalClass>;

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    TreeClass tree(configuration, particlePositionsSource, particlePositionsTarget);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << "s" << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    AlgorithmClass algorithm(configuration);

    TbfTimer timerExecute;

    algorithm.execute(tree);

    timerExecute.stop();
    std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

    return 0;
}

