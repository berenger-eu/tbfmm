#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "kernels/testkernel/tbftestkernel.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbftimer.hpp"
#include "kernels/counterkernels/tbfinteractioncounter.hpp"
#include "kernels/counterkernels/tbfinteractiontimer.hpp"


#include <iostream>


int main(){
    using RealType = double;
    const int Dim = 3;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
    const long int TreeHeight = 8;
    const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = 1000;

    TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

    std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);

    for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        particlePositions[idxPart] = randomGenerator.getNewItem();
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim;
    using ParticleRhsType = long int;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;
    const long int NbElementsPerBlock = 50;
    const bool OneGroupPerParent = false;
    using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass>;

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    TreeClass tree(configuration, NbElementsPerBlock, particlePositions, OneGroupPerParent);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

    {
        using KernelClass = TbfTestKernel<RealType>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;
        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    { // Same as above but with interaction counter
        using KernelClass = TbfInteractionCounter<TbfTestKernel<RealType>>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

        // Print the counter's result
        auto counters = typename KernelClass::ReduceType();

        algorithm.applyToAllKernels([&](const auto& inKernel){
            counters = KernelClass::ReduceType::Reduce(counters, inKernel.getReduceData());
        });

        std::cout << counters << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    { // Same as above but with interaction timer
        using KernelClass = TbfInteractionTimer<TbfTestKernel<RealType>>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

        // Print the counter's result
        auto timers = typename KernelClass::ReduceType();

        algorithm.applyToAllKernels([&](const auto& inKernel){
            timers = KernelClass::ReduceType::Reduce(timers, inKernel.getReduceData());
        });

        std::cout << timers << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    { // Same as above but with interaction counter & timer
        using KernelClass = TbfInteractionCounter<TbfInteractionTimer<TbfTestKernel<RealType>>>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

        // Print the counter's result
        auto counters = typename KernelClass::TbfInteractionCounter::ReduceType();
        auto timers = typename KernelClass::TbfInteractionTimer::ReduceType();

        algorithm.applyToAllKernels([&](const auto& inKernel){
            counters = KernelClass::TbfInteractionCounter::ReduceType::Reduce(counters, inKernel.KernelClass::getReduceData());
            timers = KernelClass::TbfInteractionTimer::ReduceType::Reduce(timers, inKernel.TbfInteractionTimer<TbfTestKernel<RealType>>::getReduceData());
        });

        std::cout << counters << std::endl;
        std::cout << timers << std::endl;
    }

    return 0;
}

