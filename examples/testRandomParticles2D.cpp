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

#include "utils/tbfparams.hpp"

#include <iostream>

int main(int argc, char **argv)
{
    if (TbfParams::ExistParameter(argc, argv, {"-h", "--help"}))
    {
        std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        std::cout << "[HELP] where params are:" << std::endl;
        std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        return 1;
    }

    using RealType = double;
    const int Dim = 2;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1}};
    const long int TreeHeight = TbfParams::GetValue<long int>(argc, argv, {"-th", "--tree-height"}, 8);
    const std::array<RealType, Dim> BoxCenter{{0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = TbfParams::GetValue<long int>(argc, argv, {"-nb", "--nb-particles"}, 1000);

    std::cout << "Particles info" << std::endl;
    std::cout << " - Tree height = " << TreeHeight << std::endl;
    std::cout << " - Number of particles = " << NbParticles << std::endl;

    TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

    std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);

    for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
    {
        particlePositions[idxPart] = randomGenerator.getNewItem();
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim;
    using ParticleRhsType = long int;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int, 1>;
    using LocalClass = std::array<long int, 1>;
    using SpaceIndexType = TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>, false>;
    using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass,
                              SpaceIndexType>;

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    TreeClass tree(configuration, particlePositions);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << "s" << std::endl;

    {
        using KernelClass = TbfTestKernel<RealType, SpaceIndexType>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;
        AlgorithmClass algorithm(configuration);
        std::cout << "algorithm -- " << algorithm << std::endl;

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    { // Same as above but with interaction counter
        using KernelClass = TbfInteractionCounter<TbfTestKernel<RealType, SpaceIndexType>>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

        // Print the counter's result
        auto counters = typename KernelClass::ReduceType();

        algorithm.applyToAllKernels([&](const auto &inKernel)
                                    { counters = KernelClass::ReduceType::Reduce(counters, inKernel.getReduceData()); });

        std::cout << counters << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    { // Same as above but with interaction timer
        using KernelClass = TbfInteractionTimer<TbfTestKernel<RealType, SpaceIndexType>>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

        // Print the counter's result
        auto timers = typename KernelClass::ReduceType();

        algorithm.applyToAllKernels([&](const auto &inKernel)
                                    { timers = KernelClass::ReduceType::Reduce(timers, inKernel.getReduceData()); });

        std::cout << timers << std::endl;
    }
    /////////////////////////////////////////////////////////////////////////////////////////
    { // Same as above but with interaction counter & timer
        using KernelClass = TbfInteractionCounter<TbfInteractionTimer<TbfTestKernel<RealType, SpaceIndexType>>>;
        using ReduceTypeCounter = KernelClass::TbfInteractionCounter::ReduceType;
        using ReduceTypeTimer = KernelClass::TbfInteractionTimer::ReduceType;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

        // Print the counter's result
        auto counters = typename KernelClass::TbfInteractionCounter::ReduceType();
        auto timers = typename KernelClass::TbfInteractionTimer::ReduceType();

        algorithm.applyToAllKernels([&](const auto &inKernel)
                                    {
            inKernel.KernelClass::getReduceData();
            counters = ReduceTypeCounter::Reduce(counters, inKernel.KernelClass::getReduceData());
            timers = ReduceTypeTimer::Reduce(timers, inKernel.TbfInteractionTimer<TbfTestKernel<RealType, SpaceIndexType>>::getReduceData()); });

        std::cout << counters << std::endl;
        std::cout << timers << std::endl;
    }

    return 0;
}
