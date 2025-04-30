
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbftimer.hpp"

#include "kernels/logkernel/tbflogkernel.hpp"
#include "loader/tbffmaloader.hpp"
#include "utils/tbfaccuracychecker.hpp"

#include "utils/tbfparams.hpp"

#include "kernels/unifkernel/FMath.hpp"

#include <iostream>

int main(int argc, char **argv)
{
    // Manage the arguments, and print the help if needed
    if (TbfParams::ExistParameter(argc, argv, {"-h", "--help"}))
    {
        std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        std::cout << "[HELP] where params are:" << std::endl;
        std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        std::cout << "[HELP]   -f, --file: to pass a particle file (FMA)" << std::endl;
        std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        std::cout << "[HELP]   -nc, --no-check: avoid comparing FMM results with direct computation" << std::endl;
        std::cout << "[HELP]   -nbl, --nb-loops: the number of extra loops (FMM iterations)" << std::endl;
        return 1;
    }

    // The data type used for the positions and in the computation
    // (could have been different but it is not the case here)
    using RealType = double;
    // In 2D
    const int Dim = 2;

    /////////////////////////////////////////////////////////////////////////////////////////

    // We store the positions + a physical value in a vector
    std::vector<std::array<RealType, Dim + 1>> particlePositions;
    long int nbParticles;
    std::array<RealType, Dim> BoxWidths;
    std::array<RealType, Dim> BoxCenter;

    // We load a file if "-f" is given
    if (TbfParams::ExistParameter(argc, argv, {"-f", "--file"}))
    {
        std::string filename = TbfParams::GetStr(argc, argv, {"-f", "--file"}, "");
        TbfFmaLoader<RealType, Dim, Dim + 1> loader(filename);

        if (!loader.isOpen())
        {
            std::cout << "[Error] There is a problem, the given file '" << filename << "' cannot be open." << std::endl;
            return -1;
        }

        nbParticles = loader.getNbParticles();
        particlePositions = loader.loadAllParticles();
        BoxWidths = loader.getBoxWidths();
        BoxCenter = loader.getBoxCenter();
    }
    // Otherwise we generate random positions
    else
    {
        // BoxWidths = std::array<RealType, Dim>{{10, 10}};
        // BoxCenter = std::array<RealType, Dim>{{0.1, 0.1}};

        // nbParticles = TbfParams::GetValue<long int>(argc, argv, {"-nb", "--nb-particles"}, 3);

        // TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

        // particlePositions.resize(nbParticles);
        // particlePositions[0][0] = 0;
        // particlePositions[0][1] = 0;
        // particlePositions[0][2] = 1 * FMath::FTwoPi<RealType>();
        // particlePositions[1][0] = std::exp(1);
        // particlePositions[1][1] = 0;
        // particlePositions[1][2] = 1 * FMath::FTwoPi<RealType>();
        // particlePositions[2][0] = std::exp(1);
        // particlePositions[2][1] = std::exp(1);
        // particlePositions[2][2] = 1 * FMath::FTwoPi<RealType>();
        BoxWidths = std::array<RealType, Dim>{{1, 1}};
        BoxCenter = std::array<RealType, Dim>{{0.5, 0.5}};

        nbParticles = TbfParams::GetValue<long int>(argc, argv, {"-nb", "--nb-particles"}, 1000);

        TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

        particlePositions.resize(nbParticles);

        for (long int idxPart = 0; idxPart < nbParticles; ++idxPart)
        {
            auto position = randomGenerator.getNewItem();
            particlePositions[idxPart][0] = position[0];
            particlePositions[idxPart][1] = position[1];
            particlePositions[idxPart][2] = position[2];
            particlePositions[idxPart][3] = 0.1; // the charge
        }
    }

    // The height of the tree
    const long int TreeHeight = TbfParams::GetValue<long int>(argc, argv, {"-th", "--tree-height"}, 4);
    // The spacial configuration of our system
    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

    std::cout << configuration << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    // Simply print some data
    std::cout << "Particles info" << std::endl;
    std::cout << " - Tree height = " << TreeHeight << std::endl;
    std::cout << " - Number of particles = " << nbParticles << std::endl;
    std::cout << " - Here are the first particles..." << std::endl;
    for (long int idxPart = 0; idxPart < std::min(5L, nbParticles); ++idxPart)
    {
        std::cout << TbfUtils::ArrayPrinter(particlePositions[idxPart]) << std::endl;
    }
    std::cout << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    // Fix our templates
    const unsigned int P = 5;
    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim + 1;
    using ParticleRhsType = RealType;
    constexpr long int NbRhsValuesPerParticle = 1;

    constexpr long int VectorSize = ((P + 2) * (P + 1)) / 2;

    using MultipoleClass = std::array<std::complex<RealType>, VectorSize>;
    using LocalClass = std::array<std::complex<RealType>, VectorSize>;

    using KernelClass = TbfLogKernel<RealType, P>;
    using AlgorithmClass = TbfAlgorithmSelecter::type<
        RealType,
        KernelClass,
        TbfDefaultSpaceIndexType2D<RealType>>;
    using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass,
                              TbfDefaultSpaceIndexType2D<RealType>>;

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    // Build the tree
    TreeClass tree(configuration, TbfUtils::make_const(particlePositions), 3);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << "s" << std::endl;
    std::cout << "Number of elements per group " << tree.getNbElementsPerGroup() << std::endl;
    std::cout << tree << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    // Here we put the kernel in the heap to make sure not to overflow the stack
    std::unique_ptr<AlgorithmClass> algorithm(new AlgorithmClass(configuration));
    std::cout << "Algorithm name " << algorithm->GetName() << std::endl;
    std::cout << "Number of threads " << algorithm->GetNbThreads() << std::endl;
    std::cout << *algorithm << std::endl;

    {
        TbfTimer timerExecute;

        // Execute a full FMM (near + far fields)
        // algorithm->execute(tree, TbfAlgorithmUtils::TbfOperations::TbfP2M);
        algorithm->execute(tree, TbfAlgorithmUtils::TbfOperations::TbfM2M);
        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;
    }

    return 0;
}
