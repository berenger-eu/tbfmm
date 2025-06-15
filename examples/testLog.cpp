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
#include <filesystem>
#include <map>

template <typename RealType, int Dim>
void generateParticles(std::vector<std::array<RealType, Dim + 1>> &particlePositions,
                       const long int nbParticles,
                       const std::string &distributionType)
{
    particlePositions.resize(nbParticles);
    std::mt19937 rng(42); // фиксированное зерно
    std::uniform_real_distribution<RealType> uniform01(0.0, 1.0);
    std::normal_distribution<RealType> normal(0.0, 1.0);

    if (distributionType == "uniform")
    {
        for (long int i = 0; i < nbParticles; ++i)
        {
            particlePositions[i][0] = uniform01(rng);
            particlePositions[i][1] = uniform01(rng);
            particlePositions[i][2] = 1.0; // заряд
        }
    }

    else if (distributionType == "normal")
    {
        std::normal_distribution<RealType> normalX(0.5, 0.15); // center=0.5, std=0.15
        std::normal_distribution<RealType> normalY(0.5, 0.15);
        for (long int i = 0; i < nbParticles; ++i)
        {
            RealType x = normalX(rng);
            RealType y = normalY(rng);
            x = std::min(std::max(x, RealType(0.0)), RealType(1.0));
            y = std::min(std::max(y, RealType(0.0)), RealType(1.0));
            particlePositions[i][0] = x;
            particlePositions[i][1] = y;
            particlePositions[i][2] = 1.0;
        }
    }

    else if (distributionType == "clustered")
    {
        std::array<std::pair<RealType, RealType>, 4> centers = {
            std::make_pair(0.25, 0.25), std::make_pair(0.75, 0.25),
            std::make_pair(0.25, 0.75), std::make_pair(0.75, 0.75)};
        long int clustered = nbParticles / 4;
        long int remaining = nbParticles - clustered * 4;
        long int idx = 0;

        for (const auto &c : centers)
        {
            for (int i = 0; i < clustered; ++i)
            {
                RealType x = c.first + 0.05 * normal(rng);
                RealType y = c.second + 0.05 * normal(rng);
                x = std::min(std::max(x, RealType(0.0)), RealType(1.0));
                y = std::min(std::max(y, RealType(0.0)), RealType(1.0));
                particlePositions[idx][0] = x;
                particlePositions[idx][1] = y;
                particlePositions[idx][2] = 1.0;
                ++idx;
            }
        }

        for (int i = 0; i < remaining; ++i)
        {
            particlePositions[idx][0] = uniform01(rng);
            particlePositions[idx][1] = uniform01(rng);
            particlePositions[idx][2] = 1.0;
            ++idx;
        }
    }

    else
    {
        std::cerr << "Unknown distribution type: " << distributionType << std::endl;
        std::exit(1);
    }
}

template <unsigned int P>
void runWithP(const long int TreeHeight, const long int nbParticles, const std::string &distributionType, bool noCheck)
{
    using RealType = double;
    const int Dim = 2;

    std::vector<std::array<RealType, Dim + 1>> particlePositions;
    std::array<RealType, Dim> BoxWidths;
    std::array<RealType, Dim> BoxCenter;
    BoxWidths = std::array<RealType, Dim>{{1, 1}};
    BoxCenter = std::array<RealType, Dim>{{0.5, 0.5}};

    TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

    generateParticles<RealType, Dim>(particlePositions, nbParticles, distributionType);

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

    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim + 1;
    using ParticleRhsType = RealType;
    constexpr long int NbRhsValuesPerParticle = 1;

    constexpr long int VectorSize = P + 1;

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
    TreeClass tree(configuration, TbfUtils::make_const(particlePositions));

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

    TbfTimer timerExecute;
    {
        // Execute a full FMM (near + far fields)
        algorithm->execute(tree);
    }
    timerExecute.stop();
    std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

    std::ofstream resultFile("results.csv", std::ios::app);
    if (!std::filesystem::exists("results.csv"))
    {
        std::ofstream header("results.csv");
        header << "TreeHeight,P,Distribution,AccuracyError,TimeFMM,TimeDirect,N\n";
    }
    if (!resultFile.is_open())
    {
        std::cerr << "Failed to open file!\n";
        return;
    }

    std::map<long int, int> hist;
    tree.applyToAllLeaves([&hist](auto &&leafHeader, const long int *,
                                  const std::array<ParticleDataType *, Dim + 1> /*particleDataPtr*/,
                                  const std::array<ParticleRhsType *, NbRhsValuesPerParticle> /*particleRhsPtr*/)
                          {
    long int n = leafHeader.nbParticles;
    hist[n] += 1; });

    // Save histogram to CSV
    // std::string histFileName = distributionType + "_hist.csv";
    // std::ofstream histFile(histFileName, std::ios::app);
    // histFile << "N,TreeHeight,ParticlesPerLeaf,LeafCount\n";
    // for (const auto &kv : hist)
    // {
    //     histFile << nbParticles << "," << TreeHeight << "," << kv.first << "," << kv.second << "\n";
    // }
    // histFile.close();
    // std::cout << "Histogram saved to: " << histFileName << std::endl;
    // if (nbParticles < 1e4)
    // {
    //     std::string particleFile = distributionType + "_particles.csv";
    //     std::ofstream particleOut(particleFile);
    //     particleOut << "x,y,charge\n";
    //     for (const auto &p : particlePositions)
    //     {
    //         particleOut << p[0] << "," << p[1] << "," << p[2] << "\n";
    //     }
    //     std::cout << "Saved particle positions to: " << particleFile << std::endl;
    // }
    if (!noCheck)
    {
        std::array<RealType *, 3> particles;
        for (auto &vec : particles)
        {
            vec = new RealType[nbParticles]();
        }
        std::array<RealType *, NbRhsValuesPerParticle> particlesRhs;
        for (auto &vec : particlesRhs)
        {
            vec = new RealType[nbParticles]();
        }

        for (long int idxPart = 0; idxPart < nbParticles; ++idxPart)
        {
            particles[0][idxPart] = particlePositions[idxPart][0];
            particles[1][idxPart] = particlePositions[idxPart][1];
            particles[2][idxPart] = particlePositions[idxPart][2];
        }

        TbfTimer timerDirect;

        FP2PLog::template GenericInner<RealType>(particles, particlesRhs, nbParticles);

        timerDirect.stop();

        std::cout << "Direct execute in " << timerDirect.getElapsed() << "s" << std::endl;

        //////////////////////////////////////////////////////////////////////

        std::array<TbfAccuracyChecker<RealType>, 3> partcilesAccuracy;
        std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

        tree.applyToAllLeaves([&particles, &partcilesAccuracy, &particlesRhs, &partcilesRhsAccuracy, NbRhsValuesPerParticle](auto &&leafHeader, const long int *particleIndexes,
                                                                                                                             const std::array<ParticleDataType *, Dim + 1> particleDataPtr,
                                                                                                                             const std::array<ParticleRhsType *, NbRhsValuesPerParticle> particleRhsPtr)
                              {
            for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                for(int idxValue = 0 ; idxValue < Dim+1 ; ++idxValue){
                   partcilesAccuracy[idxValue].addValues(particles[idxValue][particleIndexes[idxPart]],
                                                        particleDataPtr[idxValue][idxPart]);
                }
                for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                   partcilesRhsAccuracy[idxValue].addValues(particlesRhs[idxValue][particleIndexes[idxPart]],
                                                        particleRhsPtr[idxValue][idxPart]);
                }
            } });

        std::cout << "Relative differences:" << std::endl;
        for (int idxValue = 0; idxValue < NbRhsValuesPerParticle; ++idxValue)
        {
            std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
        }

        //////////////////////////////////////////////////////////////////////
        // result.csv
        //  resultFile << TreeHeight << "," << P << "," << distributionType << ","
        //             << partcilesRhsAccuracy[0].getRelativeInfNorm() << ","
        //             << timerExecute.getElapsed() << ","
        //             << timerDirect.getElapsed() << "," << nbParticles << std::endl;
        std::cout << TreeHeight << "," << P << "," << distributionType << ","
                  << partcilesRhsAccuracy[0].getRelativeInfNorm() << ","
                  << timerExecute.getElapsed() << ","
                  << timerDirect.getElapsed() << "," << nbParticles << std::endl;
    }
    else
    {
        // result.csv
        //  resultFile << TreeHeight << "," << P << "," << distributionType << ","
        //             << 0.0 << "," << timerExecute.getElapsed() << "," << 0.0 << "," << nbParticles << std::endl;
        std::cout << TreeHeight << "," << P << "," << distributionType << ","
                  << 0.0 << "," << timerExecute.getElapsed() << "," << 0.0 << "," << nbParticles << std::endl;
    }
}

int main(int argc, char **argv)
{

    const long int nbParticles = TbfParams::GetValue<long int>(
        argc, argv, {"-nb", "--nb-particles"}, 2e3);

    const long int TreeHeight = TbfParams::GetValue<long int>(
        argc, argv, {"-th", "--tree-height"}, 4);

    const unsigned int p = TbfParams::GetValue<unsigned int>(
        argc, argv, {"-p", "--order"}, 10);

    std::string distributionType = TbfParams::GetValue<std::string>(
        argc, argv, {"-d", "--distribution"}, "clustered");

    bool noCheck = TbfParams::ExistParameter(argc, argv, {"-nc", "--no-check"});
    // noCheck = true;
    switch (p)
    {
    case 4:
        runWithP<4>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    case 6:
        runWithP<6>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    case 8:
        runWithP<8>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    case 10:
        runWithP<10>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    case 12:
        runWithP<12>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    case 16:
        runWithP<16>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    case 20:
        runWithP<20>(TreeHeight, nbParticles, distributionType, noCheck);
        break;
    default:
        std::cerr << "Unsupported P value. Allowed: 4,6,8,10,12,16,20\n";
        return 2;
    }

    return 0;
}