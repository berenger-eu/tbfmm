#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbfaccuracychecker.hpp"
#include "utils/tbfparams.hpp"
#include "utils/tbftimer.hpp"

#include "kernels/gausskernel/tbfgausskernel.hpp"
#include "kernels/P2P/FP2PGauss.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

template <class RealType, int Dim>
void generateParticles(
    std::vector<std::array<RealType, Dim + 1>>& particles,
    const long int nbParticles)
{
    particles.resize(nbParticles);

    std::mt19937 rng(42);
    std::uniform_real_distribution<RealType> uniform01(0.0, 1.0);
    std::uniform_real_distribution<RealType> chargeDist(-1.0, 1.0);

    for (long int i = 0; i < nbParticles; ++i)
    {
        particles[i][0] = uniform01(rng);
        particles[i][1] = uniform01(rng);
        particles[i][2] = chargeDist(rng);
    }
}

template <unsigned int P>
void runGaussKernel(
    const long int treeHeight,
    const long int nbParticles,
    const double delta,
    const double coefficient)
{
    using RealType = double;

    constexpr int Dim = 2;
    constexpr long int NbDataValuesPerParticle = Dim + 1;
    constexpr long int NbRhsValuesPerParticle = 1;

    constexpr long int VectorSize = (P + 1) * (P + 1);

    using ParticleDataType = RealType;
    using ParticleRhsType = RealType;

    using MultipoleClass = std::array<RealType, VectorSize>;
    using LocalClass = std::array<RealType, VectorSize>;

    using SpaceIndexType = TbfDefaultSpaceIndexType2D<RealType>;
    using KernelClass = TbfGaussKernel<RealType, P, SpaceIndexType>;

    using AlgorithmClass = TbfAlgorithmSelecter::type<
        RealType,
        KernelClass,
        SpaceIndexType>;

    using TreeClass = TbfTree<
        RealType,
        ParticleDataType,
        NbDataValuesPerParticle,
        ParticleRhsType,
        NbRhsValuesPerParticle,
        MultipoleClass,
        LocalClass,
        SpaceIndexType>;

    std::array<RealType, Dim> boxWidths{{1.0, 1.0}};
    std::array<RealType, Dim> boxCenter{{0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(
        treeHeight,
        boxWidths,
        boxCenter);

    std::vector<std::array<RealType, Dim + 1>> particlePositions;
    generateParticles<RealType, Dim>(particlePositions, nbParticles);

    std::cout << "Gauss full kernel test\n";
    std::cout << "Tree height = " << treeHeight << "\n";
    std::cout << "Number of particles = " << nbParticles << "\n";
    std::cout << "P = " << P << "\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n\n";

    TbfTimer timerBuildTree;
    TreeClass tree(configuration, TbfUtils::make_const(particlePositions));
    timerBuildTree.stop();

    std::cout << "Build tree time = "
              << timerBuildTree.getElapsed()
              << "s\n";

    KernelClass kernel(configuration, delta, coefficient);

    AlgorithmClass algorithm(configuration, kernel);

    std::cout << "Algorithm name = " << algorithm.GetName() << "\n";
    std::cout << "Number of threads = " << algorithm.GetNbThreads() << "\n";

    TbfTimer timerFmm;
    KernelClass::ResetCounters();
    algorithm.execute(tree);
    KernelClass::PrintCounters();
    timerFmm.stop();

    std::cout << "FMM execute time = "
              << timerFmm.getElapsed()
              << "s\n";

    std::array<RealType*, NbDataValuesPerParticle> directParticles;
    std::array<RealType*, NbRhsValuesPerParticle> directRhs;

    for (auto& ptr : directParticles)
    {
        ptr = new RealType[nbParticles]();
    }

    for (auto& ptr : directRhs)
    {
        ptr = new RealType[nbParticles]();
    }

    for (long int i = 0; i < nbParticles; ++i)
    {
        directParticles[0][i] = particlePositions[i][0];
        directParticles[1][i] = particlePositions[i][1];
        directParticles[2][i] = particlePositions[i][2];
    }

    TbfTimer timerDirect;

    FP2PGauss::template GenericInner<RealType>(
        directParticles,
        directRhs,
        nbParticles,
        RealType(delta),
        RealType(coefficient));

    timerDirect.stop();

    std::cout << "Direct P2P time = "
              << timerDirect.getElapsed()
              << "s\n\n";

    std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle>
        rhsAccuracy;

    tree.applyToAllLeaves(
        [&directRhs, &rhsAccuracy](
            auto&& leafHeader,
            const long int* particleIndexes,
            const std::array<ParticleDataType*, NbDataValuesPerParticle>
                /*particleDataPtr*/,
            const std::array<ParticleRhsType*, NbRhsValuesPerParticle>
                particleRhsPtr)
        {
            for (long int idxPart = 0;
                 idxPart < leafHeader.nbParticles;
                 ++idxPart)
            {
                const long int originalIndex =
                    particleIndexes[idxPart];

                rhsAccuracy[0].addValues(
                    directRhs[0][originalIndex],
                    particleRhsPtr[0][idxPart]);
            }
        });

    std::cout << "Relative differences:\n";
    std::cout << "RHS 0 = " << rhsAccuracy[0] << "\n";

    for (auto& ptr : directParticles)
    {
        delete[] ptr;
    }

    for (auto& ptr : directRhs)
    {
        delete[] ptr;
    }
}

int main(int argc, char** argv)
{
    const long int nbParticles =
        TbfParams::GetValue<long int>(
            argc,
            argv,
            {"-nb", "--nb-particles"},
            2000);

    const long int treeHeight =
        TbfParams::GetValue<long int>(
            argc,
            argv,
            {"-th", "--tree-height"},
            4);

    const unsigned int p =
        TbfParams::GetValue<unsigned int>(
            argc,
            argv,
            {"-p", "--order"},
            6);

    const double delta =
        TbfParams::GetValue<double>(
            argc,
            argv,
            {"-delta", "--delta"},
            0.05);

    const double coefficient =
        TbfParams::GetValue<double>(
            argc,
            argv,
            {"-coef", "--coefficient"},
            1.0);

    switch (p)
    {
    case 2:
        runGaussKernel<2>(treeHeight, nbParticles, delta, coefficient);
        break;
    case 4:
        runGaussKernel<4>(treeHeight, nbParticles, delta, coefficient);
        break;
    case 6:
        runGaussKernel<6>(treeHeight, nbParticles, delta, coefficient);
        break;
    case 8:
        runGaussKernel<8>(treeHeight, nbParticles, delta, coefficient);
        break;
    case 10:
        runGaussKernel<10>(treeHeight, nbParticles, delta, coefficient);
        break;
    case 12:
        runGaussKernel<12>(treeHeight, nbParticles, delta, coefficient);
        break;
    default:
        std::cerr << "Unsupported P value. Allowed: 2,4,6,8,10,12\n";
        return 2;
    }

    return 0;
}