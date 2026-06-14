#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbfparams.hpp"
#include "utils/tbftimer.hpp"

#include "kernels/gausskernel/tbfgausskernel.hpp"
#include "kernels/P2P/FP2PGauss.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

template <class RealType, int Dim>
void generateTwoClusters(
    std::vector<std::array<RealType, Dim + 1>>& particles)
{
    constexpr long int wellsPerCluster = 10;
    constexpr long int nbParticles = 2 * wellsPerCluster;

    particles.resize(nbParticles);

    std::mt19937 rng(42);

    std::normal_distribution<RealType> offset(0.0, 0.025);
    std::uniform_real_distribution<RealType> chargeDist(0.5, 1.5);

    const std::array<RealType, Dim> clusterA{{0.20, 0.20}};
    const std::array<RealType, Dim> clusterB{{0.80, 0.80}};

    for (long int i = 0; i < wellsPerCluster; ++i)
    {
        particles[i][0] =
            std::clamp(clusterA[0] + offset(rng), RealType(0), RealType(1));

        particles[i][1] =
            std::clamp(clusterA[1] + offset(rng), RealType(0), RealType(1));

        particles[i][2] = chargeDist(rng);
    }

    for (long int i = 0; i < wellsPerCluster; ++i)
    {
        const long int idx = wellsPerCluster + i;

        particles[idx][0] =
            std::clamp(clusterB[0] + offset(rng), RealType(0), RealType(1));

        particles[idx][1] =
            std::clamp(clusterB[1] + offset(rng), RealType(0), RealType(1));

        particles[idx][2] = chargeDist(rng);
    }
}

template <unsigned int P>
void runTwoClusters(
    const long int treeHeight,
    const double delta,
    const double coefficient)
{
    using RealType = double;

    constexpr int Dim = 2;
    constexpr long int nbParticles = 20;

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
    generateTwoClusters<RealType, Dim>(particlePositions);

    std::cout << "Gauss two-clusters test\n";
    std::cout << "Tree height = " << treeHeight << "\n";
    std::cout << "Number of wells = " << nbParticles << "\n";
    std::cout << "P = " << P << "\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n\n";

    std::cout << "Particles:\n";
    for (long int i = 0; i < nbParticles; ++i)
    {
        std::cout
            << i
            << " | x = " << particlePositions[i][0]
            << " | y = " << particlePositions[i][1]
            << " | q = " << particlePositions[i][2]
            << "\n";
    }
    std::cout << "\n";

    TbfTimer timerBuildTree;

    TreeClass tree(
        configuration,
        TbfUtils::make_const(particlePositions));

    timerBuildTree.stop();

    KernelClass kernel(
        configuration,
        RealType(delta),
        RealType(coefficient));

    AlgorithmClass algorithm(configuration, kernel);

    TbfTimer timerFmm;

    algorithm.execute(tree);

    timerFmm.stop();

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

    RealType errorL2 = RealType(0);
    RealType referenceL2 = RealType(0);
    RealType errorInf = RealType(0);
    RealType referenceInf = RealType(0);
    RealType maxRelativeDiff = RealType(0);

    std::vector<RealType> fmmValues(nbParticles, RealType(0));

    tree.applyToAllLeaves(
        [&](auto&& leafHeader,
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
                const long int originalIndex = particleIndexes[idxPart];

                const RealType ref = directRhs[0][originalIndex];
                const RealType fmm = particleRhsPtr[0][idxPart];

                fmmValues[originalIndex] = fmm;

                const RealType diff = fmm - ref;

                errorL2 += diff * diff;
                referenceL2 += ref * ref;

                errorInf = std::max(errorInf, std::abs(diff));
                referenceInf = std::max(referenceInf, std::abs(ref));

                maxRelativeDiff = std::max(
                    maxRelativeDiff,
                    std::abs(diff) / std::max(std::abs(ref), RealType(1e-30)));
            }
        });

    const RealType relativeL2 =
        std::sqrt(errorL2)
        / std::max(std::sqrt(referenceL2), RealType(1e-30));

    const RealType rmsError =
        std::sqrt(errorL2 / RealType(nbParticles));

    const RealType relativeInf =
        errorInf / std::max(referenceInf, RealType(1e-30));

    const double timeBuildTree = timerBuildTree.getElapsed();
    const double timeFmm = timerFmm.getElapsed();
    const double timeDirect = timerDirect.getElapsed();

    std::cout << "Results per well:\n";
    for (long int i = 0; i < nbParticles; ++i)
    {
        const RealType ref = directRhs[0][i];
        const RealType fmm = fmmValues[i];
        const RealType diff = fmm - ref;

        std::cout
            << i
            << " | FMM = " << fmm
            << " | direct = " << ref
            << " | diff = " << diff
            << "\n";
    }

    std::cout << "\n";
    std::cout << "Build tree time = " << timeBuildTree << "s\n";
    std::cout << "FMM execute time = " << timeFmm << "s\n";
    std::cout << "Direct P2P time = " << timeDirect << "s\n";

    if (timeFmm > 0.0)
    {
        std::cout << "Speedup = " << timeDirect / timeFmm << "\n";
    }

    std::cout << "\n";
    std::cout << "Relative differences:\n";
    std::cout << "Relative L2Norm = " << relativeL2 << "\n";
    std::cout << "RMS Norm = " << rmsError << "\n";
    std::cout << "Relative Inf = " << relativeInf << "\n";
    std::cout << "Max Relative diff = " << maxRelativeDiff << "\n";

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
            8);

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
        runTwoClusters<2>(treeHeight, delta, coefficient);
        break;
    case 4:
        runTwoClusters<4>(treeHeight, delta, coefficient);
        break;
    case 6:
        runTwoClusters<6>(treeHeight, delta, coefficient);
        break;
    case 8:
        runTwoClusters<8>(treeHeight, delta, coefficient);
        break;
    case 10:
        runTwoClusters<10>(treeHeight, delta, coefficient);
        break;
    case 12:
        runTwoClusters<12>(treeHeight, delta, coefficient);
        break;
    case 16:
        runTwoClusters<16>(treeHeight, delta, coefficient);
        break;
    default:
        std::cerr << "Unsupported P value. Allowed: 2,4,6,8,10,12,16\n";
        return 2;
    }

    return 0;
}