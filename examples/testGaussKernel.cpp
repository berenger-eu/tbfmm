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
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

template <class RealType, int Dim>
void generateParticles(
    std::vector<std::array<RealType, Dim + 1>>& particles,
    const long int nbParticles,
    const std::string& distributionType)
{
    particles.resize(nbParticles);

    std::mt19937 rng(42);
    std::uniform_real_distribution<RealType> uniform01(0.0, 1.0);
    std::uniform_real_distribution<RealType> chargeDist(-1.0, 1.0);
    std::normal_distribution<RealType> normal01(0.0, 1.0);

    if (distributionType == "uniform")
    {
        for (long int i = 0; i < nbParticles; ++i)
        {
            particles[i][0] = uniform01(rng);
            particles[i][1] = uniform01(rng);
            particles[i][2] = chargeDist(rng);
        }
    }
    else if (distributionType == "normal")
    {
        std::normal_distribution<RealType> normalX(0.5, 0.15);
        std::normal_distribution<RealType> normalY(0.5, 0.15);

        for (long int i = 0; i < nbParticles; ++i)
        {
            particles[i][0] =
                std::clamp(normalX(rng), RealType(0), RealType(1));

            particles[i][1] =
                std::clamp(normalY(rng), RealType(0), RealType(1));

            particles[i][2] = chargeDist(rng);
        }
    }
    else if (distributionType == "clustered")
    {
        std::array<std::pair<RealType, RealType>, 4> centers = {
            std::make_pair(0.25, 0.25),
            std::make_pair(0.75, 0.25),
            std::make_pair(0.25, 0.75),
            std::make_pair(0.75, 0.75)};

        long int idx = 0;
        const long int perCluster = nbParticles / 4;

        for (const auto& center : centers)
        {
            for (long int k = 0; k < perCluster && idx < nbParticles; ++k)
            {
                particles[idx][0] =
                    std::clamp(center.first + RealType(0.05) * normal01(rng),
                               RealType(0),
                               RealType(1));

                particles[idx][1] =
                    std::clamp(center.second + RealType(0.05) * normal01(rng),
                               RealType(0),
                               RealType(1));

                particles[idx][2] = chargeDist(rng);
                ++idx;
            }
        }

        while (idx < nbParticles)
        {
            particles[idx][0] = uniform01(rng);
            particles[idx][1] = uniform01(rng);
            particles[idx][2] = chargeDist(rng);
            ++idx;
        }
    }
    else
    {
        std::cerr << "Unknown distribution: " << distributionType << "\n";
        std::exit(1);
    }
}

template <unsigned int P>
void runGaussKernel(
    const long int treeHeight,
    const long int nbParticles,
    const std::string& distributionType,
    const double delta,
    const double coefficient,
    const std::string& outputFile)
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
    generateParticles<RealType, Dim>(
        particlePositions,
        nbParticles,
        distributionType);

    std::cout << "Gauss full kernel test\n";
    std::cout << "Tree height = " << treeHeight << "\n";
    std::cout << "Number of particles = " << nbParticles << "\n";
    std::cout << "Distribution = " << distributionType << "\n";
    std::cout << "P = " << P << "\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n\n";

    TbfTimer timerBuildTree;
    TreeClass tree(configuration, TbfUtils::make_const(particlePositions));
    timerBuildTree.stop();

    KernelClass kernel(configuration, delta, coefficient);
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
        std::sqrt(errorL2) / std::max(std::sqrt(referenceL2), RealType(1e-30));

    const RealType rmsError =
        std::sqrt(errorL2 / RealType(nbParticles));

    const RealType relativeInf =
        errorInf / std::max(referenceInf, RealType(1e-30));

    const double timeFmm = timerFmm.getElapsed();
    const double timeDirect = timerDirect.getElapsed();
    const double speedup = timeDirect / timeFmm;

    std::cout << "Build tree time = " << timerBuildTree.getElapsed() << "s\n";
    std::cout << "FMM execute time = " << timeFmm << "s\n";
    std::cout << "Direct P2P time = " << timeDirect << "s\n";
    std::cout << "Speedup = " << speedup << "\n\n";

    std::cout << "Relative differences:\n";
    std::cout << "Relative L2Norm = " << relativeL2 << "\n";
    std::cout << "RMS Norm = " << rmsError << "\n";
    std::cout << "Relative Inf = " << relativeInf << "\n";
    std::cout << "Max Relative diff = " << maxRelativeDiff << "\n";

    const bool needHeader = !std::filesystem::exists(outputFile);

    std::ofstream out(outputFile, std::ios::app);

    if (needHeader)
    {
        out << "N,TreeHeight,P,Delta,Coefficient,Distribution,"
            << "RelativeL2,RMS,RelativeInf,MaxRelativeDiff,"
            << "TimeFMM,TimeDirect,Speedup\n";
    }

    out << nbParticles << ","
        << treeHeight << ","
        << P << ","
        << delta << ","
        << coefficient << ","
        << distributionType << ","
        << relativeL2 << ","
        << rmsError << ","
        << relativeInf << ","
        << maxRelativeDiff << ","
        << timeFmm << ","
        << timeDirect << ","
        << speedup << "\n";

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

    const std::string distributionType =
        TbfParams::GetValue<std::string>(
            argc,
            argv,
            {"-d", "--distribution"},
            "uniform");

    const std::string outputFile =
        TbfParams::GetValue<std::string>(
            argc,
            argv,
            {"-o", "--output"},
            "results_gauss.csv");

    switch (p)
    {
    case 2:
        runGaussKernel<2>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    case 4:
        runGaussKernel<4>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    case 6:
        runGaussKernel<6>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    case 8:
        runGaussKernel<8>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    case 10:
        runGaussKernel<10>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    case 12:
        runGaussKernel<12>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    case 16:
        runGaussKernel<16>(
            treeHeight,
            nbParticles,
            distributionType,
            delta,
            coefficient,
            outputFile);
        break;
    default:
        std::cerr << "Unsupported P value. Allowed: 2,4,6,8,10,12,16\n";
        return 2;
    }

    return 0;
}