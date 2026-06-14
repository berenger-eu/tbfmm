#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbfparams.hpp"
#include "utils/tbftimer.hpp"

#include "kernels/gausskernel/tbfgausskernel.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

template <class RealType>
RealType gaussianKernel(
    const RealType sx,
    const RealType sy,
    const RealType tx,
    const RealType ty,
    const RealType delta,
    const RealType coefficient)
{
    const RealType dx = tx - sx;
    const RealType dy = ty - sy;

    return coefficient * std::exp(-(dx * dx + dy * dy) / delta);
}

std::string makeMonthLabel(const long int month)
{
    std::ostringstream stream;
    stream << std::setw(2) << std::setfill('0') << month;
    return stream.str();
}

template <class RealType, int Dim>
void generateTwoPositiveWellClustersAndGrid(
    std::vector<std::array<RealType, Dim + 1>>& particles,
    const long int gridSize,
    const long int wellsPerCluster)
{
    const long int nbWells = 2 * wellsPerCluster;
    const long int nbGrid = gridSize * gridSize;

    particles.resize(nbWells + nbGrid);

    std::mt19937 rng(42);

    std::normal_distribution<RealType> offset(0.0, 0.025);
    std::uniform_real_distribution<RealType> positiveQ(0.8, 1.2);

    const std::array<RealType, Dim> cluster0Center{{0.20, 0.20}};
    const std::array<RealType, Dim> cluster1Center{{0.80, 0.80}};

    // Первая положительная группа
    for (long int i = 0; i < wellsPerCluster; ++i)
    {
        particles[i][0] =
            std::clamp(cluster0Center[0] + offset(rng), RealType(0), RealType(1));

        particles[i][1] =
            std::clamp(cluster0Center[1] + offset(rng), RealType(0), RealType(1));

        particles[i][2] = positiveQ(rng);
    }

    // Вторая положительная группа
    for (long int i = 0; i < wellsPerCluster; ++i)
    {
        const long int idx = wellsPerCluster + i;

        particles[idx][0] =
            std::clamp(cluster1Center[0] + offset(rng), RealType(0), RealType(1));

        particles[idx][1] =
            std::clamp(cluster1Center[1] + offset(rng), RealType(0), RealType(1));

        particles[idx][2] = positiveQ(rng);
    }

    // Регулярная сетка наблюдения q = 0
    for (long int iy = 0; iy < gridSize; ++iy)
    {
        for (long int ix = 0; ix < gridSize; ++ix)
        {
            const long int gridIdx = iy * gridSize + ix;
            const long int particleIdx = nbWells + gridIdx;

            particles[particleIdx][0] =
                (RealType(ix) + RealType(0.5)) / RealType(gridSize);

            particles[particleIdx][1] =
                (RealType(iy) + RealType(0.5)) / RealType(gridSize);

            particles[particleIdx][2] = RealType(0);
        }
    }
}

template <unsigned int P>
void computeOneDeltaFmm(
    const long int treeHeight,
    const std::vector<std::array<double, 3>>& baseParticles,
    const double delta,
    const double coefficient,
    std::vector<double>& fmmValues,
    double& buildTreeTime,
    double& fmmTime)
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

    TbfTimer timerBuildTree;

    TreeClass tree(
        configuration,
        TbfUtils::make_const(baseParticles));

    timerBuildTree.stop();

    KernelClass kernel(
        configuration,
        RealType(delta),
        RealType(coefficient));

    AlgorithmClass algorithm(configuration, kernel);

    TbfTimer timerFmm;
    algorithm.execute(tree);
    timerFmm.stop();

    fmmValues.assign(baseParticles.size(), RealType(0));

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
                fmmValues[originalIndex] = particleRhsPtr[0][idxPart];
            }
        });

    buildTreeTime = timerBuildTree.getElapsed();
    fmmTime = timerFmm.getElapsed();
}

template <unsigned int P>
void runAccumulatedTimeCase(
    const long int treeHeight,
    const long int gridSize,
    const long int nbMonths,
    const double dt,
    const double kappa,
    const double coefficient,
    const std::string& outputPrefix)
{
    using RealType = double;

    constexpr int Dim = 2;
    constexpr long int wellsPerCluster = 10;
    constexpr long int nbWells = 2 * wellsPerCluster;

    const long int nbGrid = gridSize * gridSize;
    const long int nbParticles = nbWells + nbGrid;

    std::vector<std::array<RealType, Dim + 1>> particles;

    generateTwoPositiveWellClustersAndGrid<RealType, Dim>(
        particles,
        gridSize,
        wellsPerCluster);

    // Сохраняем скважины один раз
    {
        const std::string wellsFile = outputPrefix + "_wells.csv";
        std::ofstream wellsOut(wellsFile);

        wellsOut << "well_id,x,y,q,cluster\n";

        for (long int i = 0; i < nbWells; ++i)
        {
            wellsOut
                << i << ","
                << particles[i][0] << ","
                << particles[i][1] << ","
                << particles[i][2] << ","
                << (i < wellsPerCluster ? 0 : 1)
                << "\n";
        }
    }

    const std::string summaryFile = outputPrefix + "_summary.csv";

    {
        std::ofstream summaryOut(summaryFile);

        summaryOut
            << "Month,Time,GridSize,TreeHeight,P,Kappa,Dt,Coefficient,"
            << "RelativeL2,RMS,RelativeInf,MaxRelativeDiff,"
            << "BuildTreeTimeTotal,TimeFMMTotal,TimeDirectTotal,Speedup\n";
    }

    std::vector<RealType> accumulatedDirect(nbGrid, RealType(0));
    std::vector<RealType> accumulatedFmm(nbGrid, RealType(0));

    std::cout << "Accumulated Gaussian time case\n";
    std::cout << "Grid = " << gridSize << " x " << gridSize << "\n";
    std::cout << "Months = " << nbMonths << "\n";
    std::cout << "dt = " << dt << "\n";
    std::cout << "kappa = " << kappa << "\n";
    std::cout << "P = " << P << "\n";
    std::cout << "Tree height = " << treeHeight << "\n\n";

    for (long int month = 1; month <= nbMonths; ++month)
    {
        double buildTreeTimeTotal = 0.0;
        double fmmTimeTotal = 0.0;
        double directTimeTotal = 0.0;

        std::fill(accumulatedDirect.begin(), accumulatedDirect.end(), RealType(0));
        std::fill(accumulatedFmm.begin(), accumulatedFmm.end(), RealType(0));

        // Накопленное поле к концу месяца:
        // p(x, T_month) = sum_{m=1}^{month} contribution with lag = month - m + 1
        for (long int lag = 1; lag <= month; ++lag)
        {
            const RealType physicalTime = RealType(lag) * RealType(dt);
            const RealType delta =
                RealType(4) * RealType(kappa) * physicalTime;

            // FMM для данного lag
            std::vector<RealType> fmmValues;
            double buildTreeTime = 0.0;
            double fmmTime = 0.0;

            computeOneDeltaFmm<P>(
                treeHeight,
                particles,
                delta,
                coefficient,
                fmmValues,
                buildTreeTime,
                fmmTime);

            buildTreeTimeTotal += buildTreeTime;
            fmmTimeTotal += fmmTime;

            for (long int gridIdx = 0; gridIdx < nbGrid; ++gridIdx)
            {
                const long int particleIdx = nbWells + gridIdx;
                accumulatedFmm[gridIdx] += fmmValues[particleIdx];
            }

            // Direct для данного lag
            TbfTimer timerDirect;

            for (long int gridIdx = 0; gridIdx < nbGrid; ++gridIdx)
            {
                const long int particleIdx = nbWells + gridIdx;

                const RealType tx = particles[particleIdx][0];
                const RealType ty = particles[particleIdx][1];

                RealType value = RealType(0);

                for (long int wellIdx = 0; wellIdx < nbWells; ++wellIdx)
                {
                    const RealType sx = particles[wellIdx][0];
                    const RealType sy = particles[wellIdx][1];
                    const RealType q = particles[wellIdx][2];

                    value += q * gaussianKernel(
                                     sx,
                                     sy,
                                     tx,
                                     ty,
                                     delta,
                                     RealType(coefficient));
                }

                accumulatedDirect[gridIdx] += value;
            }

            timerDirect.stop();
            directTimeTotal += timerDirect.getElapsed();
        }

        const std::string monthLabel = makeMonthLabel(month);
        const std::string mapFile =
            outputPrefix + "_map_month_" + monthLabel + ".csv";

        std::ofstream mapOut(mapFile);

        mapOut
            << "ix,iy,x,y,month,time,direct,fmm,abs_error,rel_error\n";

        RealType errorL2 = RealType(0);
        RealType referenceL2 = RealType(0);
        RealType errorInf = RealType(0);
        RealType referenceInf = RealType(0);
        RealType maxRelativeDiff = RealType(0);

        for (long int iy = 0; iy < gridSize; ++iy)
        {
            for (long int ix = 0; ix < gridSize; ++ix)
            {
                const long int gridIdx = iy * gridSize + ix;
                const long int particleIdx = nbWells + gridIdx;

                const RealType x = particles[particleIdx][0];
                const RealType y = particles[particleIdx][1];

                const RealType direct = accumulatedDirect[gridIdx];
                const RealType fmm = accumulatedFmm[gridIdx];

                const RealType diff = fmm - direct;
                const RealType absError = std::abs(diff);
                const RealType relError =
                    absError / std::max(std::abs(direct), RealType(1e-12));

                errorL2 += diff * diff;
                referenceL2 += direct * direct;

                errorInf = std::max(errorInf, absError);
                referenceInf = std::max(referenceInf, std::abs(direct));
                maxRelativeDiff = std::max(maxRelativeDiff, relError);

                mapOut
                    << ix << ","
                    << iy << ","
                    << x << ","
                    << y << ","
                    << month << ","
                    << RealType(month) * RealType(dt) << ","
                    << direct << ","
                    << fmm << ","
                    << absError << ","
                    << relError << "\n";
            }
        }

        const RealType relativeL2 =
            std::sqrt(errorL2)
            / std::max(std::sqrt(referenceL2), RealType(1e-30));

        const RealType rmsError =
            std::sqrt(errorL2 / RealType(nbGrid));

        const RealType relativeInf =
            errorInf / std::max(referenceInf, RealType(1e-30));

        {
            std::ofstream summaryOut(summaryFile, std::ios::app);

            summaryOut
                << month << ","
                << RealType(month) * RealType(dt) << ","
                << gridSize << ","
                << treeHeight << ","
                << P << ","
                << kappa << ","
                << dt << ","
                << coefficient << ","
                << relativeL2 << ","
                << rmsError << ","
                << relativeInf << ","
                << maxRelativeDiff << ","
                << buildTreeTimeTotal << ","
                << fmmTimeTotal << ","
                << directTimeTotal << ","
                << (fmmTimeTotal > 0.0 ? directTimeTotal / fmmTimeTotal : 0.0)
                << "\n";
        }

        std::cout
            << "month = " << month
            << " | time = " << RealType(month) * RealType(dt)
            << " | RelativeL2 = " << relativeL2
            << " | FMM total = " << fmmTimeTotal
            << " | Direct total = " << directTimeTotal
            << " | map = " << mapFile
            << "\n";
    }
}

int main(int argc, char** argv)
{
    const long int treeHeight =
        TbfParams::GetValue<long int>(
            argc,
            argv,
            {"-th", "--tree-height"},
            5);

    const long int gridSize =
        TbfParams::GetValue<long int>(
            argc,
            argv,
            {"-g", "--grid-size"},
            100);

    const long int nbMonths =
        TbfParams::GetValue<long int>(
            argc,
            argv,
            {"-m", "--months"},
            12);

    const unsigned int p =
        TbfParams::GetValue<unsigned int>(
            argc,
            argv,
            {"-p", "--order"},
            8);

    const double dt =
        TbfParams::GetValue<double>(
            argc,
            argv,
            {"-dt", "--time-step"},
            0.0125);

    const double kappa =
        TbfParams::GetValue<double>(
            argc,
            argv,
            {"-kappa", "--kappa"},
            1.0);

    const double coefficient =
        TbfParams::GetValue<double>(
            argc,
            argv,
            {"-coef", "--coefficient"},
            1.0);

    const std::string outputPrefix =
        TbfParams::GetValue<std::string>(
            argc,
            argv,
            {"-o", "--output-prefix"},
            "accum_case");

    switch (p)
    {
    case 2:
        runAccumulatedTimeCase<2>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    case 4:
        runAccumulatedTimeCase<4>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    case 6:
        runAccumulatedTimeCase<6>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    case 8:
        runAccumulatedTimeCase<8>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    case 10:
        runAccumulatedTimeCase<10>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    case 12:
        runAccumulatedTimeCase<12>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    case 16:
        runAccumulatedTimeCase<16>(
            treeHeight,
            gridSize,
            nbMonths,
            dt,
            kappa,
            coefficient,
            outputPrefix);
        break;
    default:
        std::cerr << "Unsupported P value. Allowed: 2,4,6,8,10,12,16\n";
        return 2;
    }

    return 0;
}