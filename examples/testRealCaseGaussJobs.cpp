#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbfparams.hpp"
#include "utils/tbftimer.hpp"

#include "kernels/gausskernel/tbfgausskernel.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


struct Target
{
    long int targetId;
    double xNorm;
    double yNorm;
};


struct Source
{
    double xNorm;
    double yNorm;
    double weightPa;
};


struct Job
{
    long int jobId;
    double deltaNorm;
    std::string sourceFile;
};


std::string joinPath(const std::string& dir, const std::string& file)
{
    if (dir.empty())
    {
        return file;
    }

    const char last = dir.back();

    if (last == '/' || last == '\\')
    {
        return dir + file;
    }

#ifdef _WIN32
    return dir + "\\" + file;
#else
    return dir + "/" + file;
#endif
}


std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> result;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ','))
    {
        result.push_back(cell);
    }

    return result;
}


std::vector<Target> readTargets(const std::string& filename)
{
    std::ifstream in(filename);

    if (!in)
    {
        throw std::runtime_error("Cannot open targets file: " + filename);
    }

    std::vector<Target> targets;

    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        const auto cols = splitCsvLine(line);

        if (cols.size() < 7)
        {
            throw std::runtime_error("Bad targets.csv line: " + line);
        }

        // target_id,ix,iy,x,y,x_norm,y_norm,inside
        Target t;
        t.targetId = std::stol(cols[0]);
        t.xNorm = std::stod(cols[5]);
        t.yNorm = std::stod(cols[6]);

        targets.push_back(t);
    }

    return targets;
}


std::vector<Source> readSources(const std::string& filename)
{
    std::ifstream in(filename);

    if (!in)
    {
        throw std::runtime_error("Cannot open sources file: " + filename);
    }

    std::vector<Source> sources;

    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        const auto cols = splitCsvLine(line);

        if (cols.size() < 7)
        {
            throw std::runtime_error("Bad sources csv line: " + line);
        }

        // source_id,well_name,x,y,x_norm,y_norm,weight_pa
        Source s;
        s.xNorm = std::stod(cols[4]);
        s.yNorm = std::stod(cols[5]);
        s.weightPa = std::stod(cols[6]);

        sources.push_back(s);
    }

    return sources;
}


std::vector<Job> readJobs(const std::string& filename)
{
    std::ifstream in(filename);

    if (!in)
    {
        throw std::runtime_error("Cannot open jobs file: " + filename);
    }

    std::vector<Job> jobs;

    std::string line;
    std::getline(in, line); // header

    while (std::getline(in, line))
    {
        if (line.empty())
        {
            continue;
        }

        const auto cols = splitCsvLine(line);

        if (cols.size() < 8)
        {
            throw std::runtime_error("Bad jobs.csv line: " + line);
        }

        // job_id,month_index,source_month_zero,quad_id,
        // s_seconds,delta_m2,delta_norm,source_file
        Job job;
        job.jobId = std::stol(cols[0]);
        job.deltaNorm = std::stod(cols[6]);
        job.sourceFile = cols[7];

        jobs.push_back(job);
    }

    return jobs;
}


template <unsigned int P>
void computeOneJobFmm(
    const long int treeHeight,
    const std::vector<Target>& targets,
    const std::vector<Source>& sources,
    const double deltaNorm,
    std::vector<double>& jobValues)
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

    const long int nbSources = static_cast<long int>(sources.size());
    const long int nbTargets = static_cast<long int>(targets.size());
    const long int nbParticles = nbSources + nbTargets;

    std::vector<std::array<RealType, Dim + 1>> particles(nbParticles);

    // Sources: weight_pa is q in kernel.
    for (long int i = 0; i < nbSources; ++i)
    {
        particles[i][0] = RealType(sources[i].xNorm);
        particles[i][1] = RealType(sources[i].yNorm);
        particles[i][2] = RealType(sources[i].weightPa);
    }

    // Targets: zero weight.
    for (long int i = 0; i < nbTargets; ++i)
    {
        const long int idx = nbSources + i;

        particles[idx][0] = RealType(targets[i].xNorm);
        particles[idx][1] = RealType(targets[i].yNorm);
        particles[idx][2] = RealType(0.0);
    }

    std::array<RealType, Dim> boxWidths{{1.0, 1.0}};
    std::array<RealType, Dim> boxCenter{{0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(
        treeHeight,
        boxWidths,
        boxCenter);

    TreeClass tree(
        configuration,
        TbfUtils::make_const(particles));

    // coefficient = 1, потому что weight_pa уже содержит физический множитель.
    KernelClass kernel(
        configuration,
        RealType(deltaNorm),
        RealType(1.0));

    AlgorithmClass algorithm(configuration, kernel);
    algorithm.execute(tree);

    std::vector<RealType> allValues(nbParticles, RealType(0));

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
                allValues[originalIndex] = particleRhsPtr[0][idxPart];
            }
        });

    jobValues.assign(nbTargets, RealType(0));

    for (long int i = 0; i < nbTargets; ++i)
    {
        const long int particleIdx = nbSources + i;
        jobValues[i] = allValues[particleIdx];
    }
}


template <unsigned int P>
void runAllJobs(
    const std::string& inputDir,
    const std::string& outputFile,
    const long int treeHeight)
{
    const std::string targetsFile = joinPath(inputDir, "targets.csv");
    const std::string jobsFile = joinPath(inputDir, "jobs.csv");

    const auto targets = readTargets(targetsFile);
    const auto jobs = readJobs(jobsFile);

    std::vector<double> totalValues(targets.size(), 0.0);

    std::cout << "Real case Gauss FMM jobs\n";
    std::cout << "inputDir = " << inputDir << "\n";
    std::cout << "targets = " << targets.size() << "\n";
    std::cout << "jobs = " << jobs.size() << "\n";
    std::cout << "P = " << P << "\n";
    std::cout << "treeHeight = " << treeHeight << "\n\n";

    TbfTimer timerTotal;

    for (const auto& job : jobs)
    {
        const std::string sourcePath = joinPath(inputDir, job.sourceFile);

        const auto sources = readSources(sourcePath);

        std::vector<double> jobValues;

        TbfTimer timerJob;

        computeOneJobFmm<P>(
            treeHeight,
            targets,
            sources,
            job.deltaNorm,
            jobValues);

        timerJob.stop();

        if (jobValues.size() != totalValues.size())
        {
            throw std::runtime_error("jobValues size mismatch.");
        }

        for (std::size_t i = 0; i < totalValues.size(); ++i)
        {
            totalValues[i] += jobValues[i];
        }

        std::cout
            << "job = " << job.jobId
            << " | sources = " << sources.size()
            << " | deltaNorm = " << job.deltaNorm
            << " | time = " << timerJob.getElapsed()
            << "\n";
    }

    timerTotal.stop();

    std::ofstream out(outputFile);

    if (!out)
    {
        throw std::runtime_error("Cannot open output file: " + outputFile);
    }

    out << "target_id,fmm_pa\n";

    out << std::setprecision(17);

    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        out
            << targets[i].targetId << ","
            << totalValues[i] << "\n";
    }

    std::cout << "\nSaved: " << outputFile << "\n";
    std::cout << "Total FMM jobs time = " << timerTotal.getElapsed() << "\n";
}


int main(int argc, char** argv)
{
    const std::string inputDir =
        TbfParams::GetValue<std::string>(
            argc,
            argv,
            {"--input-dir"},
            "real_case_fmm_input");

    const std::string outputFile =
        TbfParams::GetValue<std::string>(
            argc,
            argv,
            {"--output"},
            joinPath(inputDir, "fmm_total.csv"));

    const long int treeHeight =
        TbfParams::GetValue<long int>(
            argc,
            argv,
            {"-th", "--tree-height"},
            5);

    const unsigned int p =
        TbfParams::GetValue<unsigned int>(
            argc,
            argv,
            {"-p", "--order"},
            8);

    try
    {
        switch (p)
        {
        case 2:
            runAllJobs<2>(inputDir, outputFile, treeHeight);
            break;
        case 4:
            runAllJobs<4>(inputDir, outputFile, treeHeight);
            break;
        case 6:
            runAllJobs<6>(inputDir, outputFile, treeHeight);
            break;
        case 8:
            runAllJobs<8>(inputDir, outputFile, treeHeight);
            break;
        case 10:
            runAllJobs<10>(inputDir, outputFile, treeHeight);
            break;
        case 12:
            runAllJobs<12>(inputDir, outputFile, treeHeight);
            break;
        case 16:
            runAllJobs<16>(inputDir, outputFile, treeHeight);
            break;
        default:
            std::cerr << "Unsupported P value. Allowed: 2,4,6,8,10,12,16\n";
            return 2;
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }

    return 0;
}