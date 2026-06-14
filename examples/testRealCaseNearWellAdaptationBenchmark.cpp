/**
 * Semi-synthetic adaptation benchmark on real-field geometry.
 *
 * Goal: measure repeated application of the same Gaussian interaction
 * operator when geometry and kernel parameter delta are fixed, while source
 * intensities change between adaptation iterations. This is the natural
 * online loop of a local near-well adaptation / inverse problem.
 *
 * Real data used:
 *   - well coordinates from real_wells.csv;
 *   - reservoir contour from contour.csv;
 *   - pressure-map target grid inside the real bounding box.
 *
 * Semi-synthetic part:
 *   - each real well is replaced by M local source points on a small circle;
 *   - source weights vary deterministically between iterations, imitating
 *     iterative correction of near-well parameters.
 *
 * Direct and source-target TSM-FMM calculate exactly the same operator
 * application. Trees are built once per discretization level and reused for
 * all adaptation iterations. Library files are not modified.
 */
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"
#include "kernels/gausskernel/tbfgausskernel.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <numeric>
#include <random>

namespace fs = std::filesystem;

constexpr double PI = 3.14159265358979323846;


// ==========================================================
// ЛОКАЛЬНАЯ SOURCE-TARGET ОБЁРТКА БЕЗ ПРАВОК БИБЛИОТЕКИ
// ==========================================================

/**
 * The stock TbfTreeTsm constructor estimates a block size through a helper
 * whose default space-index type may not preserve our explicit 2D type when
 * particles contain an additional third value {x, y, weight}.
 *
 * This local wrapper composes two ordinary TbfTree instances with an explicit
 * 2D SpaceIndexType and forwards the interface required by TSM algorithms.
 */
template <
    class RealType,
    class DataType,
    long int NbDataValuesPerParticle,
    class RhsType,
    long int NbRhsValuesPerParticle,
    class MultipoleClass,
    class LocalClass,
    class SpaceIndexType>
class LocalTreeTsm
{
public:
    using TreeClass =
        TbfTree<
            RealType,
            DataType,
            NbDataValuesPerParticle,
            RhsType,
            NbRhsValuesPerParticle,
            MultipoleClass,
            LocalClass,
            SpaceIndexType>;

    using SpacialConfiguration =
        TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

    using IndexType = typename SpaceIndexType::IndexType;

private:
    const SpacialConfiguration configuration;
    const SpaceIndexType spaceSystem;

    TreeClass treeSource;
    TreeClass treeTarget;

public:
    template <class ParticleContainerSource, class ParticleContainerTarget>
    LocalTreeTsm(
        const SpacialConfiguration& inConfiguration,
        const ParticleContainerSource& inParticleSourcePositions,
        const ParticleContainerTarget& inParticleTargetPositions,
        const long int inNbElementsPerBlock,
        const bool inOneGroupPerParent = false)
        : configuration(inConfiguration),
          spaceSystem(inConfiguration),
          treeSource(
              inConfiguration,
              inParticleSourcePositions,
              inNbElementsPerBlock,
              inOneGroupPerParent),
          treeTarget(
              inConfiguration,
              inParticleTargetPositions,
              inNbElementsPerBlock,
              inOneGroupPerParent)
    {
    }

    long int getNbParticles() const
    {
        return treeSource.getNbParticles()
            + treeTarget.getNbParticles();
    }

    long int getNbElementsPerGroupSource() const
    {
        return treeSource.getNbElementsPerGroup();
    }

    long int getNbElementsPerGroupTarget() const
    {
        return treeTarget.getNbElementsPerGroup();
    }

    const SpacialConfiguration& getSpacialConfiguration() const
    {
        return configuration;
    }

    const SpaceIndexType& getSpacialSystem() const
    {
        return spaceSystem;
    }

    long int getHeight() const
    {
        return configuration.getTreeHeight();
    }

    long int getNbCellGroupsAtLevelSource(const long int inLevel) const
    {
        return treeSource.getNbCellGroupsAtLevel(inLevel);
    }

    auto& getCellGroupsAtLevelSource(const long int inLevel)
    {
        return treeSource.getCellGroupsAtLevel(inLevel);
    }

    const auto& getCellGroupsAtLevelSource(const long int inLevel) const
    {
        return treeSource.getCellGroupsAtLevel(inLevel);
    }

    auto& getLeafGroupsSource()
    {
        return treeSource.getLeafGroups();
    }

    const auto& getLeafGroupsSource() const
    {
        return treeSource.getLeafGroups();
    }

    long int getNbParticleGroupsSource() const
    {
        return treeSource.getNbParticleGroups();
    }

    auto& getParticleGroupsSource()
    {
        return treeSource.getParticleGroups();
    }

    const auto& getParticleGroupsSource() const
    {
        return treeSource.getParticleGroups();
    }

    long int getNbCellGroupsAtLevelTarget(const long int inLevel) const
    {
        return treeTarget.getNbCellGroupsAtLevel(inLevel);
    }

    auto& getCellGroupsAtLevelTarget(const long int inLevel)
    {
        return treeTarget.getCellGroupsAtLevel(inLevel);
    }

    const auto& getCellGroupsAtLevelTarget(const long int inLevel) const
    {
        return treeTarget.getCellGroupsAtLevel(inLevel);
    }

    auto& getLeafGroupsTarget()
    {
        return treeTarget.getLeafGroups();
    }

    const auto& getLeafGroupsTarget() const
    {
        return treeTarget.getLeafGroups();
    }

    long int getNbParticleGroupsTarget() const
    {
        return treeTarget.getNbParticleGroups();
    }

    auto& getParticleGroupsTarget()
    {
        return treeTarget.getParticleGroups();
    }

    const auto& getParticleGroupsTarget() const
    {
        return treeTarget.getParticleGroups();
    }

    auto findGroupWithCellSource(
        const long int inLevel,
        const IndexType inMIndex)
    {
        return treeSource.findGroupWithCell(inLevel, inMIndex);
    }

    auto findGroupWithCellTarget(
        const long int inLevel,
        const IndexType inMIndex)
    {
        return treeTarget.findGroupWithCell(inLevel, inMIndex);
    }

    auto findGroupWithLeafSource(const IndexType inMIndex)
    {
        return treeSource.findGroupWithLeaf(inMIndex);
    }

    auto findGroupWithLeafTarget(const IndexType inMIndex)
    {
        return treeTarget.findGroupWithLeaf(inMIndex);
    }

    template <class FuncClass>
    void applyToAllCellsSource(FuncClass&& inFunc)
    {
        treeSource.applyToAllCells(std::forward<FuncClass>(inFunc));
    }

    template <class FuncClass>
    void applyToAllCellsTarget(FuncClass&& inFunc)
    {
        treeTarget.applyToAllCells(std::forward<FuncClass>(inFunc));
    }

    template <class FuncClass>
    void applyToAllLeavesSource(FuncClass&& inFunc)
    {
        treeSource.applyToAllLeaves(std::forward<FuncClass>(inFunc));
    }

    template <class FuncClass>
    void applyToAllLeavesTarget(FuncClass&& inFunc)
    {
        treeTarget.applyToAllLeaves(std::forward<FuncClass>(inFunc));
    }

    auto getAllParticlesDataSource()
    {
        return treeSource.getAllParticlesData();
    }

    auto getAllParticlesDataTarget()
    {
        return treeTarget.getAllParticlesData();
    }

    auto getAllParticlesRhsTarget()
    {
        return treeTarget.getAllParticlesRhs();
    }

    void rebuild()
    {
        treeSource.rebuild();
        treeTarget.rebuild();
    }
};


// ==========================================================
// CONFIG
// ==========================================================

struct Config
{
    fs::path inputDir = "real_case_integrated_input";
    fs::path outputDir = "real_case_adaptation_output";

    int gridSize = 180;
    int iterations = 20;
    int warmupIterations = 1;

    long int treeHeight = 5;
    unsigned int pOrder = 8;
    long int blockSize = 100;

    double localRadiusM = 10.0;
    double perturbationAmplitude = 0.20;
    double lagMonths = 1.0;

    double kMd = 24.0;
    double muCp = 1.69;
    double porosity = 0.19;
    double ctPaInv = 2.236e-9;
    double dtDays = 30.4375;

    std::vector<int> pointsPerWellList{1, 4, 8, 16, 32, 64};
};

constexpr double MD_TO_M2 = 9.869233e-16;
constexpr double CP_TO_PA_S = 1e-3;
constexpr double DAY_TO_SEC = 24.0 * 3600.0;

struct Well
{
    std::string name;
    double x = 0.0;
    double y = 0.0;
};

struct Point
{
    long int targetId = 0;
    int ix = 0;
    int iy = 0;
    double x = 0.0;
    double y = 0.0;
    bool inside = false;
};

struct LocalSource
{
    std::size_t wellIndex = 0;
    int localIndex = 0;
    double x = 0.0;
    double y = 0.0;
    double baseWeight = 0.0;
};

struct Normalization
{
    double centerX = 0.0;
    double centerY = 0.0;
    double boxSize = 1.0;

    double xNorm(const double x) const { return (x - centerX) / boxSize + 0.5; }
    double yNorm(const double y) const { return (y - centerY) / boxSize + 0.5; }
};

struct ErrorMetrics
{
    double relativeL2 = 0.0;
    double relativeInf = 0.0;
    double maxAbsError = 0.0;
    double meanAbsError = 0.0;
};

std::string trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) return "";
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;
    while (std::getline(stream, cell, ',')) cells.push_back(trim(cell));
    return cells;
}

std::unordered_map<std::string, std::size_t> buildHeaderMap(const std::string& headerLine)
{
    const auto headers = splitCsvLine(headerLine);
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t i = 0; i < headers.size(); ++i) result[headers[i]] = i;
    return result;
}

std::size_t requireColumn(
    const std::unordered_map<std::string, std::size_t>& header,
    const std::string& column)
{
    const auto iterator = header.find(column);
    if (iterator == header.end()) throw std::runtime_error("Missing CSV column: " + column);
    return iterator->second;
}

std::vector<int> parseIntList(const std::string& text)
{
    std::vector<int> result;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ','))
    {
        const int value = std::stoi(trim(item));
        if (value <= 0) throw std::runtime_error("points-per-well values must be positive");
        result.push_back(value);
    }
    if (result.empty()) throw std::runtime_error("Empty points-per-well list");
    return result;
}

Config parseArgs(const int argc, char** argv)
{
    Config config;
    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];
        auto value = [&]() -> std::string
        {
            if (index + 1 >= argc) throw std::runtime_error("Missing value after " + arg);
            return argv[++index];
        };

        if (arg == "--input-dir") config.inputDir = value();
        else if (arg == "--output-dir") config.outputDir = value();
        else if (arg == "--grid-size") config.gridSize = std::stoi(value());
        else if (arg == "--iterations") config.iterations = std::stoi(value());
        else if (arg == "--warmup") config.warmupIterations = std::stoi(value());
        else if (arg == "--tree-height") config.treeHeight = std::stol(value());
        else if (arg == "--order") config.pOrder = static_cast<unsigned int>(std::stoul(value()));
        else if (arg == "--block-size") config.blockSize = std::stol(value());
        else if (arg == "--local-radius-m") config.localRadiusM = std::stod(value());
        else if (arg == "--amplitude") config.perturbationAmplitude = std::stod(value());
        else if (arg == "--lag-months") config.lagMonths = std::stod(value());
        else if (arg == "--points-per-well") config.pointsPerWellList = parseIntList(value());
        else if (arg == "--k-md") config.kMd = std::stod(value());
        else if (arg == "--mu-cp") config.muCp = std::stod(value());
        else if (arg == "--porosity") config.porosity = std::stod(value());
        else if (arg == "--ct-pa-inv") config.ctPaInv = std::stod(value());
        else if (arg == "--dt-days") config.dtDays = std::stod(value());
        else throw std::runtime_error("Unknown argument: " + arg);
    }

    if (config.gridSize < 2) throw std::runtime_error("grid-size must be >= 2");
    if (config.iterations <= 0) throw std::runtime_error("iterations must be positive");
    if (config.warmupIterations < 0) throw std::runtime_error("warmup must be non-negative");
    if (config.localRadiusM < 0.0) throw std::runtime_error("local-radius-m must be non-negative");
    if (config.lagMonths <= 0.0) throw std::runtime_error("lag-months must be positive");
    return config;
}

double computeKappa(const Config& config)
{
    const double kM2 = config.kMd * MD_TO_M2;
    const double muPaS = config.muCp * CP_TO_PA_S;
    return kM2 / (config.porosity * muPaS * config.ctPaInv);
}

std::vector<Well> readWells(const fs::path& filename)
{
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open wells file: " + filename.string());

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty wells file");
    const auto header = buildHeaderMap(line);
    const auto nameColumn = requireColumn(header, "well_name");
    const auto xColumn = requireColumn(header, "x");
    const auto yColumn = requireColumn(header, "y");

    std::vector<Well> wells;
    while (std::getline(input, line))
    {
        if (trim(line).empty()) continue;
        const auto cells = splitCsvLine(line);
        wells.push_back({cells.at(nameColumn), std::stod(cells.at(xColumn)), std::stod(cells.at(yColumn))});
    }
    if (wells.empty()) throw std::runtime_error("No wells loaded");
    return wells;
}

std::vector<std::array<double, 2>> readOuterContour(const fs::path& filename)
{
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("Cannot open contour file: " + filename.string());

    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty contour file");
    const auto header = buildHeaderMap(line);
    const auto idColumn = requireColumn(header, "contour_id");
    const auto xColumn = requireColumn(header, "x");
    const auto yColumn = requireColumn(header, "y");

    std::vector<std::array<double, 2>> contour;
    while (std::getline(input, line))
    {
        if (trim(line).empty()) continue;
        const auto cells = splitCsvLine(line);
        if (std::stoi(cells.at(idColumn)) != 0) continue;
        contour.push_back({std::stod(cells.at(xColumn)), std::stod(cells.at(yColumn))});
    }
    if (contour.size() < 3) throw std::runtime_error("Outer contour has fewer than 3 points");
    return contour;
}

bool pointInsidePolygon(
    const double x,
    const double y,
    const std::vector<std::array<double, 2>>& polygon)
{
    bool inside = false;
    for (std::size_t i = 0, j = polygon.size() - 1; i < polygon.size(); j = i++)
    {
        const double xi = polygon[i][0];
        const double yi = polygon[i][1];
        const double xj = polygon[j][0];
        const double yj = polygon[j][1];
        const bool intersects = ((yi > y) != (yj > y))
            && (x < (xj - xi) * (y - yi)
                / ((yj - yi) + std::numeric_limits<double>::epsilon()) + xi);
        if (intersects) inside = !inside;
    }
    return inside;
}

std::vector<Point> buildGrid(
    const std::vector<std::array<double, 2>>& contour,
    const int gridSize)
{
    double minX = contour[0][0], maxX = contour[0][0];
    double minY = contour[0][1], maxY = contour[0][1];
    for (const auto& p : contour)
    {
        minX = std::min(minX, p[0]); maxX = std::max(maxX, p[0]);
        minY = std::min(minY, p[1]); maxY = std::max(maxY, p[1]);
    }

    std::vector<Point> grid;
    grid.reserve(static_cast<std::size_t>(gridSize) * gridSize);
    for (int iy = 0; iy < gridSize; ++iy)
    {
        const double y = minY + (maxY - minY) * static_cast<double>(iy) / (gridSize - 1);
        for (int ix = 0; ix < gridSize; ++ix)
        {
            const double x = minX + (maxX - minX) * static_cast<double>(ix) / (gridSize - 1);
            grid.push_back({static_cast<long int>(iy) * gridSize + ix, ix, iy, x, y,
                            pointInsidePolygon(x, y, contour)});
        }
    }
    return grid;
}

std::vector<LocalSource> buildLocalSources(
    const std::vector<Well>& wells,
    const int pointsPerWell,
    const double radiusM)
{
    std::vector<LocalSource> sources;
    sources.reserve(wells.size() * static_cast<std::size_t>(pointsPerWell));

    for (std::size_t wellIndex = 0; wellIndex < wells.size(); ++wellIndex)
    {
        // Deterministic synthetic base total weight for the well.
        // Negative sign imitates production. The exact value is irrelevant for timing.
        const double totalWeight = -(50.0 + 4.0 * static_cast<double>(wellIndex % 19));

        for (int localIndex = 0; localIndex < pointsPerWell; ++localIndex)
        {
            const double angle = pointsPerWell == 1
                ? 0.0
                : 2.0 * PI * static_cast<double>(localIndex) / pointsPerWell;
            const double radius = pointsPerWell == 1 ? 0.0 : radiusM;
            sources.push_back({wellIndex,
                               localIndex,
                               wells[wellIndex].x + radius * std::cos(angle),
                               wells[wellIndex].y + radius * std::sin(angle),
                               totalWeight / static_cast<double>(pointsPerWell)});
        }
    }
    return sources;
}

Normalization buildNormalization(
    const std::vector<LocalSource>& sources,
    const std::vector<Point>& targets)
{
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& s : sources)
    {
        minX = std::min(minX, s.x); maxX = std::max(maxX, s.x);
        minY = std::min(minY, s.y); maxY = std::max(maxY, s.y);
    }
    for (const auto& t : targets)
    {
        minX = std::min(minX, t.x); maxX = std::max(maxX, t.x);
        minY = std::min(minY, t.y); maxY = std::max(maxY, t.y);
    }

    Normalization n;
    n.centerX = 0.5 * (minX + maxX);
    n.centerY = 0.5 * (minY + maxY);
    n.boxSize = 1.01 * std::max(maxX - minX, maxY - minY);
    if (n.boxSize <= 0.0) throw std::runtime_error("Invalid normalization box");
    return n;
}

std::vector<double> buildIterationWeights(
    const std::vector<LocalSource>& sources,
    const int iteration,
    const double amplitude)
{
    std::vector<double> weights(sources.size(), 0.0);
    for (std::size_t index = 0; index < sources.size(); ++index)
    {
        const auto& source = sources[index];
        const double phase = 0.37 * static_cast<double>(iteration)
            + 0.11 * static_cast<double>(source.wellIndex)
            + 0.73 * static_cast<double>(source.localIndex);
        weights[index] = source.baseWeight * (1.0 + amplitude * std::sin(phase));
    }
    return weights;
}

std::vector<double> computeDirect(
    const std::vector<LocalSource>& sources,
    const std::vector<Point>& targets,
    const std::vector<double>& weights,
    const double deltaPhysical)
{
    std::vector<double> values(targets.size(), 0.0);
    for (std::size_t sourceIndex = 0; sourceIndex < sources.size(); ++sourceIndex)
    {
        const auto& source = sources[sourceIndex];
        const double weight = weights[sourceIndex];
        if (std::abs(weight) < 1e-30) continue;

        for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex)
        {
            const double dx = targets[targetIndex].x - source.x;
            const double dy = targets[targetIndex].y - source.y;
            values[targetIndex] += weight * std::exp(-(dx * dx + dy * dy) / deltaPhysical);
        }
    }
    return values;
}

ErrorMetrics computeErrors(
    const std::vector<Point>& targets,
    const std::vector<double>& direct,
    const std::vector<double>& fmm)
{
    double errL2 = 0.0, refL2 = 0.0, errInf = 0.0, refInf = 0.0, sumAbs = 0.0;
    std::size_t count = 0;
    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        if (!targets[i].inside) continue;
        const double diff = fmm[i] - direct[i];
        const double absDiff = std::abs(diff);
        errL2 += diff * diff;
        refL2 += direct[i] * direct[i];
        errInf = std::max(errInf, absDiff);
        refInf = std::max(refInf, std::abs(direct[i]));
        sumAbs += absDiff;
        ++count;
    }
    ErrorMetrics m;
    m.relativeL2 = std::sqrt(errL2) / std::max(std::sqrt(refL2), 1e-30);
    m.relativeInf = errInf / std::max(refInf, 1e-30);
    m.maxAbsError = errInf;
    m.meanAbsError = count ? sumAbs / static_cast<double>(count) : 0.0;
    return m;
}

template <class TreeClass>
void clearFmmState(TreeClass& tree)
{
    tree.applyToAllCellsSource(
        [](const long int, auto&&, auto cellMultipole, auto&&)
        {
            if (cellMultipole) cellMultipole->get().fill(0.0);
        });
    tree.applyToAllCellsTarget(
        [](const long int, auto&&, auto&&, auto cellLocal)
        {
            if (cellLocal) cellLocal->get().fill(0.0);
        });
    tree.applyToAllLeavesTarget(
        [](auto&& leafHeader, const long int*, auto&&, auto&& rhs)
        {
            for (long int i = 0; i < leafHeader.nbParticles; ++i) rhs[0][i] = 0.0;
        });
}

template <class TreeClass>
void updateSourceWeights(TreeClass& tree, const std::vector<double>& weights)
{
    tree.applyToAllLeavesSource(
        [&weights](auto&& leafHeader, const long int* indexes, auto&& data, auto&&)
        {
            for (long int i = 0; i < leafHeader.nbParticles; ++i)
            {
                data[2][i] = weights.at(static_cast<std::size_t>(indexes[i]));
            }
        });
}

template <class TreeClass>
std::vector<double> readTargetRhs(TreeClass& tree, const std::size_t targetCount)
{
    std::vector<double> values(targetCount, 0.0);
    tree.applyToAllLeavesTarget(
        [&values](auto&& leafHeader, const long int* indexes, auto&&, auto&& rhs)
        {
            for (long int i = 0; i < leafHeader.nbParticles; ++i)
            {
                values.at(static_cast<std::size_t>(indexes[i])) = rhs[0][i];
            }
        });
    return values;
}

void saveLastMap(
    const fs::path& filename,
    const std::vector<Point>& targets,
    const std::vector<double>& direct,
    const std::vector<double>& fmm)
{
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("Cannot write map file: " + filename.string());
    output << std::setprecision(17);
    output << "target_id,ix,iy,x,y,inside,direct,fmm,error,abs_error\n";
    for (std::size_t i = 0; i < targets.size(); ++i)
    {
        const double diff = fmm[i] - direct[i];
        output << targets[i].targetId << ',' << targets[i].ix << ',' << targets[i].iy << ','
               << targets[i].x << ',' << targets[i].y << ',' << (targets[i].inside ? 1 : 0) << ','
               << direct[i] << ',' << fmm[i] << ',' << diff << ',' << std::abs(diff) << '\n';
    }
}

template <unsigned int P>
void runBenchmarkForOrder(
    const Config& config,
    const std::vector<Well>& wells,
    const std::vector<Point>& targets)
{
    using RealType = double;
    constexpr int Dim = 2;
    constexpr long int NbDataValuesPerParticle = Dim + 1;
    constexpr long int NbRhsValuesPerParticle = 1;
    constexpr long int VectorSize = (P + 1) * (P + 1);

    using MultipoleClass = std::array<RealType, VectorSize>;
    using LocalClass = std::array<RealType, VectorSize>;
    using SpaceIndexType = TbfDefaultSpaceIndexType2D<RealType>;
    using KernelClass = TbfGaussKernel<RealType, P, SpaceIndexType>;
    using AlgorithmClass = TbfAlgorithmTsm<RealType, KernelClass, SpaceIndexType>;
    using TreeClass = LocalTreeTsm<RealType,
                                   RealType,
                                   NbDataValuesPerParticle,
                                   RealType,
                                   NbRhsValuesPerParticle,
                                   MultipoleClass,
                                   LocalClass,
                                   SpaceIndexType>;

    const double kappa = computeKappa(config);
    const double lagSeconds = config.lagMonths * config.dtDays * DAY_TO_SEC;
    const double deltaPhysical = 4.0 * kappa * lagSeconds;

    fs::create_directories(config.outputDir);
    const fs::path summaryFile = config.outputDir / ("adaptation_benchmark_p" + std::to_string(P) + "_h" + std::to_string(config.treeHeight) + ".csv");
    std::ofstream summary(summaryFile);
    if (!summary) throw std::runtime_error("Cannot write summary: " + summaryFile.string());
    summary << std::setprecision(17);
    summary << "points_per_well,n_wells,n_sources,n_targets,iterations,warmup_iterations,P,tree_height,block_size,lag_months,delta_physical,tree_build_time_sec,direct_total_sec,fmm_total_sec,direct_mean_sec,fmm_mean_sec,speedup,relative_l2,relative_inf,max_abs_error,mean_abs_error\n";

    std::cout << "Adaptation benchmark: P=" << P << ", H=" << config.treeHeight << "\n";
    std::cout << "wells=" << wells.size() << " targets=" << targets.size() << " iterations=" << config.iterations << "\n";
    std::cout << "lagMonths=" << config.lagMonths << " deltaPhysical=" << deltaPhysical << "\n\n";

    for (const int pointsPerWell : config.pointsPerWellList)
    {
        const auto sources = buildLocalSources(wells, pointsPerWell, config.localRadiusM);
        const Normalization normalization = buildNormalization(sources, targets);
        const double deltaNormalized = deltaPhysical / (normalization.boxSize * normalization.boxSize);

        std::vector<std::array<RealType, Dim + 1>> sourceParticles(sources.size());
        for (std::size_t i = 0; i < sources.size(); ++i)
        {
            sourceParticles[i] = {normalization.xNorm(sources[i].x), normalization.yNorm(sources[i].y), 0.0};
        }
        std::vector<std::array<RealType, Dim + 1>> targetParticles(targets.size());
        for (std::size_t i = 0; i < targets.size(); ++i)
        {
            targetParticles[i] = {normalization.xNorm(targets[i].x), normalization.yNorm(targets[i].y), 0.0};
        }

        const std::array<RealType, Dim> boxWidths{{1.0, 1.0}};
        const std::array<RealType, Dim> boxCenter{{0.5, 0.5}};
        const TbfSpacialConfiguration<RealType, Dim> configuration(config.treeHeight, boxWidths, boxCenter);

        const auto buildStart = std::chrono::steady_clock::now();
        TreeClass tree(configuration, sourceParticles, targetParticles, config.blockSize, false);
        const auto buildEnd = std::chrono::steady_clock::now();
        const double buildSec = std::chrono::duration<double>(buildEnd - buildStart).count();

        KernelClass kernel(configuration, RealType(deltaNormalized), RealType(1.0));
        AlgorithmClass algorithm(configuration, kernel);

        // Warmup is excluded from timings.
        for (int iteration = -config.warmupIterations; iteration < 0; ++iteration)
        {
            const auto weights = buildIterationWeights(sources, iteration, config.perturbationAmplitude);
            clearFmmState(tree);
            updateSourceWeights(tree, weights);
            algorithm.execute(tree);
            (void)readTargetRhs(tree, targets.size());
        }

        double directTotalSec = 0.0;
        double fmmTotalSec = 0.0;
        std::vector<double> lastDirect;
        std::vector<double> lastFmm;

        for (int iteration = 0; iteration < config.iterations; ++iteration)
        {
            const auto weights = buildIterationWeights(sources, iteration, config.perturbationAmplitude);

            const auto directStart = std::chrono::steady_clock::now();
            auto direct = computeDirect(sources, targets, weights, deltaPhysical);
            const auto directEnd = std::chrono::steady_clock::now();
            directTotalSec += std::chrono::duration<double>(directEnd - directStart).count();

            const auto fmmStart = std::chrono::steady_clock::now();
            clearFmmState(tree);
            updateSourceWeights(tree, weights);
            algorithm.execute(tree);
            auto fmm = readTargetRhs(tree, targets.size());
            const auto fmmEnd = std::chrono::steady_clock::now();
            fmmTotalSec += std::chrono::duration<double>(fmmEnd - fmmStart).count();

            if (iteration + 1 == config.iterations)
            {
                lastDirect = std::move(direct);
                lastFmm = std::move(fmm);
            }
        }

        const ErrorMetrics metrics = computeErrors(targets, lastDirect, lastFmm);
        const double speedup = directTotalSec / std::max(fmmTotalSec, 1e-30);

        summary << pointsPerWell << ',' << wells.size() << ',' << sources.size() << ',' << targets.size() << ','
                << config.iterations << ',' << config.warmupIterations << ',' << P << ',' << config.treeHeight << ','
                << config.blockSize << ',' << config.lagMonths << ',' << deltaPhysical << ',' << buildSec << ','
                << directTotalSec << ',' << fmmTotalSec << ',' << directTotalSec / config.iterations << ','
                << fmmTotalSec / config.iterations << ',' << speedup << ',' << metrics.relativeL2 << ','
                << metrics.relativeInf << ',' << metrics.maxAbsError << ',' << metrics.meanAbsError << '\n';

        std::cout << "points/well=" << pointsPerWell
                  << " sources=" << sources.size()
                  << " | build=" << buildSec << " s"
                  << " | direct=" << directTotalSec << " s"
                  << " | fmm=" << fmmTotalSec << " s"
                  << " | speedup=" << speedup
                  << " | relL2=" << metrics.relativeL2
                  << " | maxAbs=" << metrics.maxAbsError << "\n";

        const fs::path mapFile = config.outputDir / ("adaptation_last_map_sources_" + std::to_string(sources.size()) + "_p" + std::to_string(P) + "_h" + std::to_string(config.treeHeight) + ".csv");
        saveLastMap(mapFile, targets, lastDirect, lastFmm);
    }

    std::cout << "\nSaved summary: " << summaryFile << "\n";
}

int main(int argc, char** argv)
{
    try
    {
        const Config config = parseArgs(argc, argv);
        const auto wells = readWells(config.inputDir / "real_wells.csv");
        const auto contour = readOuterContour(config.inputDir / "contour.csv");
        const auto targets = buildGrid(contour, config.gridSize);

        const auto insideCount = std::count_if(targets.begin(), targets.end(), [](const Point& p){ return p.inside; });
        std::cout << "Semi-synthetic near-well adaptation benchmark\n";
        std::cout << "inputDir=" << config.inputDir << "\noutputDir=" << config.outputDir << '\n';
        std::cout << "grid=" << config.gridSize << "x" << config.gridSize << " targets=" << targets.size()
                  << " inside=" << insideCount << " wells=" << wells.size() << '\n';
        std::cout << "points-per-well=";
        for (std::size_t i = 0; i < config.pointsPerWellList.size(); ++i)
        {
            if (i) std::cout << ',';
            std::cout << config.pointsPerWellList[i];
        }
        std::cout << "\n\n";

        switch (config.pOrder)
        {
        case 4: runBenchmarkForOrder<4>(config, wells, targets); break;
        case 6: runBenchmarkForOrder<6>(config, wells, targets); break;
        case 8: runBenchmarkForOrder<8>(config, wells, targets); break;
        case 10: runBenchmarkForOrder<10>(config, wells, targets); break;
        case 12: runBenchmarkForOrder<12>(config, wells, targets); break;
        case 16: runBenchmarkForOrder<16>(config, wells, targets); break;
        default: throw std::runtime_error("Supported --order values: 4,6,8,10,12,16");
        }

        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
