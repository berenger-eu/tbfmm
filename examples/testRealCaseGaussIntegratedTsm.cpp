/**
 * Integrated real-case benchmark for ONE requested pressure-map month:
 * Direct vs reusable source-target TSM FMM.
 *
 * This file does not modify TBFMM library headers. Instead of using the
 * library TbfTreeTsm constructor directly, it defines a small local wrapper
 * composed of two ordinary 2D TbfTree instances:
 *   - source tree: real wells only;
 *   - target tree: pressure-map grid points only.
 *
 * The wrapper exposes the interface expected by TbfAlgorithmSelecterTsm.
 * This avoids the dimensionality deduction problem in the stock TbfTreeTsm
 * constructor when particles store {x, y, physical_weight}.
 *
 * The two trees are built once. For each historical temporal contribution:
 *   - well weights are updated;
 *   - source multipoles, target locals and target RHS are cleared;
 *   - a Gaussian kernel with the current delta is created;
 *   - the TSM FMM pass is executed on the existing trees.
 *
 * The executable evaluates exactly one map selected by --month. Historical
 * source contributions are included once for that map; previous maps are not
 * recomputed inside the same run.
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
// ПАРАМЕТРЫ ПО УМОЛЧАНИЮ
// ==========================================================

struct Config
{
    fs::path inputDir = "real_case_integrated_input";
    fs::path outputDir = "real_case_integrated_output";

    int gridSize = 180;

    // Строим и сравниваем только одну карту давления.
    // История работы скважин учитывается от месяца 0 до targetMonth - 1.
    int targetMonth = 47;

    long int treeHeight = 5;
    unsigned int pOrder = 16;
    long int blockSize = 100;
    // Начинаем с одной эффективной точки времени на месяц.
    // Позже можно проверить 2 и 4 отдельно.
    int timeQuadPoints = 1;

    // false — использовать текущий месячный дебит q_curr.
    // true  — использовать изменение дебита q_curr - q_prev.
    bool useDeltaQ = false;

    // Для массового benchmark по месяцам карту обычно сохранять не нужно.
    // Для контрольного запуска можно передать --save-map true.
    bool saveMap = false;

    double p0Atm = 268.0;

    double kMd = 24.0;
    double muCp = 1.69;
    double porosity = 0.19;
    double ctPaInv = 2.236e-9;
    double hM = 10.0;

    double dtDays = 30.4375;
};


// ==========================================================
// ЕДИНИЦЫ
// ==========================================================

constexpr double MD_TO_M2 = 9.869233e-16;
constexpr double CP_TO_PA_S = 1e-3;
constexpr double DAY_TO_SEC = 24.0 * 3600.0;
constexpr double ATM_TO_PA = 101325.0;


// ==========================================================
// СТРУКТУРЫ ДАННЫХ
// ==========================================================

struct Well
{
    std::string name;
    double x = 0.0;
    double y = 0.0;
    double h = 0.0;

    // rates[month_zero] — q_model, м^3/сут
    std::vector<double> rates;
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


struct QuadPoint
{
    double node = 0.0;
    double weight = 0.0;
};


// ==========================================================
// ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
// ==========================================================

std::string trim(const std::string& value)
{
    const auto begin = value.find_first_not_of(" \t\r\n");

    if (begin == std::string::npos)
    {
        return "";
    }

    const auto end = value.find_last_not_of(" \t\r\n");

    return value.substr(begin, end - begin + 1);
}


std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> cells;
    std::stringstream stream(line);
    std::string cell;

    while (std::getline(stream, cell, ','))
    {
        cells.push_back(trim(cell));
    }

    return cells;
}


std::unordered_map<std::string, std::size_t> buildHeaderMap(
    const std::string& headerLine)
{
    const auto headers = splitCsvLine(headerLine);

    std::unordered_map<std::string, std::size_t> result;

    for (std::size_t index = 0; index < headers.size(); ++index)
    {
        result[headers[index]] = index;
    }

    return result;
}


std::size_t requireColumn(
    const std::unordered_map<std::string, std::size_t>& header,
    const std::string& column)
{
    const auto iterator = header.find(column);

    if (iterator == header.end())
    {
        throw std::runtime_error("Missing required CSV column: " + column);
    }

    return iterator->second;
}


bool parseBool(const std::string& value)
{
    return value == "1"
        || value == "true"
        || value == "True"
        || value == "TRUE";
}


Config parseArgs(const int argc, char** argv)
{
    Config config;

    for (int index = 1; index < argc; ++index)
    {
        const std::string arg = argv[index];

        auto requireValue = [&](const std::string& name) -> std::string
        {
            if (index + 1 >= argc)
            {
                throw std::runtime_error("Missing value after " + name);
            }

            return argv[++index];
        };

        if (arg == "--input-dir")
        {
            config.inputDir = requireValue(arg);
        }
        else if (arg == "--tree-height")
        {
            config.treeHeight = std::stol(requireValue(arg));
        }
        else if (arg == "--order")
        {
            config.pOrder = static_cast<unsigned int>(
                std::stoul(requireValue(arg)));
        }
        else if (arg == "--block-size")
        {
            config.blockSize = std::stol(requireValue(arg));
        }
        else if (arg == "--output-dir")
        {
            config.outputDir = requireValue(arg);
        }
        else if (arg == "--grid-size")
        {
            config.gridSize = std::stoi(requireValue(arg));
        }
        else if (arg == "--month")
        {
            config.targetMonth = std::stoi(requireValue(arg));
        }
        else if (arg == "--quad-points")
        {
            config.timeQuadPoints = std::stoi(requireValue(arg));
        }
        else if (arg == "--use-delta-q")
        {
            config.useDeltaQ = parseBool(requireValue(arg));
        }
        else if (arg == "--save-map")
        {
            config.saveMap = parseBool(requireValue(arg));
        }
        else if (arg == "--p0-atm")
        {
            config.p0Atm = std::stod(requireValue(arg));
        }
        else if (arg == "--k-md")
        {
            config.kMd = std::stod(requireValue(arg));
        }
        else if (arg == "--mu-cp")
        {
            config.muCp = std::stod(requireValue(arg));
        }
        else if (arg == "--porosity")
        {
            config.porosity = std::stod(requireValue(arg));
        }
        else if (arg == "--ct-pa-inv")
        {
            config.ctPaInv = std::stod(requireValue(arg));
        }
        else if (arg == "--h-m")
        {
            config.hM = std::stod(requireValue(arg));
        }
        else if (arg == "--dt-days")
        {
            config.dtDays = std::stod(requireValue(arg));
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    return config;
}


// ==========================================================
// ФИЗИКА
// ==========================================================

double computeKappa(const Config& config)
{
    const double kM2 = config.kMd * MD_TO_M2;
    const double muPaS = config.muCp * CP_TO_PA_S;

    return kM2 / (config.porosity * muPaS * config.ctPaInv);
}


double computePrefactor(const Config& config)
{
    const double kM2 = config.kMd * MD_TO_M2;
    const double muPaS = config.muCp * CP_TO_PA_S;

    return muPaS / (4.0 * PI * kM2 * config.hM);
}


std::vector<QuadPoint> getGaussLegendreRule(const int points)
{
    if (points == 1)
    {
        return {
            {0.0, 2.0},
        };
    }

    if (points == 2)
    {
        const double a = 1.0 / std::sqrt(3.0);

        return {
            {-a, 1.0},
            { a, 1.0},
        };
    }

    if (points == 4)
    {
        return {
            {-0.8611363115940526, 0.3478548451374538},
            {-0.3399810435848563, 0.6521451548625461},
            { 0.3399810435848563, 0.6521451548625461},
            { 0.8611363115940526, 0.3478548451374538},
        };
    }

    throw std::runtime_error(
        "Supported --quad-points values: 1, 2, 4.");
}


double getSourceQ(
    const Well& well,
    const int sourceMonthZero,
    const bool useDeltaQ)
{
    const double qCurrent =
        sourceMonthZero < static_cast<int>(well.rates.size())
        ? well.rates[sourceMonthZero]
        : 0.0;

    if (!useDeltaQ)
    {
        return qCurrent;
    }

    const double qPrevious =
        sourceMonthZero > 0
        && sourceMonthZero - 1 < static_cast<int>(well.rates.size())
        ? well.rates[sourceMonthZero - 1]
        : 0.0;

    return qCurrent - qPrevious;
}


// ==========================================================
// ЧТЕНИЕ CSV
// ==========================================================

std::vector<Well> readWells(
    const fs::path& filename,
    const int maxMonth)
{
    std::ifstream input(filename);

    if (!input)
    {
        throw std::runtime_error(
            "Cannot open wells file: " + filename.string());
    }

    std::string line;

    if (!std::getline(input, line))
    {
        throw std::runtime_error("Empty wells file.");
    }

    const auto header = buildHeaderMap(line);

    const auto nameColumn = requireColumn(header, "well_name");
    const auto xColumn = requireColumn(header, "x");
    const auto yColumn = requireColumn(header, "y");

    const auto hIterator = header.find("h");

    std::vector<Well> wells;

    while (std::getline(input, line))
    {
        if (trim(line).empty())
        {
            continue;
        }

        const auto cells = splitCsvLine(line);

        Well well;

        well.name = cells.at(nameColumn);
        well.x = std::stod(cells.at(xColumn));
        well.y = std::stod(cells.at(yColumn));

        if (hIterator != header.end())
        {
            well.h = std::stod(cells.at(hIterator->second));
        }

        well.rates.assign(maxMonth, 0.0);

        wells.push_back(well);
    }

    return wells;
}


void readRates(
    const fs::path& filename,
    std::vector<Well>& wells,
    const int maxMonth)
{
    std::ifstream input(filename);

    if (!input)
    {
        throw std::runtime_error(
            "Cannot open rates file: " + filename.string());
    }

    std::unordered_map<std::string, std::size_t> wellIndex;

    for (std::size_t index = 0; index < wells.size(); ++index)
    {
        wellIndex[wells[index].name] = index;
    }

    std::string line;

    if (!std::getline(input, line))
    {
        throw std::runtime_error("Empty rates file.");
    }

    const auto header = buildHeaderMap(line);

    const auto nameColumn = requireColumn(header, "well_name");
    const auto monthColumn = requireColumn(header, "month");
    const auto qColumn = requireColumn(header, "q_model");

    while (std::getline(input, line))
    {
        if (trim(line).empty())
        {
            continue;
        }

        const auto cells = splitCsvLine(line);

        const std::string wellName = cells.at(nameColumn);
        const int monthZero = std::stoi(cells.at(monthColumn));
        const double qModel = std::stod(cells.at(qColumn));

        const auto iterator = wellIndex.find(wellName);

        if (iterator == wellIndex.end())
        {
            throw std::runtime_error(
                "Rate exists for unknown well: " + wellName);
        }

        if (monthZero >= 0 && monthZero < maxMonth)
        {
            wells[iterator->second].rates[monthZero] = qModel;
        }
    }
}


std::vector<std::array<double, 2>> readOuterContour(
    const fs::path& filename)
{
    std::ifstream input(filename);

    if (!input)
    {
        throw std::runtime_error(
            "Cannot open contour file: " + filename.string());
    }

    std::string line;

    if (!std::getline(input, line))
    {
        throw std::runtime_error("Empty contour file.");
    }

    const auto header = buildHeaderMap(line);

    const auto idColumn = requireColumn(header, "contour_id");
    const auto xColumn = requireColumn(header, "x");
    const auto yColumn = requireColumn(header, "y");

    std::vector<std::array<double, 2>> contour;

    while (std::getline(input, line))
    {
        if (trim(line).empty())
        {
            continue;
        }

        const auto cells = splitCsvLine(line);

        const int contourId = std::stoi(cells.at(idColumn));

        if (contourId != 0)
        {
            continue;
        }

        contour.push_back({
            std::stod(cells.at(xColumn)),
            std::stod(cells.at(yColumn)),
        });
    }

    if (contour.size() < 3)
    {
        throw std::runtime_error(
            "Outer contour must contain at least 3 points.");
    }

    return contour;
}


// ==========================================================
// СЕТКА
// ==========================================================

bool pointInsidePolygon(
    const double x,
    const double y,
    const std::vector<std::array<double, 2>>& polygon)
{
    bool inside = false;

    const std::size_t size = polygon.size();

    for (std::size_t i = 0, j = size - 1; i < size; j = i++)
    {
        const double xi = polygon[i][0];
        const double yi = polygon[i][1];

        const double xj = polygon[j][0];
        const double yj = polygon[j][1];

        const bool intersects =
            ((yi > y) != (yj > y))
            && (
                x
                < (xj - xi) * (y - yi)
                    / ((yj - yi) + std::numeric_limits<double>::epsilon())
                    + xi
            );

        if (intersects)
        {
            inside = !inside;
        }
    }

    return inside;
}


std::vector<Point> buildGrid(
    const std::vector<std::array<double, 2>>& contour,
    const int gridSize)
{
    double minX = contour[0][0];
    double maxX = contour[0][0];
    double minY = contour[0][1];
    double maxY = contour[0][1];

    for (const auto& point : contour)
    {
        minX = std::min(minX, point[0]);
        maxX = std::max(maxX, point[0]);

        minY = std::min(minY, point[1]);
        maxY = std::max(maxY, point[1]);
    }

    std::vector<Point> grid;

    grid.reserve(gridSize * gridSize);

    for (int iy = 0; iy < gridSize; ++iy)
    {
        const double y =
            minY
            + (maxY - minY) * static_cast<double>(iy)
                / static_cast<double>(gridSize - 1);

        for (int ix = 0; ix < gridSize; ++ix)
        {
            const double x =
                minX
                + (maxX - minX) * static_cast<double>(ix)
                    / static_cast<double>(gridSize - 1);

            Point point;

            point.targetId =
                static_cast<long int>(iy) * gridSize + ix;

            point.ix = ix;
            point.iy = iy;

            point.x = x;
            point.y = y;

            point.inside = pointInsidePolygon(x, y, contour);

            grid.push_back(point);
        }
    }

    return grid;
}


// ==========================================================
// DIRECT
// ==========================================================

std::vector<double> computeDirectMonth(
    const Config& config,
    const std::vector<Well>& wells,
    const std::vector<Point>& targets,
    const int targetMonthIndex)
{
    const double kappa = computeKappa(config);
    const double prefactor = computePrefactor(config);
    const double dtSeconds = config.dtDays * DAY_TO_SEC;

    const auto quadrature = getGaussLegendreRule(
        config.timeQuadPoints);

    const int targetMonthZero = targetMonthIndex - 1;

    const double finalTime =
        static_cast<double>(targetMonthIndex) * dtSeconds;

    std::vector<double> pressurePa(targets.size(), 0.0);

    for (int sourceMonthZero = 0;
         sourceMonthZero <= targetMonthZero;
         ++sourceMonthZero)
    {
        const double intervalStart =
            static_cast<double>(sourceMonthZero) * dtSeconds;

        const double intervalEnd =
            static_cast<double>(sourceMonthZero + 1) * dtSeconds;

        for (const auto& quadPoint : quadrature)
        {
            const double tau =
                0.5 * (intervalEnd - intervalStart) * quadPoint.node
                + 0.5 * (intervalEnd + intervalStart);

            const double weightTime =
                0.5 * (intervalEnd - intervalStart) * quadPoint.weight;

            const double lagSeconds =
                std::max(finalTime - tau, 1e-12);

            const double delta =
                4.0 * kappa * lagSeconds;

            for (const auto& well : wells)
            {
                const double qModel =
                    getSourceQ(
                        well,
                        sourceMonthZero,
                        config.useDeltaQ);

                if (std::abs(qModel) < 1e-14)
                {
                    continue;
                }

                const double qM3S =
                    qModel / DAY_TO_SEC;

                const double sourceWeight =
                    qM3S * prefactor * weightTime / lagSeconds;

                for (std::size_t targetIndex = 0;
                     targetIndex < targets.size();
                     ++targetIndex)
                {
                    const double dx =
                        targets[targetIndex].x - well.x;

                    const double dy =
                        targets[targetIndex].y - well.y;

                    const double rSquared =
                        dx * dx + dy * dy;

                    pressurePa[targetIndex] +=
                        sourceWeight
                        * std::exp(-rSquared / delta);
                }
            }
        }
    }

    return pressurePa;
}


// ==========================================================
// СОХРАНЕНИЕ
// ==========================================================

void saveMap(
    const fs::path& filename,
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<double>& pressureDeltaPa,
    const int monthIndex)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write map file: " + filename.string());
    }

    output << std::setprecision(17);

    output
        << "month_index,target_id,ix,iy,x,y,inside,"
        << "delta_pressure_pa,delta_pressure_atm,pressure_atm\n";

    for (std::size_t index = 0; index < targets.size(); ++index)
    {
        const double deltaAtm =
            pressureDeltaPa[index] / ATM_TO_PA;

        output
            << monthIndex << ","
            << targets[index].targetId << ","
            << targets[index].ix << ","
            << targets[index].iy << ","
            << targets[index].x << ","
            << targets[index].y << ","
            << (targets[index].inside ? 1 : 0) << ","
            << pressureDeltaPa[index] << ","
            << deltaAtm << ","
            << config.p0Atm + deltaAtm << "\n";
    }
}


int countCurrentActiveWells(
    const std::vector<Well>& wells,
    const int monthZero)
{
    int result = 0;

    for (const auto& well : wells)
    {
        if (
            monthZero < static_cast<int>(well.rates.size())
            && std::abs(well.rates[monthZero]) > 1e-14
        )
        {
            ++result;
        }
    }

    return result;
}

// ==========================================================
// НОРМИРОВКА КООРДИНАТ ДЛЯ FMM
// ==========================================================

struct Normalization
{
    double centerX = 0.0;
    double centerY = 0.0;

    // Одинаковый размер по X и Y, чтобы работать в квадратном box.
    double boxSize = 1.0;

    double xNorm(const double x) const
    {
        return (x - centerX) / boxSize + 0.5;
    }

    double yNorm(const double y) const
    {
        return (y - centerY) / boxSize + 0.5;
    }
};


Normalization buildNormalization(
    const std::vector<Well>& wells,
    const std::vector<Point>& targets)
{
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();

    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& well : wells)
    {
        minX = std::min(minX, well.x);
        maxX = std::max(maxX, well.x);

        minY = std::min(minY, well.y);
        maxY = std::max(maxY, well.y);
    }

    for (const auto& target : targets)
    {
        minX = std::min(minX, target.x);
        maxX = std::max(maxX, target.x);

        minY = std::min(minY, target.y);
        maxY = std::max(maxY, target.y);
    }

    Normalization normalization;

    normalization.centerX = 0.5 * (minX + maxX);
    normalization.centerY = 0.5 * (minY + maxY);

    // Небольшой padding нужен, чтобы частицы не лежали строго
    // на внешней границе box.
    normalization.boxSize =
        1.01 * std::max(maxX - minX, maxY - minY);

    if (normalization.boxSize <= 0.0)
    {
        throw std::runtime_error(
            "Invalid normalization box size.");
    }

    return normalization;
}


// ==========================================================
// МЕТРИКИ ОШИБКИ
// ==========================================================

struct ErrorMetrics
{
    double relativeL2 = 0.0;
    double relativeInf = 0.0;

    double maxAbsErrorAtm = 0.0;
    double meanAbsErrorAtm = 0.0;
};


ErrorMetrics computeErrorMetrics(
    const std::vector<Point>& targets,
    const std::vector<double>& directPa,
    const std::vector<double>& fmmPa)
{
    if (directPa.size() != fmmPa.size())
    {
        throw std::runtime_error(
            "Direct and FMM vectors have different sizes.");
    }

    double errorL2Squared = 0.0;
    double directL2Squared = 0.0;

    double errorInf = 0.0;
    double directInf = 0.0;

    double absErrorAtmSum = 0.0;
    std::size_t insideCount = 0;

    for (std::size_t index = 0;
         index < targets.size();
         ++index)
    {
        if (!targets[index].inside)
        {
            continue;
        }

        const double diff =
            fmmPa[index] - directPa[index];

        const double absDiff =
            std::abs(diff);

        errorL2Squared += diff * diff;
        directL2Squared += directPa[index] * directPa[index];

        errorInf =
            std::max(errorInf, absDiff);

        directInf =
            std::max(directInf, std::abs(directPa[index]));

        absErrorAtmSum +=
            absDiff / ATM_TO_PA;

        ++insideCount;
    }

    ErrorMetrics metrics;

    metrics.relativeL2 =
        std::sqrt(errorL2Squared)
        / std::max(std::sqrt(directL2Squared), 1e-30);

    metrics.relativeInf =
        errorInf
        / std::max(directInf, 1e-30);

    metrics.maxAbsErrorAtm =
        errorInf / ATM_TO_PA;

    metrics.meanAbsErrorAtm =
        insideCount > 0
        ? absErrorAtmSum / static_cast<double>(insideCount)
        : 0.0;

    return metrics;
}


// ==========================================================
// СОХРАНЕНИЕ DIRECT / FMM / ERROR
// ==========================================================

void saveCompareMap(
    const fs::path& filename,
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<double>& directPa,
    const std::vector<double>& fmmPa,
    const int monthIndex)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write compare map: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "month_index,target_id,ix,iy,x,y,inside,"
        << "direct_delta_pa,fmm_delta_pa,error_pa,"
        << "direct_pressure_atm,fmm_pressure_atm,"
        << "abs_error_atm\n";

    for (std::size_t index = 0;
         index < targets.size();
         ++index)
    {
        const double directAtm =
            config.p0Atm + directPa[index] / ATM_TO_PA;

        const double fmmAtm =
            config.p0Atm + fmmPa[index] / ATM_TO_PA;

        const double errorPa =
            fmmPa[index] - directPa[index];

        output
            << monthIndex << ","
            << targets[index].targetId << ","
            << targets[index].ix << ","
            << targets[index].iy << ","
            << targets[index].x << ","
            << targets[index].y << ","
            << (targets[index].inside ? 1 : 0) << ","
            << directPa[index] << ","
            << fmmPa[index] << ","
            << errorPa << ","
            << directAtm << ","
            << fmmAtm << ","
            << std::abs(errorPa) / ATM_TO_PA
            << "\n";
    }
}


// ==========================================================
// ОЧИСТКА И ОБНОВЛЕНИЕ СОСТОЯНИЯ SOURCE-TARGET ДЕРЕВА
// ==========================================================

template <class TreeClass>
void clearFmmState(TreeClass& tree)
{
    // P2M и M2M накапливают multipole-коэффициенты source-ячеек.
    tree.applyToAllCellsSource(
        [](const long int /*level*/,
           auto&& /*cellHeader*/,
           auto cellMultipole,
           auto&& /*cellLocal*/)
        {
            if (cellMultipole)
            {
                cellMultipole->get().fill(0.0);
            }
        });

    // M2L и L2L накапливают local-коэффициенты target-ячеек.
    tree.applyToAllCellsTarget(
        [](const long int /*level*/,
           auto&& /*cellHeader*/,
           auto&& /*cellMultipole*/,
           auto cellLocal)
        {
            if (cellLocal)
            {
                cellLocal->get().fill(0.0);
            }
        });

    // L2P и P2PTsm накапливают RHS только у target-частиц.
    tree.applyToAllLeavesTarget(
        [](auto&& leafHeader,
           const long int* /*particleIndexes*/,
           auto&& /*particleDataPtr*/,
           auto&& particleRhsPtr)
        {
            for (long int index = 0;
                 index < leafHeader.nbParticles;
                 ++index)
            {
                particleRhsPtr[0][index] = 0.0;
            }
        });
}


template <class TreeClass>
void updateSourceWeights(
    TreeClass& tree,
    const std::vector<double>& sourceWeights)
{
    tree.applyToAllLeavesSource(
        [&sourceWeights](
            auto&& leafHeader,
            const long int* particleIndexes,
            auto&& particleDataPtr,
            auto&& /*particleRhsPtr*/)
        {
            for (long int index = 0;
                 index < leafHeader.nbParticles;
                 ++index)
            {
                const long int originalIndex =
                    particleIndexes[index];

                particleDataPtr[2][index] =
                    sourceWeights.at(
                        static_cast<std::size_t>(
                            originalIndex));
            }
        });
}


template <class TreeClass>
void accumulateTargetRhs(
    TreeClass& tree,
    std::vector<double>& accumulatedValues)
{
    tree.applyToAllLeavesTarget(
        [&accumulatedValues](
            auto&& leafHeader,
            const long int* particleIndexes,
            auto&& /*particleDataPtr*/,
            auto&& particleRhsPtr)
        {
            for (long int index = 0;
                 index < leafHeader.nbParticles;
                 ++index)
            {
                const long int originalIndex =
                    particleIndexes[index];

                accumulatedValues.at(
                    static_cast<std::size_t>(
                        originalIndex))
                    += particleRhsPtr[0][index];
            }
        });
}


// ==========================================================
// ОДИН МЕСЯЦ: ИНТЕГРИРОВАННЫЙ FMM
// ==========================================================

template <
    unsigned int P,
    class TreeClass,
    class SpacialConfiguration>
std::vector<double> computeFmmMonth(
    const Config& config,
    const std::vector<Well>& wells,
    const std::vector<Point>& targets,
    TreeClass& tree,
    const SpacialConfiguration& configuration,
    const Normalization& normalization,
    const int targetMonthIndex)
{
    using RealType = double;

    using SpaceIndexType =
        TbfDefaultSpaceIndexType2D<RealType>;

    using KernelClass =
        TbfGaussKernel<
            RealType,
            P,
            SpaceIndexType>;

    // Force the sequential TSM implementation for the first stable benchmark.
    // TbfAlgorithmSelecterTsm may choose the OpenMP path in builds compiled
    // with TBF_USE_OPENMP. We first validate the reusable source-target flow
    // without parallel scheduling.
    using AlgorithmClass =
        TbfAlgorithmTsm<
            RealType,
            KernelClass,
            SpaceIndexType>;

    const double kappa =
        computeKappa(config);

    const double prefactor =
        computePrefactor(config);

    const double dtSeconds =
        config.dtDays * DAY_TO_SEC;

    const auto quadrature =
        getGaussLegendreRule(
            config.timeQuadPoints);

    const int targetMonthZero =
        targetMonthIndex - 1;

    const double finalTime =
        static_cast<double>(targetMonthIndex)
        * dtSeconds;

    std::vector<double> accumulatedPressurePa(
        targets.size(),
        0.0);

    std::vector<double> sourceWeights(
        wells.size(),
        0.0);

    for (int sourceMonthZero = 0;
         sourceMonthZero <= targetMonthZero;
         ++sourceMonthZero)
    {
        const double intervalStart =
            static_cast<double>(sourceMonthZero)
            * dtSeconds;

        const double intervalEnd =
            static_cast<double>(sourceMonthZero + 1)
            * dtSeconds;

        for (const auto& quadPoint : quadrature)
        {
            const double tau =
                0.5
                    * (intervalEnd - intervalStart)
                    * quadPoint.node
                + 0.5
                    * (intervalEnd + intervalStart);

            const double weightTime =
                0.5
                * (intervalEnd - intervalStart)
                * quadPoint.weight;

            const double lagSeconds =
                std::max(finalTime - tau, 1e-12);

            const double deltaPhysical =
                4.0 * kappa * lagSeconds;

            const double deltaNormalized =
                deltaPhysical
                / (
                    normalization.boxSize
                    * normalization.boxSize
                );

            for (std::size_t wellIndex = 0;
                 wellIndex < wells.size();
                 ++wellIndex)
            {
                const double qModel =
                    getSourceQ(
                        wells[wellIndex],
                        sourceMonthZero,
                        config.useDeltaQ);

                const double qM3S =
                    qModel / DAY_TO_SEC;

                sourceWeights[wellIndex] =
                    qM3S
                    * prefactor
                    * weightTime
                    / lagSeconds;
            }

            std::cout
                << "  lag sourceMonth="
                << sourceMonthZero
                << " | clear ... "
                << std::flush;

            clearFmmState(tree);

            std::cout
                << "update weights ... "
                << std::flush;

            updateSourceWeights(
                tree,
                sourceWeights);

            std::cout
                << "kernel ... "
                << std::flush;

            KernelClass kernel(
                configuration,
                RealType(deltaNormalized),
                RealType(1.0));

            AlgorithmClass algorithm(
                configuration,
                kernel);

            std::cout
                << "execute ... "
                << std::flush;

            algorithm.execute(tree);

            std::cout
                << "accumulate ... "
                << std::flush;

            accumulateTargetRhs(
                tree,
                accumulatedPressurePa);

            std::cout
                << "ok\n";
        }
    }

    return accumulatedPressurePa;
}


// ==========================================================
// ЗАПУСК ДЛЯ КОНКРЕТНОГО P
// ==========================================================

template <unsigned int P>
void runIntegratedBenchmark(
    const Config& config,
    const std::vector<Well>& wells,
    const std::vector<Point>& targets)
{
    using RealType = double;

    constexpr int Dim = 2;

    constexpr long int NbDataValuesPerParticle =
        Dim + 1;

    constexpr long int NbRhsValuesPerParticle =
        1;

    constexpr long int VectorSize =
        (P + 1) * (P + 1);

    using ParticleDataType =
        RealType;

    using ParticleRhsType =
        RealType;

    using MultipoleClass =
        std::array<RealType, VectorSize>;

    using LocalClass =
        std::array<RealType, VectorSize>;

    using SpaceIndexType =
        TbfDefaultSpaceIndexType2D<RealType>;

    using TreeClass =
        LocalTreeTsm<
            RealType,
            ParticleDataType,
            NbDataValuesPerParticle,
            ParticleRhsType,
            NbRhsValuesPerParticle,
            MultipoleClass,
            LocalClass,
            SpaceIndexType>;

    const Normalization normalization =
        buildNormalization(
            wells,
            targets);

    const std::size_t nbWells =
        wells.size();

    const std::size_t nbTargets =
        targets.size();

    std::vector<std::array<RealType, Dim + 1>>
        sourceParticles(
            nbWells);

    for (std::size_t index = 0;
         index < nbWells;
         ++index)
    {
        sourceParticles[index][0] =
            normalization.xNorm(
                wells[index].x);

        sourceParticles[index][1] =
            normalization.yNorm(
                wells[index].y);

        // Обновляется перед каждым временным вкладом.
        sourceParticles[index][2] =
            0.0;
    }

    std::vector<std::array<RealType, Dim + 1>>
        targetParticles(
            nbTargets);

    for (std::size_t index = 0;
         index < nbTargets;
         ++index)
    {
        targetParticles[index][0] =
            normalization.xNorm(
                targets[index].x);

        targetParticles[index][1] =
            normalization.yNorm(
                targets[index].y);

        // Третье значение target-частиц ядром не используется.
        targetParticles[index][2] =
            0.0;
    }

    const std::array<RealType, Dim>
        boxWidths{{1.0, 1.0}};

    const std::array<RealType, Dim>
        boxCenter{{0.5, 0.5}};

    const TbfSpacialConfiguration<
        RealType,
        Dim>
        configuration(
            config.treeHeight,
            boxWidths,
            boxCenter);

    const auto treeBuildStart =
        std::chrono::steady_clock::now();

    // Два дерева строятся ровно один раз:
    // source tree — только скважины;
    // target tree — только точки карты.
    TreeClass tree(
        configuration,
        sourceParticles,
        targetParticles,
        config.blockSize,
        false);

    const auto treeBuildEnd =
        std::chrono::steady_clock::now();

    const double treeBuildTimeSec =
        std::chrono::duration<double>(
            treeBuildEnd
            - treeBuildStart).count();

    std::cout
        << "TSM trees built once in "
        << treeBuildTimeSec
        << " sec\n";

    const int monthIndex =
        config.targetMonth;

    const fs::path timingFile =
        config.outputDir
        / (
            "integrated_single_month_tsm_direct_vs_fmm_month_"
            + std::to_string(monthIndex)
            + "_p"
            + std::to_string(P)
            + "_h"
            + std::to_string(config.treeHeight)
            + ".csv"
        );

    std::ofstream timingOutput(
        timingFile);

    if (!timingOutput)
    {
        throw std::runtime_error(
            "Cannot write timing file: "
            + timingFile.string());
    }

    timingOutput
        << std::setprecision(17);

    timingOutput
        << "month_index,P,tree_height,block_size,active_wells,"
        << "temporal_terms,quad_points,"
        << "tree_build_time_sec,direct_time_sec,"
        << "fmm_execute_time_sec,fmm_with_tree_build_sec,"
        << "speedup_execute,speedup_with_tree_build,"
        << "relative_l2,relative_inf,"
        << "max_abs_error_atm,mean_abs_error_atm\n";

    std::cout
        << "Month "
        << monthIndex
        << " ... "
        << std::flush;

    const auto directStart =
        std::chrono::steady_clock::now();

    const auto directPa =
        computeDirectMonth(
            config,
            wells,
            targets,
            monthIndex);

    const auto directEnd =
        std::chrono::steady_clock::now();

    const double directTimeSec =
        std::chrono::duration<double>(
            directEnd
            - directStart).count();

    const auto fmmStart =
        std::chrono::steady_clock::now();

    const auto fmmPa =
        computeFmmMonth<P>(
            config,
            wells,
            targets,
            tree,
            configuration,
            normalization,
            monthIndex);

    const auto fmmEnd =
        std::chrono::steady_clock::now();

    const double fmmTimeSec =
        std::chrono::duration<double>(
            fmmEnd
            - fmmStart).count();

    const double fmmWithTreeBuildSec =
        treeBuildTimeSec
        + fmmTimeSec;

    const ErrorMetrics metrics =
        computeErrorMetrics(
            targets,
            directPa,
            fmmPa);

    const int activeWells =
        countCurrentActiveWells(
            wells,
            monthIndex - 1);

    const int temporalTerms =
        monthIndex
        * config.timeQuadPoints;

    const double speedupExecute =
        directTimeSec
        / std::max(
            fmmTimeSec,
            1e-30);

    const double speedupWithTreeBuild =
        directTimeSec
        / std::max(
            fmmWithTreeBuildSec,
            1e-30);

    timingOutput
        << monthIndex << ","
        << P << ","
        << config.treeHeight << ","
        << config.blockSize << ","
        << activeWells << ","
        << temporalTerms << ","
        << config.timeQuadPoints << ","
        << treeBuildTimeSec << ","
        << directTimeSec << ","
        << fmmTimeSec << ","
        << fmmWithTreeBuildSec << ","
        << speedupExecute << ","
        << speedupWithTreeBuild << ","
        << metrics.relativeL2 << ","
        << metrics.relativeInf << ","
        << metrics.maxAbsErrorAtm << ","
        << metrics.meanAbsErrorAtm
        << "\n";

    std::cout
        << "Direct = "
        << directTimeSec
        << " sec"
        << " | TSM FMM execute = "
        << fmmTimeSec
        << " sec"
        << " | TSM FMM with build = "
        << fmmWithTreeBuildSec
        << " sec"
        << " | speedup execute = "
        << speedupExecute
        << " | speedup with build = "
        << speedupWithTreeBuild
        << " | relL2 = "
        << metrics.relativeL2
        << " | maxAbsAtm = "
        << metrics.maxAbsErrorAtm
        << "\n";

    if (config.saveMap)
    {
        const fs::path compareMapFile =
            config.outputDir
            / (
                "compare_map_tsm_month_"
                + std::to_string(monthIndex)
                + "_p"
                + std::to_string(P)
                + "_h"
                + std::to_string(
                    config.treeHeight)
                + ".csv"
            );

        saveCompareMap(
            compareMapFile,
            config,
            targets,
            directPa,
            fmmPa,
            monthIndex);

        std::cout
            << "Saved map: "
            << compareMapFile
            << "\n";
    }

    std::cout
        << "Saved timing: "
        << timingFile
        << "\n";
}


// ==========================================================
// MAIN
// ==========================================================

int main(int argc, char** argv)
{
    try
    {
        const Config config =
            parseArgs(
                argc,
                argv);

        fs::create_directories(
            config.outputDir);

        const fs::path wellsFile =
            config.inputDir
            / "real_wells.csv";

        const fs::path ratesFile =
            config.inputDir
            / "well_rates_clean.csv";

        const fs::path contourFile =
            config.inputDir
            / "contour.csv";

        std::cout
            << "Integrated real-case Direct vs source-target TSM FMM benchmark\n";

        std::cout
            << "inputDir = "
            << config.inputDir
            << "\n";

        std::cout
            << "outputDir = "
            << config.outputDir
            << "\n";

        std::cout
            << "gridSize = "
            << config.gridSize
            << "\n";

        std::cout
            << "targetMonth = "
            << config.targetMonth
            << "\n";

        std::cout
            << "timeQuadPoints = "
            << config.timeQuadPoints
            << "\n";

        std::cout
            << "useDeltaQ = "
            << config.useDeltaQ
            << "\n";

        std::cout
            << "treeHeight = "
            << config.treeHeight
            << "\n";

        std::cout
            << "P = "
            << config.pOrder
            << "\n";

        std::cout
            << "blockSize = "
            << config.blockSize
            << "\n";

        const double kappa =
            computeKappa(
                config);

        std::cout
            << "kappa = "
            << kappa
            << " m^2/s\n";

        std::cout
            << "sqrt(delta 1 month) = "
            << std::sqrt(
                4.0
                * kappa
                * config.dtDays
                * DAY_TO_SEC)
            << " m\n\n";

        if (config.targetMonth < 1)
        {
            throw std::runtime_error(
                "--month must be >= 1.");
        }

        auto wells =
            readWells(
                wellsFile,
                config.targetMonth);

        readRates(
            ratesFile,
            wells,
            config.targetMonth);

        const auto contour =
            readOuterContour(
                contourFile);

        const auto targets =
            buildGrid(
                contour,
                config.gridSize);

        const auto insideCount =
            std::count_if(
                targets.begin(),
                targets.end(),
                [](const Point& point)
                {
                    return point.inside;
                });

        std::cout
            << "wells = "
            << wells.size()
            << "\n";

        std::cout
            << "targets = "
            << targets.size()
            << "\n";

        std::cout
            << "inside targets = "
            << insideCount
            << "\n\n";

        switch (config.pOrder)
        {
        case 8:
            runIntegratedBenchmark<8>(
                config,
                wells,
                targets);
            break;

        case 10:
            runIntegratedBenchmark<10>(
                config,
                wells,
                targets);
            break;

        case 12:
            runIntegratedBenchmark<12>(
                config,
                wells,
                targets);
            break;

        case 16:
            runIntegratedBenchmark<16>(
                config,
                wells,
                targets);
            break;

        default:
            throw std::runtime_error(
                "Unsupported P. Allowed values: "
                "8, 10, 12, 16.");
        }

        std::cout
            << "\nDone.\n";
    }
    catch (const std::exception& error)
    {
        std::cerr
            << "ERROR: "
            << error.what()
            << "\n";

        return 1;
    }

    return 0;
}