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
#include <functional>
#include <random>
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

    double dtDays = 30.4375;
    // ======================================================
    // Параметры прототипа с фиктивными источниками Дирихле
    // ======================================================

    // Сколько точек коллокации ставить на каждый отрезок контура.
    // Для начала 1: одна точка в середине каждого сегмента.
    int boundarySubdivisionsPerSegment = 1;

    // Если > 0, используем фиксированный отступ фиктивных источников в метрах.
    // Если <= 0, используем fictitiousOffsetFraction от размера bounding box.
    double fictitiousOffsetM = -1.0;

    // Отступ фиктивных источников как доля размера области.
    double fictitiousOffsetFraction = 0.03;

    // Валидационный режим: построить boundary/fictitious и завершиться.
    bool stopAfterBoundary = false;
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
    // Валидационный режим: построить boundary/fictitious,
    // выполнить один direct matvec и завершиться.
    bool stopAfterBoundaryMatvec = false;

    // Масштаб случайных тестовых фиктивных дебитов, м^3/сут.
    double matvecTestRateScale = 100.0;
    // Давление Дирихле на границе, атм.
    // Если оставить NaN, будет использовано p0Atm.
    double dirichletBoundaryAtm =
        std::numeric_limits<double>::quiet_NaN();

    // Валидационный режим:
    // построить boundary/fictitious, посчитать вклад реальных скважин
    // на границе, собрать RHS и завершиться.
    bool stopAfterBoundaryRhs = false;

    // ======================================================
    // Параметры direct-GMRES валидации
    // ======================================================

    // Валидационный режим: решить уменьшенную систему A q = b
    // через direct matvec + GMRES и завершиться.
    bool stopAfterDirectGmres = false;

    // Для direct-GMRES берём каждую N-ю граничную точку.
    // При 4504 точках stride=20 даёт примерно 225 неизвестных.
    int gmresBoundaryStride = 20;

    // Максимальное число итераций GMRES.
    int gmresMaxIterations = 80;

    // Относительная точность по невязке ||b - A q|| / ||b||.
    double gmresTolerance = 1e-6;

    // Печатать невязку на каждой итерации.
    bool gmresVerbose = true;
    bool stopAfterFmmMatvec = false;
    bool stopAfterFmmGmres = false;
    //bool stopAfterFictitiousHistoryMapDirect = false;
    bool stopAfterFictitiousHistoryAllMapsDirect = false;
    // Валидационный режим:
    // проверить, что обобщённый вклад фиктивного временного слоя
    // совпадает со старым matvec для случая finalMonth == sourceMonth.
    bool stopAfterFictitiousIntervalValidation = false;
    // Seed для воспроизводимого тестового вектора.
    // ======================================================
    // Параметры истории фиктивных источников
    // ======================================================

    // Валидационный режим:
    // последовательно решить историю фиктивных источников
    // через direct matvec + GMRES.
    bool stopAfterFictitiousHistoryDirect = false;

    // До какого месяца считать историю.
    // Если <= 0, используется targetMonth.
    int historyEndMonth = -1;
    unsigned int matvecTestSeed = 12345;
    double p0Atm = 268.0;

    double kMd = 24.0;
    double muCp = 1.69;
    double porosity = 0.19;
    double ctPaInv = 2.236e-9;
    double hM = 10.0;

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

struct BoundaryPoint
{
    long int boundaryId = 0;
    long int segmentId = 0;

    double x = 0.0;
    double y = 0.0;

    // Внешняя нормаль к границе в этой точке.
    // Для Дирихле пока не используется, но понадобится для Неймана.
    double nx = 0.0;
    double ny = 0.0;

    double segmentLength = 0.0;

    // 0 — Dirichlet. Позже можно добавить 1 — Neumann.
    int boundaryType = 0;
};


struct FictitiousSource
{
    long int sourceId = 0;
    long int boundaryId = 0;

    double x = 0.0;
    double y = 0.0;

    double offsetM = 0.0;

    // Вес пока неизвестен. Его потом найдём из СЛАУ.
    double weight = 0.0;
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
        else if (arg == "--boundary-subdivisions")
        {
            config.boundarySubdivisionsPerSegment =
                std::stoi(requireValue(arg));
        }
        else if (arg == "--stop-after-direct-gmres")
        {
            config.stopAfterDirectGmres =
                parseBool(requireValue(arg));
        }
        else if (arg == "--stop-after-fictitious-history-direct")
        {
            config.stopAfterFictitiousHistoryDirect =
                parseBool(requireValue(arg));
        }
        else if (arg == "--stop-after-fictitious-history-all-maps-direct")
        {
            config.stopAfterFictitiousHistoryAllMapsDirect =
                parseBool(requireValue(arg));
        }
        else if (arg == "--history-end-month")
        {
            config.historyEndMonth =
                std::stoi(requireValue(arg));
        }
        else if (arg == "--stop-after-fmm-matvec")
        {
            config.stopAfterFmmMatvec =
                parseBool(requireValue(arg));
        }
        else if (arg == "--stop-after-fmm-gmres")
        {
            config.stopAfterFmmGmres =
                parseBool(requireValue(arg));
        }
        else if (arg == "--stop-after-fictitious-interval-validation")
        {
            config.stopAfterFictitiousIntervalValidation =
                parseBool(requireValue(arg));
        }
        else if (arg == "--gmres-boundary-stride")
        {
            config.gmresBoundaryStride =
                std::stoi(requireValue(arg));
        }
        else if (arg == "--gmres-max-iters")
        {
            config.gmresMaxIterations =
                std::stoi(requireValue(arg));
        }
        else if (arg == "--gmres-tol")
        {
            config.gmresTolerance =
                std::stod(requireValue(arg));
        }
        else if (arg == "--gmres-verbose")
        {
            config.gmresVerbose =
                parseBool(requireValue(arg));
        }
        else if (arg == "--stop-after-boundary-matvec")
        {
            config.stopAfterBoundaryMatvec =
                parseBool(requireValue(arg));
        }
        else if (arg == "--dirichlet-boundary-atm")
        {
            config.dirichletBoundaryAtm =
                std::stod(requireValue(arg));
        }
        else if (arg == "--stop-after-boundary-rhs")
        {
            config.stopAfterBoundaryRhs =
                parseBool(requireValue(arg));
        }
        else if (arg == "--matvec-test-rate-scale")
        {
            config.matvecTestRateScale =
                std::stod(requireValue(arg));
        }
        else if (arg == "--matvec-test-seed")
        {
            config.matvecTestSeed =
                static_cast<unsigned int>(
                    std::stoul(requireValue(arg)));
        }
        else if (arg == "--fictitious-offset-m")
        {
            config.fictitiousOffsetM =
                std::stod(requireValue(arg));
        }
        else if (arg == "--fictitious-offset-frac")
        {
            config.fictitiousOffsetFraction =
                std::stod(requireValue(arg));
        }
        else if (arg == "--stop-after-boundary")
        {
            config.stopAfterBoundary =
                parseBool(requireValue(arg));
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

double computePolygonSignedArea(
    const std::vector<std::array<double, 2>>& polygon)
{
    double area = 0.0;

    const std::size_t size = polygon.size();

    for (std::size_t i = 0; i < size; ++i)
    {
        const std::size_t j = (i + 1) % size;

        area +=
            polygon[i][0] * polygon[j][1]
            - polygon[j][0] * polygon[i][1];
    }

    return 0.5 * area;
}


double computeContourBoxSize(
    const std::vector<std::array<double, 2>>& contour)
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

    return std::max(maxX - minX, maxY - minY);
}


std::vector<BoundaryPoint> buildBoundaryCollocationPoints(
    const std::vector<std::array<double, 2>>& contour,
    const int subdivisionsPerSegment)
{
    if (contour.size() < 3)
    {
        throw std::runtime_error(
            "Cannot build boundary points: contour has less than 3 points.");
    }

    const int subdivisions =
        std::max(1, subdivisionsPerSegment);

    const double signedArea =
        computePolygonSignedArea(contour);

    const bool isCounterClockwise =
        signedArea > 0.0;

    const double boxSize =
        computeContourBoxSize(contour);

    const double probeDistance =
        1e-6 * boxSize + 1e-9;

    std::vector<BoundaryPoint> boundaryPoints;

    long int boundaryId = 0;

    for (std::size_t segmentIndex = 0;
         segmentIndex < contour.size();
         ++segmentIndex)
    {
        const std::size_t nextIndex =
            (segmentIndex + 1) % contour.size();

        const double x0 = contour[segmentIndex][0];
        const double y0 = contour[segmentIndex][1];

        const double x1 = contour[nextIndex][0];
        const double y1 = contour[nextIndex][1];

        const double dx = x1 - x0;
        const double dy = y1 - y0;

        const double length =
            std::sqrt(dx * dx + dy * dy);

        // Если контур замкнут дублированием первой точки в конце,
        // последний сегмент может иметь нулевую длину.
        if (length <= 1e-12)
        {
            continue;
        }

        double nx = 0.0;
        double ny = 0.0;

        if (isCounterClockwise)
        {
            // Для CCW-обхода внутренняя область слева от ребра,
            // внешняя нормаль направлена вправо.
            nx = dy / length;
            ny = -dx / length;
        }
        else
        {
            // Для CW-обхода наоборот.
            nx = -dy / length;
            ny = dx / length;
        }

        for (int localIndex = 0;
             localIndex < subdivisions;
             ++localIndex)
        {
            const double t =
                (static_cast<double>(localIndex) + 0.5)
                / static_cast<double>(subdivisions);

            BoundaryPoint point;

            point.boundaryId = boundaryId++;
            point.segmentId =
                static_cast<long int>(segmentIndex);

            point.x = x0 + t * dx;
            point.y = y0 + t * dy;

            point.nx = nx;
            point.ny = ny;

            point.segmentLength =
                length / static_cast<double>(subdivisions);

            point.boundaryType = 0;

            // Дополнительная проверка направления нормали.
            // Если небольшой шаг по нормали попал внутрь области,
            // значит нормаль направлена не наружу, переворачиваем.
            const double probeX =
                point.x + point.nx * probeDistance;

            const double probeY =
                point.y + point.ny * probeDistance;

            if (pointInsidePolygon(probeX, probeY, contour))
            {
                point.nx *= -1.0;
                point.ny *= -1.0;
            }

            boundaryPoints.push_back(point);
        }
    }

    if (boundaryPoints.empty())
    {
        throw std::runtime_error(
            "No boundary collocation points were generated.");
    }

    return boundaryPoints;
}


std::vector<FictitiousSource> buildFictitiousSources(
    const Config& config,
    const std::vector<std::array<double, 2>>& contour,
    const std::vector<BoundaryPoint>& boundaryPoints)
{
    const double boxSize =
        computeContourBoxSize(contour);

    const double baseOffset =
        config.fictitiousOffsetM > 0.0
        ? config.fictitiousOffsetM
        : std::max(
            config.fictitiousOffsetFraction * boxSize,
            1e-9);

    std::vector<FictitiousSource> sources;

    sources.reserve(boundaryPoints.size());

    for (const auto& boundaryPoint : boundaryPoints)
    {
        double offset = baseOffset;

        double sourceX =
            boundaryPoint.x + offset * boundaryPoint.nx;

        double sourceY =
            boundaryPoint.y + offset * boundaryPoint.ny;

        // На сложных невыпуклых контурах маленький сдвиг наружу
        // иногда может всё ещё попадать внутрь полигона.
        // Поэтому увеличиваем отступ, пока точка не станет внешней.
        int attempts = 0;

        while (
            pointInsidePolygon(sourceX, sourceY, contour)
            && attempts < 12
        )
        {
            offset *= 1.5;

            sourceX =
                boundaryPoint.x + offset * boundaryPoint.nx;

            sourceY =
                boundaryPoint.y + offset * boundaryPoint.ny;

            ++attempts;
        }

        FictitiousSource source;

        source.sourceId =
            static_cast<long int>(sources.size());

        source.boundaryId =
            boundaryPoint.boundaryId;

        source.x = sourceX;
        source.y = sourceY;

        source.offsetM = offset;
        source.weight = 0.0;

        sources.push_back(source);
    }

    return sources;
}


void saveBoundaryPoints(
    const fs::path& filename,
    const std::vector<BoundaryPoint>& boundaryPoints)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write boundary points file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "boundary_id,segment_id,x,y,nx,ny,"
        << "segment_length,boundary_type\n";

    for (const auto& point : boundaryPoints)
    {
        output
            << point.boundaryId << ","
            << point.segmentId << ","
            << point.x << ","
            << point.y << ","
            << point.nx << ","
            << point.ny << ","
            << point.segmentLength << ","
            << point.boundaryType << "\n";
    }
}


void saveFictitiousSources(
    const fs::path& filename,
    const std::vector<FictitiousSource>& sources)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write fictitious sources file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "source_id,boundary_id,x,y,offset_m,weight\n";

    for (const auto& source : sources)
    {
        output
            << source.sourceId << ","
            << source.boundaryId << ","
            << source.x << ","
            << source.y << ","
            << source.offsetM << ","
            << source.weight << "\n";
    }
}


void validateBoundarySetup(
    const std::vector<std::array<double, 2>>& contour,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    if (boundaryPoints.size() != fictitiousSources.size())
    {
        throw std::runtime_error(
            "Boundary points and fictitious sources count mismatch.");
    }

    std::size_t fictitiousInsideCount = 0;

    double minOffset = std::numeric_limits<double>::max();
    double maxOffset = 0.0;
    double sumOffset = 0.0;

    for (std::size_t index = 0;
         index < fictitiousSources.size();
         ++index)
    {
        const auto& boundaryPoint =
            boundaryPoints[index];

        const auto& source =
            fictitiousSources[index];

        if (pointInsidePolygon(source.x, source.y, contour))
        {
            ++fictitiousInsideCount;
        }

        const double dx =
            source.x - boundaryPoint.x;

        const double dy =
            source.y - boundaryPoint.y;

        const double offset =
            std::sqrt(dx * dx + dy * dy);

        minOffset = std::min(minOffset, offset);
        maxOffset = std::max(maxOffset, offset);
        sumOffset += offset;
    }

    const double meanOffset =
        fictitiousSources.empty()
        ? 0.0
        : sumOffset / static_cast<double>(fictitiousSources.size());

    std::cout
        << "Dirichlet boundary prototype geometry:\n"
        << "  boundary collocation points = "
        << boundaryPoints.size()
        << "\n"
        << "  fictitious sources = "
        << fictitiousSources.size()
        << "\n"
        << "  fictitious sources inside contour = "
        << fictitiousInsideCount
        << "\n"
        << "  fictitious offset min/mean/max = "
        << minOffset
        << " / "
        << meanOffset
        << " / "
        << maxOffset
        << " m\n\n";

    if (fictitiousInsideCount > 0)
    {
        std::cout
            << "WARNING: some fictitious sources are still inside contour. "
            << "Increase --fictitious-offset-m or --fictitious-offset-frac.\n\n";
    }
}

std::vector<double> buildRandomFictitiousRates(
    const std::size_t size,
    const double rateScale,
    const unsigned int seed)
{
    std::mt19937 generator(seed);

    std::uniform_real_distribution<double> distribution(
        -rateScale,
        rateScale);

    std::vector<double> rates(size, 0.0);

    for (auto& value : rates)
    {
        value = distribution(generator);
    }

    return rates;
}


double computeL2Norm(
    const std::vector<double>& values)
{
    double sum = 0.0;

    for (const double value : values)
    {
        sum += value * value;
    }

    return std::sqrt(sum);
}


double computeMaxAbs(
    const std::vector<double>& values)
{
    double maxValue = 0.0;

    for (const double value : values)
    {
        maxValue =
            std::max(
                maxValue,
                std::abs(value));
    }

    return maxValue;
}


std::vector<double> addVectors(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(
            "addVectors: size mismatch.");
    }

    std::vector<double> result(a.size(), 0.0);

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        result[index] = a[index] + b[index];
    }

    return result;
}


std::vector<double> subtractVectors(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(
            "subtractVectors: size mismatch.");
    }

    std::vector<double> result(a.size(), 0.0);

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        result[index] = a[index] - b[index];
    }

    return result;
}


/**
 * Direct matrix-vector multiplication for the Dirichlet fictitious source system.
 *
 * Input:
 *   qFictitiousM3Day[j] — trial fictitious source rate, m^3/day.
 *
 * Output:
 *   boundaryPressurePa[i] — pressure contribution at boundary collocation point i, Pa.
 *
 * The operator uses the same integrated Gaussian heat/piezoconductivity kernel
 * as the real-case direct calculation for one current monthly interval.
 */
std::vector<double> applyDirichletFictitiousMatvecDirect(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& qFictitiousM3Day)
{
    if (fictitiousSources.size() != qFictitiousM3Day.size())
    {
        throw std::runtime_error(
            "applyDirichletFictitiousMatvecDirect: "
            "sources and vector size mismatch.");
    }

    const double kappa =
        computeKappa(config);

    const double prefactor =
        computePrefactor(config);

    const double dtSeconds =
        config.dtDays * DAY_TO_SEC;

    const auto quadrature =
        getGaussLegendreRule(
            config.timeQuadPoints);

    // Для матрицы текущего месяца рассматриваем один свежий интервал [0, dt].
    const double finalTime =
        dtSeconds;

    const double intervalStart =
        0.0;

    const double intervalEnd =
        dtSeconds;

    std::vector<double> boundaryPressurePa(
        boundaryPoints.size(),
        0.0);

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
            std::max(
                finalTime - tau,
                1e-12);

        const double delta =
            4.0 * kappa * lagSeconds;

        // qFictitiousM3Day задан в м^3/сут.
        // Делим на DAY_TO_SEC, чтобы получить м^3/с.
        const double coefficientForRate =
            prefactor
            * weightTime
            / lagSeconds
            / DAY_TO_SEC;

        for (std::size_t targetIndex = 0;
             targetIndex < boundaryPoints.size();
             ++targetIndex)
        {
            const auto& target =
                boundaryPoints[targetIndex];

            double sum = 0.0;

            for (std::size_t sourceIndex = 0;
                 sourceIndex < fictitiousSources.size();
                 ++sourceIndex)
            {
                const auto& source =
                    fictitiousSources[sourceIndex];

                const double dx =
                    target.x - source.x;

                const double dy =
                    target.y - source.y;

                const double rSquared =
                    dx * dx + dy * dy;

                sum +=
                    qFictitiousM3Day[sourceIndex]
                    * coefficientForRate
                    * std::exp(-rSquared / delta);
            }

            boundaryPressurePa[targetIndex] += sum;
        }
    }

    return boundaryPressurePa;
}

/**
 * Direct contribution of one fictitious source time layer.
 *
 * sourceMonth:
 *   month where fictitious sources were active.
 *
 * finalMonth:
 *   month where pressure is evaluated.
 *
 * Example:
 *   sourceMonth = 5, finalMonth = 10
 *   means: fictitious sources worked during month 5,
 *   and we evaluate their remaining contribution at the end of month 10.
 *
 * Input qFictitiousM3Day is interpreted as rates in m^3/day
 * active over interval:
 *
 *   [(sourceMonth - 1) * dt, sourceMonth * dt]
 *
 * Output:
 *   pressure contribution at boundary points at time finalMonth * dt, Pa.
 */
std::vector<double> applyDirichletFictitiousIntervalDirect(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& qFictitiousM3Day,
    const int finalMonth,
    const int sourceMonth)
{
    if (fictitiousSources.size() != qFictitiousM3Day.size())
    {
        throw std::runtime_error(
            "applyDirichletFictitiousIntervalDirect: "
            "sources and vector size mismatch.");
    }

    if (finalMonth < sourceMonth)
    {
        throw std::runtime_error(
            "applyDirichletFictitiousIntervalDirect: "
            "finalMonth must be >= sourceMonth.");
    }

    if (sourceMonth <= 0 || finalMonth <= 0)
    {
        throw std::runtime_error(
            "applyDirichletFictitiousIntervalDirect: "
            "months must be positive.");
    }

    const double kappa =
        computeKappa(config);

    const double prefactor =
        computePrefactor(config);

    const double dtSeconds =
        config.dtDays * DAY_TO_SEC;

    const auto quadrature =
        getGaussLegendreRule(
            config.timeQuadPoints);

    const double finalTime =
        static_cast<double>(finalMonth) * dtSeconds;

    const double intervalStart =
        static_cast<double>(sourceMonth - 1) * dtSeconds;

    const double intervalEnd =
        static_cast<double>(sourceMonth) * dtSeconds;

    std::vector<double> boundaryPressurePa(
        boundaryPoints.size(),
        0.0);

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
            std::max(
                finalTime - tau,
                1e-12);

        const double delta =
            4.0 * kappa * lagSeconds;

        // qFictitiousM3Day задан в м^3/сут.
        // Делим на DAY_TO_SEC, чтобы получить м^3/с.
        const double coefficientForRate =
            prefactor
            * weightTime
            / lagSeconds
            / DAY_TO_SEC;

        for (std::size_t targetIndex = 0;
             targetIndex < boundaryPoints.size();
             ++targetIndex)
        {
            const auto& target =
                boundaryPoints[targetIndex];

            double sum = 0.0;

            for (std::size_t sourceIndex = 0;
                 sourceIndex < fictitiousSources.size();
                 ++sourceIndex)
            {
                const auto& source =
                    fictitiousSources[sourceIndex];

                const double dx =
                    target.x - source.x;

                const double dy =
                    target.y - source.y;

                const double rSquared =
                    dx * dx + dy * dy;

                sum +=
                    qFictitiousM3Day[sourceIndex]
                    * coefficientForRate
                    * std::exp(-rSquared / delta);
            }

            boundaryPressurePa[targetIndex] += sum;
        }
    }

    return boundaryPressurePa;
}

void validateFictitiousIntervalDirectConsistency(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    const auto qTest =
        buildRandomFictitiousRates(
            fictitiousSources.size(),
            config.matvecTestRateScale,
            config.matvecTestSeed);

    // Старый оператор: свежий слой на одном месяце.
    const auto oldMatvecPa =
        applyDirichletFictitiousMatvecDirect(
            config,
            boundaryPoints,
            fictitiousSources,
            qTest);

    // Новый оператор в эквивалентном режиме:
    // sourceMonth == finalMonth.
    const auto intervalSameMonthPa =
        applyDirichletFictitiousIntervalDirect(
            config,
            boundaryPoints,
            fictitiousSources,
            qTest,
            config.targetMonth,
            config.targetMonth);

    const auto diffPa =
        subtractVectors(
            intervalSameMonthPa,
            oldMatvecPa);

    const double relativeL2 =
        computeL2Norm(diffPa)
        / std::max(
            computeL2Norm(oldMatvecPa),
            1e-300);

    std::cout
        << "Fictitious interval direct validation:\n"
        << "  sourceMonth = "
        << config.targetMonth
        << "\n"
        << "  finalMonth = "
        << config.targetMonth
        << "\n"
        << "  relative L2(new interval - old matvec) = "
        << relativeL2
        << "\n"
        << "  max|difference| = "
        << computeMaxAbs(diffPa)
        << " Pa = "
        << computeMaxAbs(diffPa) / ATM_TO_PA
        << " atm\n";

    if (config.targetMonth > 1)
    {
        const auto previousMonthContributionPa =
            applyDirichletFictitiousIntervalDirect(
                config,
                boundaryPoints,
                fictitiousSources,
                qTest,
                config.targetMonth,
                config.targetMonth - 1);

        std::cout
            << "  ||contribution from previous month||_2 = "
            << computeL2Norm(previousMonthContributionPa)
            << " Pa\n"
            << "  max|previous month contribution| = "
            << computeMaxAbs(previousMonthContributionPa)
            << " Pa = "
            << computeMaxAbs(previousMonthContributionPa) / ATM_TO_PA
            << " atm\n";
    }

    std::cout << "\n";
}

struct BoundaryNormalization
{
    double centerX = 0.0;
    double centerY = 0.0;
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


BoundaryNormalization buildBoundaryNormalization(
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();

    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& point : boundaryPoints)
    {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);

        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    for (const auto& source : fictitiousSources)
    {
        minX = std::min(minX, source.x);
        maxX = std::max(maxX, source.x);

        minY = std::min(minY, source.y);
        maxY = std::max(maxY, source.y);
    }

    BoundaryNormalization normalization;

    normalization.centerX = 0.5 * (minX + maxX);
    normalization.centerY = 0.5 * (minY + maxY);

    normalization.boxSize =
        1.01 * std::max(maxX - minX, maxY - minY);

    if (normalization.boxSize <= 0.0)
    {
        throw std::runtime_error(
            "Invalid boundary normalization box size.");
    }

    return normalization;
}

template <unsigned int P>
std::vector<double> applyDirichletFictitiousMatvecFmm(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& qFictitiousM3Day)
{
    if (fictitiousSources.size() != qFictitiousM3Day.size())
    {
        throw std::runtime_error(
            "applyDirichletFictitiousMatvecFmm: "
            "sources and vector size mismatch.");
    }

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

    using KernelClass =
        TbfGaussKernel<
            RealType,
            P,
            SpaceIndexType>;

    using AlgorithmClass =
        TbfAlgorithmTsm<
            RealType,
            KernelClass,
            SpaceIndexType>;

    const BoundaryNormalization normalization =
        buildBoundaryNormalization(
            boundaryPoints,
            fictitiousSources);

    std::vector<std::array<RealType, Dim + 1>>
        sourceParticles(
            fictitiousSources.size());

    for (std::size_t index = 0;
         index < fictitiousSources.size();
         ++index)
    {
        sourceParticles[index][0] =
            normalization.xNorm(
                fictitiousSources[index].x);

        sourceParticles[index][1] =
            normalization.yNorm(
                fictitiousSources[index].y);

        sourceParticles[index][2] =
            0.0;
    }

    std::vector<std::array<RealType, Dim + 1>>
        targetParticles(
            boundaryPoints.size());

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         ++index)
    {
        targetParticles[index][0] =
            normalization.xNorm(
                boundaryPoints[index].x);

        targetParticles[index][1] =
            normalization.yNorm(
                boundaryPoints[index].y);

        targetParticles[index][2] =
            0.0;
    }

    const std::array<RealType, Dim> boxWidths{{1.0, 1.0}};
    const std::array<RealType, Dim> boxCenter{{0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim>
        configuration(
            config.treeHeight,
            boxWidths,
            boxCenter);

    TreeClass tree(
        configuration,
        sourceParticles,
        targetParticles,
        config.blockSize,
        false);

    const double kappa =
        computeKappa(config);

    const double prefactor =
        computePrefactor(config);

    const double dtSeconds =
        config.dtDays * DAY_TO_SEC;

    const auto quadrature =
        getGaussLegendreRule(
            config.timeQuadPoints);

    const double finalTime =
        dtSeconds;

    const double intervalStart =
        0.0;

    const double intervalEnd =
        dtSeconds;

    std::vector<double> boundaryPressurePa(
        boundaryPoints.size(),
        0.0);

    std::vector<double> sourceWeights(
        fictitiousSources.size(),
        0.0);

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
            std::max(
                finalTime - tau,
                1e-12);

        const double deltaPhysical =
            4.0 * kappa * lagSeconds;

        const double deltaNormalized =
            deltaPhysical
            / (
                normalization.boxSize
                * normalization.boxSize
            );

        const double coefficientForRate =
            prefactor
            * weightTime
            / lagSeconds
            / DAY_TO_SEC;

        for (std::size_t sourceIndex = 0;
             sourceIndex < fictitiousSources.size();
             ++sourceIndex)
        {
            sourceWeights[sourceIndex] =
                qFictitiousM3Day[sourceIndex]
                * coefficientForRate;
        }

        clearFmmState(tree);

        updateSourceWeights(
            tree,
            sourceWeights);

        KernelClass kernel(
            configuration,
            RealType(deltaNormalized),
            RealType(1.0));

        AlgorithmClass algorithm(
            configuration,
            kernel);

        algorithm.execute(tree);

        accumulateTargetRhs(
            tree,
            boundaryPressurePa);
    }

    return boundaryPressurePa;
}

void saveBoundaryMatvecResult(
    const fs::path& filename,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<double>& boundaryPressurePa)
{
    if (boundaryPoints.size() != boundaryPressurePa.size())
    {
        throw std::runtime_error(
            "saveBoundaryMatvecResult: size mismatch.");
    }

    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write boundary matvec file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "boundary_id,segment_id,x,y,nx,ny,"
        << "aq_pa,aq_atm\n";

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         ++index)
    {
        const auto& point =
            boundaryPoints[index];

        const double pressurePa =
            boundaryPressurePa[index];

        output
            << point.boundaryId << ","
            << point.segmentId << ","
            << point.x << ","
            << point.y << ","
            << point.nx << ","
            << point.ny << ","
            << pressurePa << ","
            << pressurePa / ATM_TO_PA << "\n";
    }
}

void saveBoundaryMatvecComparison(
    const fs::path& filename,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<double>& directPa,
    const std::vector<double>& fmmPa)
{
    if (boundaryPoints.size() != directPa.size()
        || boundaryPoints.size() != fmmPa.size())
    {
        throw std::runtime_error(
            "saveBoundaryMatvecComparison: size mismatch.");
    }

    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write boundary matvec comparison file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "boundary_id,segment_id,x,y,nx,ny,"
        << "direct_pa,fmm_pa,error_pa,"
        << "direct_atm,fmm_atm,abs_error_atm\n";

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         ++index)
    {
        const double errorPa =
            fmmPa[index] - directPa[index];

        output
            << boundaryPoints[index].boundaryId << ","
            << boundaryPoints[index].segmentId << ","
            << boundaryPoints[index].x << ","
            << boundaryPoints[index].y << ","
            << boundaryPoints[index].nx << ","
            << boundaryPoints[index].ny << ","
            << directPa[index] << ","
            << fmmPa[index] << ","
            << errorPa << ","
            << directPa[index] / ATM_TO_PA << ","
            << fmmPa[index] / ATM_TO_PA << ","
            << std::abs(errorPa) / ATM_TO_PA << "\n";
    }
}

template <unsigned int P>
void validateDirichletFmmMatvec(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    const auto qTest =
        buildRandomFictitiousRates(
            fictitiousSources.size(),
            config.matvecTestRateScale,
            config.matvecTestSeed);

    const auto directStart =
        std::chrono::high_resolution_clock::now();

    const auto directPa =
        applyDirichletFictitiousMatvecDirect(
            config,
            boundaryPoints,
            fictitiousSources,
            qTest);

    const auto directEnd =
        std::chrono::high_resolution_clock::now();

    const auto fmmStart =
        std::chrono::high_resolution_clock::now();

    const auto fmmPa =
        applyDirichletFictitiousMatvecFmm<P>(
            config,
            boundaryPoints,
            fictitiousSources,
            qTest);

    const auto fmmEnd =
        std::chrono::high_resolution_clock::now();

    const auto diffPa =
        subtractVectors(
            fmmPa,
            directPa);

    const double directL2 =
        computeL2Norm(directPa);

    const double errorL2 =
        computeL2Norm(diffPa);

    const double relativeL2 =
        errorL2
        / std::max(directL2, 1e-300);

    const double maxAbsErrorPa =
        computeMaxAbs(diffPa);

    const double directTimeSec =
        std::chrono::duration<double>(
            directEnd - directStart).count();

    const double fmmTimeSec =
        std::chrono::duration<double>(
            fmmEnd - fmmStart).count();

    std::cout
        << "Dirichlet FMM matvec validation:\n"
        << "  P = "
        << P
        << "\n"
        << "  unknowns = "
        << fictitiousSources.size()
        << "\n"
        << "  equations = "
        << boundaryPoints.size()
        << "\n"
        << "  direct time = "
        << directTimeSec
        << " sec\n"
        << "  fmm time = "
        << fmmTimeSec
        << " sec\n"
        << "  relative L2 = "
        << relativeL2
        << "\n"
        << "  max|FMM - direct| = "
        << maxAbsErrorPa
        << " Pa = "
        << maxAbsErrorPa / ATM_TO_PA
        << " atm\n\n";

    const fs::path comparisonFile =
        config.outputDir
        / "dirichlet_boundary_matvec_fmm_vs_direct.csv";

    saveBoundaryMatvecComparison(
        comparisonFile,
        boundaryPoints,
        directPa,
        fmmPa);

    std::cout
        << "Saved FMM/direct matvec comparison: "
        << comparisonFile
        << "\n";
}

void validateDirichletFmmMatvecByOrder(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    switch (config.pOrder)
    {
    case 8:
        validateDirichletFmmMatvec<8>(
            config,
            boundaryPoints,
            fictitiousSources);
        break;

    case 10:
        validateDirichletFmmMatvec<10>(
            config,
            boundaryPoints,
            fictitiousSources);
        break;

    case 12:
        validateDirichletFmmMatvec<12>(
            config,
            boundaryPoints,
            fictitiousSources);
        break;

    case 16:
        validateDirichletFmmMatvec<16>(
            config,
            boundaryPoints,
            fictitiousSources);
        break;

    default:
        throw std::runtime_error(
            "Unsupported P for FMM matvec validation. "
            "Allowed values: 8, 10, 12, 16.");
    }
}



std::vector<Point> convertBoundaryPointsToTargets(
    const std::vector<BoundaryPoint>& boundaryPoints)
{
    std::vector<Point> targets;

    targets.reserve(boundaryPoints.size());

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         ++index)
    {
        const auto& boundaryPoint =
            boundaryPoints[index];

        Point target;

        target.targetId =
            static_cast<long int>(index);

        target.ix = 0;
        target.iy = 0;

        target.x = boundaryPoint.x;
        target.y = boundaryPoint.y;

        // Для граничных точек это поле не используется.
        target.inside = true;

        targets.push_back(target);
    }

    return targets;
}


double getDirichletBoundaryPressureAtm(
    const Config& config)
{
    if (std::isnan(config.dirichletBoundaryAtm))
    {
        return config.p0Atm;
    }

    return config.dirichletBoundaryAtm;
}


std::vector<double> buildDirichletRhs(
    const Config& config,
    const std::vector<double>& realBoundaryDeltaPressurePa)
{
    const double boundaryPressureAtm =
        getDirichletBoundaryPressureAtm(config);

    const double boundaryDeltaPressurePa =
        (boundaryPressureAtm - config.p0Atm)
        * ATM_TO_PA;

    std::vector<double> rhs(
        realBoundaryDeltaPressurePa.size(),
        0.0);

    for (std::size_t index = 0;
         index < realBoundaryDeltaPressurePa.size();
         ++index)
    {
        rhs[index] =
            boundaryDeltaPressurePa
            - realBoundaryDeltaPressurePa[index];
    }

    return rhs;
}


void saveDirichletBoundaryRhs(
    const fs::path& filename,
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<double>& realBoundaryDeltaPressurePa,
    const std::vector<double>& rhsPa)
{
    if (boundaryPoints.size() != realBoundaryDeltaPressurePa.size()
        || boundaryPoints.size() != rhsPa.size())
    {
        throw std::runtime_error(
            "saveDirichletBoundaryRhs: size mismatch.");
    }

    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write Dirichlet RHS file: "
            + filename.string());
    }

    const double boundaryPressureAtm =
        getDirichletBoundaryPressureAtm(config);

    const double boundaryDeltaPressurePa =
        (boundaryPressureAtm - config.p0Atm)
        * ATM_TO_PA;

    output << std::setprecision(17);

    output
        << "boundary_id,segment_id,x,y,nx,ny,"
        << "p0_atm,boundary_pressure_atm,"
        << "boundary_delta_pressure_pa,"
        << "real_delta_pressure_pa,"
        << "real_delta_pressure_atm,"
        << "rhs_pa,"
        << "rhs_atm\n";

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         ++index)
    {
        const auto& point =
            boundaryPoints[index];

        output
            << point.boundaryId << ","
            << point.segmentId << ","
            << point.x << ","
            << point.y << ","
            << point.nx << ","
            << point.ny << ","
            << config.p0Atm << ","
            << boundaryPressureAtm << ","
            << boundaryDeltaPressurePa << ","
            << realBoundaryDeltaPressurePa[index] << ","
            << realBoundaryDeltaPressurePa[index] / ATM_TO_PA << ","
            << rhsPa[index] << ","
            << rhsPa[index] / ATM_TO_PA << "\n";
    }
}


void validateDirichletRhs(
    const Config& config,
    const std::vector<double>& realBoundaryDeltaPressurePa,
    const std::vector<double>& rhsPa)
{
    const double boundaryPressureAtm =
        getDirichletBoundaryPressureAtm(config);

    const double boundaryDeltaPressurePa =
        (boundaryPressureAtm - config.p0Atm)
        * ATM_TO_PA;

    const double realL2 =
        computeL2Norm(realBoundaryDeltaPressurePa);

    const double rhsL2 =
        computeL2Norm(rhsPa);

    const double realMax =
        computeMaxAbs(realBoundaryDeltaPressurePa);

    const double rhsMax =
        computeMaxAbs(rhsPa);

    std::cout
        << "Dirichlet RHS validation:\n"
        << "  boundary pressure = "
        << boundaryPressureAtm
        << " atm\n"
        << "  p0 = "
        << config.p0Atm
        << " atm\n"
        << "  boundary delta pressure = "
        << boundaryDeltaPressurePa
        << " Pa = "
        << boundaryDeltaPressurePa / ATM_TO_PA
        << " atm\n"
        << "  ||real boundary delta p||_2 = "
        << realL2
        << " Pa\n"
        << "  max|real boundary delta p| = "
        << realMax
        << " Pa = "
        << realMax / ATM_TO_PA
        << " atm\n"
        << "  ||rhs||_2 = "
        << rhsL2
        << " Pa\n"
        << "  max|rhs| = "
        << rhsMax
        << " Pa = "
        << rhsMax / ATM_TO_PA
        << " atm\n\n";

    if (std::abs(boundaryDeltaPressurePa) < 1e-12)
    {
        const auto check =
            addVectors(
                realBoundaryDeltaPressurePa,
                rhsPa);

        std::cout
            << "  check for p_boundary = p0: "
            << "||real + rhs||_2 / ||real||_2 = "
            << computeL2Norm(check)
                / std::max(realL2, 1e-300)
            << "\n\n";
    }
}

struct BoundarySystem
{
    std::vector<BoundaryPoint> boundaryPoints;
    std::vector<FictitiousSource> fictitiousSources;
};


struct GmresResult
{
    std::vector<double> solution;

    int iterations = 0;

    bool converged = false;

    double initialResidualNorm = 0.0;
    double finalResidualNorm = 0.0;
    double relativeResidual = 0.0;
};


double dotProduct(
    const std::vector<double>& a,
    const std::vector<double>& b)
{
    if (a.size() != b.size())
    {
        throw std::runtime_error(
            "dotProduct: size mismatch.");
    }

    double result = 0.0;

    for (std::size_t index = 0;
         index < a.size();
         ++index)
    {
        result += a[index] * b[index];
    }

    return result;
}


std::vector<double> multiplyVector(
    const std::vector<double>& values,
    const double coefficient)
{
    std::vector<double> result(
        values.size(),
        0.0);

    for (std::size_t index = 0;
         index < values.size();
         ++index)
    {
        result[index] =
            coefficient * values[index];
    }

    return result;
}


void addScaledVectorInPlace(
    std::vector<double>& target,
    const std::vector<double>& source,
    const double coefficient)
{
    if (target.size() != source.size())
    {
        throw std::runtime_error(
            "addScaledVectorInPlace: size mismatch.");
    }

    for (std::size_t index = 0;
         index < target.size();
         ++index)
    {
        target[index] += coefficient * source[index];
    }
}


BoundarySystem buildReducedBoundarySystem(
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const int stride)
{
    if (boundaryPoints.size() != fictitiousSources.size())
    {
        throw std::runtime_error(
            "buildReducedBoundarySystem: size mismatch.");
    }

    const int actualStride =
        std::max(1, stride);

    BoundarySystem result;

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         index += static_cast<std::size_t>(actualStride))
    {
        result.boundaryPoints.push_back(
            boundaryPoints[index]);

        result.fictitiousSources.push_back(
            fictitiousSources[index]);
    }

    if (result.boundaryPoints.empty())
    {
        throw std::runtime_error(
            "buildReducedBoundarySystem: empty reduced system.");
    }

    return result;
}


std::vector<double> solveDenseLinearSystem(
    std::vector<std::vector<double>> matrix,
    std::vector<double> rhs)
{
    const std::size_t n =
        rhs.size();

    if (matrix.size() != n)
    {
        throw std::runtime_error(
            "solveDenseLinearSystem: matrix row mismatch.");
    }

    for (const auto& row : matrix)
    {
        if (row.size() != n)
        {
            throw std::runtime_error(
                "solveDenseLinearSystem: matrix column mismatch.");
        }
    }

    for (std::size_t pivotIndex = 0;
         pivotIndex < n;
         ++pivotIndex)
    {
        std::size_t bestRow =
            pivotIndex;

        double bestValue =
            std::abs(matrix[pivotIndex][pivotIndex]);

        for (std::size_t row = pivotIndex + 1;
             row < n;
             ++row)
        {
            const double value =
                std::abs(matrix[row][pivotIndex]);

            if (value > bestValue)
            {
                bestValue = value;
                bestRow = row;
            }
        }

        if (bestValue < 1e-300)
        {
            throw std::runtime_error(
                "solveDenseLinearSystem: singular matrix.");
        }

        if (bestRow != pivotIndex)
        {
            std::swap(
                matrix[pivotIndex],
                matrix[bestRow]);

            std::swap(
                rhs[pivotIndex],
                rhs[bestRow]);
        }

        const double pivot =
            matrix[pivotIndex][pivotIndex];

        for (std::size_t row = pivotIndex + 1;
             row < n;
             ++row)
        {
            const double factor =
                matrix[row][pivotIndex] / pivot;

            matrix[row][pivotIndex] = 0.0;

            for (std::size_t col = pivotIndex + 1;
                 col < n;
                 ++col)
            {
                matrix[row][col] -=
                    factor * matrix[pivotIndex][col];
            }

            rhs[row] -=
                factor * rhs[pivotIndex];
        }
    }

    std::vector<double> solution(
        n,
        0.0);

    for (int row = static_cast<int>(n) - 1;
         row >= 0;
         --row)
    {
        double sum =
            rhs[static_cast<std::size_t>(row)];

        for (std::size_t col =
                 static_cast<std::size_t>(row) + 1;
             col < n;
             ++col)
        {
            sum -=
                matrix[static_cast<std::size_t>(row)][col]
                * solution[col];
        }

        solution[static_cast<std::size_t>(row)] =
            sum
            / matrix[static_cast<std::size_t>(row)]
                    [static_cast<std::size_t>(row)];
    }

    return solution;
}


std::vector<double> solveLeastSquaresByNormalEquations(
    const std::vector<std::vector<double>>& hessenberg,
    const double beta,
    const int columns)
{
    const int rows =
        columns + 1;

    std::vector<std::vector<double>> normalMatrix(
        static_cast<std::size_t>(columns),
        std::vector<double>(
            static_cast<std::size_t>(columns),
            0.0));

    std::vector<double> normalRhs(
        static_cast<std::size_t>(columns),
        0.0);

    for (int i = 0; i < columns; ++i)
    {
        normalRhs[static_cast<std::size_t>(i)] =
            beta * hessenberg[0][static_cast<std::size_t>(i)];

        for (int j = 0; j < columns; ++j)
        {
            double sum = 0.0;

            for (int row = 0; row < rows; ++row)
            {
                sum +=
                    hessenberg[static_cast<std::size_t>(row)]
                              [static_cast<std::size_t>(i)]
                    * hessenberg[static_cast<std::size_t>(row)]
                                [static_cast<std::size_t>(j)];
            }

            normalMatrix[static_cast<std::size_t>(i)]
                        [static_cast<std::size_t>(j)] =
                sum;
        }
    }

    return solveDenseLinearSystem(
        normalMatrix,
        normalRhs);
}


GmresResult runGmres(
    const std::function<std::vector<double>(
        const std::vector<double>&)>& applyOperator,
    const std::vector<double>& rhs,
    const int maxIterations,
    const double tolerance,
    const bool verbose)
{
    const std::size_t n =
        rhs.size();

    GmresResult result;

    result.solution.assign(
        n,
        0.0);

    std::vector<double> residual =
        rhs;

    const double beta =
        computeL2Norm(residual);

    result.initialResidualNorm =
        beta;

    if (beta < 1e-300)
    {
        result.finalResidualNorm = 0.0;
        result.relativeResidual = 0.0;
        result.converged = true;
        result.iterations = 0;

        return result;
    }

    const int iterationsLimit =
        std::min(
            maxIterations,
            static_cast<int>(n));

    std::vector<std::vector<double>> krylovBasis(
        static_cast<std::size_t>(iterationsLimit + 1),
        std::vector<double>(
            n,
            0.0));

    for (std::size_t index = 0;
         index < n;
         ++index)
    {
        krylovBasis[0][index] =
            residual[index] / beta;
    }

    std::vector<std::vector<double>> hessenberg(
        static_cast<std::size_t>(iterationsLimit + 1),
        std::vector<double>(
            static_cast<std::size_t>(iterationsLimit),
            0.0));

    std::vector<double> bestSolution(
        n,
        0.0);

    double bestResidualNorm =
        beta;

    for (int iteration = 0;
         iteration < iterationsLimit;
         ++iteration)
    {
        std::vector<double> w =
            applyOperator(
                krylovBasis[static_cast<std::size_t>(iteration)]);

        for (int basisIndex = 0;
             basisIndex <= iteration;
             ++basisIndex)
        {
            const double projection =
                dotProduct(
                    w,
                    krylovBasis[static_cast<std::size_t>(basisIndex)]);

            hessenberg[static_cast<std::size_t>(basisIndex)]
                      [static_cast<std::size_t>(iteration)] =
                projection;

            addScaledVectorInPlace(
                w,
                krylovBasis[static_cast<std::size_t>(basisIndex)],
                -projection);
        }

        const double nextNorm =
            computeL2Norm(w);

        hessenberg[static_cast<std::size_t>(iteration + 1)]
                  [static_cast<std::size_t>(iteration)] =
            nextNorm;

        if (nextNorm > 1e-300
            && iteration + 1 < iterationsLimit + 1)
        {
            for (std::size_t index = 0;
                 index < n;
                 ++index)
            {
                krylovBasis[static_cast<std::size_t>(iteration + 1)]
                           [index] =
                    w[index] / nextNorm;
            }
        }

        const int columns =
            iteration + 1;

        const auto y =
            solveLeastSquaresByNormalEquations(
                hessenberg,
                beta,
                columns);

        std::vector<double> candidate(
            n,
            0.0);

        for (int basisIndex = 0;
             basisIndex < columns;
             ++basisIndex)
        {
            addScaledVectorInPlace(
                candidate,
                krylovBasis[static_cast<std::size_t>(basisIndex)],
                y[static_cast<std::size_t>(basisIndex)]);
        }

        const auto applied =
            applyOperator(candidate);

        const auto currentResidual =
            subtractVectors(
                rhs,
                applied);

        const double residualNorm =
            computeL2Norm(currentResidual);

        const double relativeResidual =
            residualNorm / beta;

        if (residualNorm < bestResidualNorm)
        {
            bestResidualNorm =
                residualNorm;

            bestSolution =
                candidate;
        }

        if (verbose)
        {
            std::cout
                << "  GMRES iteration "
                << columns
                << ": relative residual = "
                << relativeResidual
                << "\n";
        }

        if (relativeResidual <= tolerance)
        {
            result.solution =
                candidate;

            result.iterations =
                columns;

            result.finalResidualNorm =
                residualNorm;

            result.relativeResidual =
                relativeResidual;

            result.converged =
                true;

            return result;
        }

        if (nextNorm <= 1e-300)
        {
            result.solution =
                candidate;

            result.iterations =
                columns;

            result.finalResidualNorm =
                residualNorm;

            result.relativeResidual =
                relativeResidual;

            result.converged =
                relativeResidual <= tolerance;

            return result;
        }
    }

    result.solution =
        bestSolution;

    result.iterations =
        iterationsLimit;

    result.finalResidualNorm =
        bestResidualNorm;

    result.relativeResidual =
        bestResidualNorm / beta;

    result.converged =
        result.relativeResidual <= tolerance;

    return result;
}


void saveFictitiousSolution(
    const fs::path& filename,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& solutionM3Day)
{
    if (fictitiousSources.size() != solutionM3Day.size())
    {
        throw std::runtime_error(
            "saveFictitiousSolution: size mismatch.");
    }

    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write fictitious solution file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "source_id,boundary_id,x,y,offset_m,q_fic_m3_day\n";

    for (std::size_t index = 0;
         index < fictitiousSources.size();
         ++index)
    {
        const auto& source =
            fictitiousSources[index];

        output
            << source.sourceId << ","
            << source.boundaryId << ","
            << source.x << ","
            << source.y << ","
            << source.offsetM << ","
            << solutionM3Day[index] << "\n";
    }
}


void saveGmresBoundaryResidual(
    const fs::path& filename,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<double>& rhsPa,
    const std::vector<double>& aqPa)
{
    if (boundaryPoints.size() != rhsPa.size()
        || boundaryPoints.size() != aqPa.size())
    {
        throw std::runtime_error(
            "saveGmresBoundaryResidual: size mismatch.");
    }

    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write GMRES residual file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "boundary_id,segment_id,x,y,nx,ny,"
        << "rhs_pa,aq_pa,system_residual_pa,"
        << "system_residual_atm,boundary_residual_pa,"
        << "boundary_residual_atm\n";

    for (std::size_t index = 0;
         index < boundaryPoints.size();
         ++index)
    {
        const auto& point =
            boundaryPoints[index];

        const double systemResidual =
            rhsPa[index] - aqPa[index];

        // Граничная невязка:
        // real + fic - boundary = A q - rhs.
        // Она отличается знаком от системной невязки.
        const double boundaryResidual =
            aqPa[index] - rhsPa[index];

        output
            << point.boundaryId << ","
            << point.segmentId << ","
            << point.x << ","
            << point.y << ","
            << point.nx << ","
            << point.ny << ","
            << rhsPa[index] << ","
            << aqPa[index] << ","
            << systemResidual << ","
            << systemResidual / ATM_TO_PA << ","
            << boundaryResidual << ","
            << boundaryResidual / ATM_TO_PA << "\n";
    }
}


void saveGmresSummary(
    const fs::path& filename,
    const GmresResult& result,
    const std::size_t systemSize,
    const int stride)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write GMRES summary file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "system_size,boundary_stride,iterations,"
        << "converged,initial_residual_norm,"
        << "final_residual_norm,relative_residual\n";

    output
        << systemSize << ","
        << stride << ","
        << result.iterations << ","
        << static_cast<int>(result.converged) << ","
        << result.initialResidualNorm << ","
        << result.finalResidualNorm << ","
        << result.relativeResidual << "\n";
}

std::vector<double> computeDirectMonth(
    const Config& config,
    const std::vector<Well>& wells,
    const std::vector<Point>& targets,
    const int monthIndex);

struct FictitiousHistoryLayer
{
    int month = 0;

    std::vector<double> qFicM3Day;

    int iterations = 0;
    bool converged = false;

    double relativeResidual = 0.0;
    double maxBoundaryResidualPa = 0.0;
};


int getHistoryEndMonth(
    const Config& config)
{
    if (config.historyEndMonth > 0)
    {
        return config.historyEndMonth;
    }

    return config.targetMonth;
}


std::vector<double> buildDirichletRhsWithKnownFictitious(
    const Config& config,
    const std::vector<double>& realBoundaryDeltaPressurePa,
    const std::vector<double>& knownFictitiousBoundaryDeltaPressurePa)
{
    if (realBoundaryDeltaPressurePa.size()
        != knownFictitiousBoundaryDeltaPressurePa.size())
    {
        throw std::runtime_error(
            "buildDirichletRhsWithKnownFictitious: size mismatch.");
    }

    const double boundaryPressureAtm =
        getDirichletBoundaryPressureAtm(config);

    const double boundaryDeltaPressurePa =
        (boundaryPressureAtm - config.p0Atm)
        * ATM_TO_PA;

    std::vector<double> rhs(
        realBoundaryDeltaPressurePa.size(),
        0.0);

    for (std::size_t index = 0;
         index < rhs.size();
         ++index)
    {
        rhs[index] =
            boundaryDeltaPressurePa
            - realBoundaryDeltaPressurePa[index]
            - knownFictitiousBoundaryDeltaPressurePa[index];
    }

    return rhs;
}


std::vector<double> computeOldFictitiousBoundaryContributionDirect(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history,
    const int finalMonth)
{
    std::vector<double> result(
        boundaryPoints.size(),
        0.0);

    for (const auto& layer : history)
    {
        if (layer.month >= finalMonth)
        {
            continue;
        }

        const auto contribution =
            applyDirichletFictitiousIntervalDirect(
                config,
                boundaryPoints,
                fictitiousSources,
                layer.qFicM3Day,
                finalMonth,
                layer.month);

        addScaledVectorInPlace(
            result,
            contribution,
            1.0);
    }

    return result;
}


void saveFictitiousHistoryDirect(
    const fs::path& filename,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write fictitious history file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "month,source_id,boundary_id,x,y,offset_m,q_fic_m3_day\n";

    for (const auto& layer : history)
    {
        if (layer.qFicM3Day.size() != fictitiousSources.size())
        {
            throw std::runtime_error(
                "saveFictitiousHistoryDirect: size mismatch.");
        }

        for (std::size_t index = 0;
             index < fictitiousSources.size();
             ++index)
        {
            const auto& source =
                fictitiousSources[index];

            output
                << layer.month << ","
                << source.sourceId << ","
                << source.boundaryId << ","
                << source.x << ","
                << source.y << ","
                << source.offsetM << ","
                << layer.qFicM3Day[index] << "\n";
        }
    }
}


void saveFictitiousHistorySummaryDirect(
    const fs::path& filename,
    const std::vector<FictitiousHistoryLayer>& history)
{
    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write fictitious history summary file: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "month,iterations,converged,"
        << "relative_residual,max_boundary_residual_pa,"
        << "max_boundary_residual_atm\n";

    for (const auto& layer : history)
    {
        output
            << layer.month << ","
            << layer.iterations << ","
            << static_cast<int>(layer.converged) << ","
            << layer.relativeResidual << ","
            << layer.maxBoundaryResidualPa << ","
            << layer.maxBoundaryResidualPa / ATM_TO_PA << "\n";
    }
}


std::vector<FictitiousHistoryLayer> runFictitiousHistoryDirectValidation(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<Well>& wells)
{
    const int historyEndMonth =
        getHistoryEndMonth(config);

    if (historyEndMonth <= 0)
    {
        throw std::runtime_error(
            "runFictitiousHistoryDirectValidation: invalid history end month.");
    }

    const auto reducedSystem =
        buildReducedBoundarySystem(
            boundaryPoints,
            fictitiousSources,
            config.gmresBoundaryStride);

    const auto reducedBoundaryTargets =
        convertBoundaryPointsToTargets(
            reducedSystem.boundaryPoints);

    std::vector<FictitiousHistoryLayer> history;

    history.reserve(
        static_cast<std::size_t>(historyEndMonth));

    std::cout
        << "Direct fictitious history validation:\n"
        << "  full boundary points = "
        << boundaryPoints.size()
        << "\n"
        << "  reduced boundary points = "
        << reducedSystem.boundaryPoints.size()
        << "\n"
        << "  boundary stride = "
        << config.gmresBoundaryStride
        << "\n"
        << "  history end month = "
        << historyEndMonth
        << "\n"
        << "  max GMRES iterations = "
        << config.gmresMaxIterations
        << "\n"
        << "  GMRES tolerance = "
        << config.gmresTolerance
        << "\n\n";

    const auto totalStartTime =
        std::chrono::high_resolution_clock::now();

    for (int month = 1;
         month <= historyEndMonth;
         ++month)
    {
        const auto realBoundaryDeltaPressurePa =
            computeDirectMonth(
                config,
                wells,
                reducedBoundaryTargets,
                month);

        const auto oldFictitiousBoundaryPa =
            computeOldFictitiousBoundaryContributionDirect(
                config,
                reducedSystem.boundaryPoints,
                reducedSystem.fictitiousSources,
                history,
                month);

        const auto rhsPa =
            buildDirichletRhsWithKnownFictitious(
                config,
                realBoundaryDeltaPressurePa,
                oldFictitiousBoundaryPa);

        const auto applyCurrentLayer =
            [&](const std::vector<double>& qCurrentM3Day)
            {
                return applyDirichletFictitiousIntervalDirect(
                    config,
                    reducedSystem.boundaryPoints,
                    reducedSystem.fictitiousSources,
                    qCurrentM3Day,
                    month,
                    month);
            };

        const auto monthStartTime =
            std::chrono::high_resolution_clock::now();

        const auto gmresResult =
            runGmres(
                applyCurrentLayer,
                rhsPa,
                config.gmresMaxIterations,
                config.gmresTolerance,
                config.gmresVerbose);

        const auto monthEndTime =
            std::chrono::high_resolution_clock::now();

        const auto currentFictitiousBoundaryPa =
            applyCurrentLayer(
                gmresResult.solution);

        const auto systemResidualPa =
            subtractVectors(
                rhsPa,
                currentFictitiousBoundaryPa);

        const double maxBoundaryResidualPa =
            computeMaxAbs(systemResidualPa);

        const double monthTimeSec =
            std::chrono::duration<double>(
                monthEndTime - monthStartTime).count();

        FictitiousHistoryLayer layer;

        layer.month = month;
        layer.qFicM3Day = gmresResult.solution;
        layer.iterations = gmresResult.iterations;
        layer.converged = gmresResult.converged;
        layer.relativeResidual = gmresResult.relativeResidual;
        layer.maxBoundaryResidualPa = maxBoundaryResidualPa;

        history.push_back(layer);

        std::cout
            << "  month "
            << month
            << " / "
            << historyEndMonth
            << ": converged = "
            << gmresResult.converged
            << ", iterations = "
            << gmresResult.iterations
            << ", rel residual = "
            << gmresResult.relativeResidual
            << ", max boundary residual = "
            << maxBoundaryResidualPa / ATM_TO_PA
            << " atm"
            << ", time = "
            << monthTimeSec
            << " sec\n";
    }

    const auto totalEndTime =
        std::chrono::high_resolution_clock::now();

    const double totalTimeSec =
        std::chrono::duration<double>(
            totalEndTime - totalStartTime).count();

    std::cout
        << "\nDirect fictitious history completed:\n"
        << "  layers = "
        << history.size()
        << "\n"
        << "  total time = "
        << totalTimeSec
        << " sec\n\n";

    const fs::path historyFile =
        config.outputDir
        / "dirichlet_fictitious_history_direct.csv";

    const fs::path summaryFile =
        config.outputDir
        / "dirichlet_fictitious_history_summary_direct.csv";

    saveFictitiousHistoryDirect(
        historyFile,
        reducedSystem.fictitiousSources,
        history);

    saveFictitiousHistorySummaryDirect(
        summaryFile,
        history);

    std::cout
        << "Saved direct fictitious history: "
        << historyFile
        << "\n";

    std::cout
        << "Saved direct fictitious history summary: "
        << summaryFile
        << "\n";
    return history;
}

std::vector<double> applyFictitiousIntervalDirectToPoints(
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& qFictitiousM3Day,
    const int finalMonth,
    const int sourceMonth)
{
    if (fictitiousSources.size() != qFictitiousM3Day.size())
    {
        throw std::runtime_error(
            "applyFictitiousIntervalDirectToPoints: "
            "sources and vector size mismatch.");
    }

    if (finalMonth < sourceMonth)
    {
        throw std::runtime_error(
            "applyFictitiousIntervalDirectToPoints: "
            "finalMonth must be >= sourceMonth.");
    }

    const double kappa =
        computeKappa(config);

    const double prefactor =
        computePrefactor(config);

    const double dtSeconds =
        config.dtDays * DAY_TO_SEC;

    const auto quadrature =
        getGaussLegendreRule(
            config.timeQuadPoints);

    const double finalTime =
        static_cast<double>(finalMonth) * dtSeconds;

    const double intervalStart =
        static_cast<double>(sourceMonth - 1) * dtSeconds;

    const double intervalEnd =
        static_cast<double>(sourceMonth) * dtSeconds;

    std::vector<double> pressurePa(
        targets.size(),
        0.0);

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
            std::max(
                finalTime - tau,
                1e-12);

        const double delta =
            4.0 * kappa * lagSeconds;

        const double coefficientForRate =
            prefactor
            * weightTime
            / lagSeconds
            / DAY_TO_SEC;

        for (std::size_t targetIndex = 0;
             targetIndex < targets.size();
             ++targetIndex)
        {
            const auto& target =
                targets[targetIndex];

            if (!target.inside)
            {
                continue;
            }

            double sum = 0.0;

            for (std::size_t sourceIndex = 0;
                 sourceIndex < fictitiousSources.size();
                 ++sourceIndex)
            {
                const auto& source =
                    fictitiousSources[sourceIndex];

                const double dx =
                    target.x - source.x;

                const double dy =
                    target.y - source.y;

                const double rSquared =
                    dx * dx + dy * dy;

                sum +=
                    qFictitiousM3Day[sourceIndex]
                    * coefficientForRate
                    * std::exp(-rSquared / delta);
            }

            pressurePa[targetIndex] += sum;
        }
    }

    return pressurePa;
}


std::vector<double> computeFictitiousHistoryOnTargetsDirect(
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history,
    const int finalMonth)
{
    std::vector<double> result(
        targets.size(),
        0.0);

    for (const auto& layer : history)
    {
        if (layer.month > finalMonth)
        {
            continue;
        }

        const auto contribution =
            applyFictitiousIntervalDirectToPoints(
                config,
                targets,
                fictitiousSources,
                layer.qFicM3Day,
                finalMonth,
                layer.month);

        addScaledVectorInPlace(
            result,
            contribution,
            1.0);
    }

    return result;
}

void saveDirichletHistoryPressureMap(
    const fs::path& filename,
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<double>& realDeltaPa,
    const std::vector<double>& fictitiousDeltaPa,
    const int finalMonth)
{
    if (targets.size() != realDeltaPa.size()
        || targets.size() != fictitiousDeltaPa.size())
    {
        throw std::runtime_error(
            "saveDirichletHistoryPressureMap: size mismatch.");
    }

    std::ofstream output(filename);

    if (!output)
    {
        throw std::runtime_error(
            "Cannot write Dirichlet history pressure map: "
            + filename.string());
    }

    output << std::setprecision(17);

    output
        << "target_id,ix,iy,x,y,inside,month,"
        << "real_delta_pa,"
        << "fictitious_delta_pa,"
        << "total_delta_pa,"
        << "real_pressure_atm,"
        << "total_pressure_atm\n";

    for (std::size_t index = 0;
         index < targets.size();
         ++index)
    {
        const auto& target =
            targets[index];

        const double realPa =
            realDeltaPa[index];

        const double fictitiousPa =
            fictitiousDeltaPa[index];

        const double totalDeltaPa =
            realPa + fictitiousPa;

        const double realPressureAtm =
            config.p0Atm
            + realPa / ATM_TO_PA;

        const double totalPressureAtm =
            config.p0Atm
            + totalDeltaPa / ATM_TO_PA;

        output
            << target.targetId << ","
            << target.ix << ","
            << target.iy << ","
            << target.x << ","
            << target.y << ","
            << static_cast<int>(target.inside) << ","
            << finalMonth << ","
            << realPa << ","
            << fictitiousPa << ","
            << totalDeltaPa << ","
            << realPressureAtm << ","
            << totalPressureAtm << "\n";
    }
}

void saveDirichletHistoryPressureMapsDirect(
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<Well>& wells,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history,
    const int finalMonth)
{
    const fs::path mapsDir =
        config.outputDir / "dirichlet_history_pressure_maps_direct";

    fs::create_directories(mapsDir);

    const fs::path indexFile =
        mapsDir / "index.csv";

    std::ofstream indexOutput(indexFile);

    if (!indexOutput)
    {
        throw std::runtime_error(
            "Cannot write maps index file: "
            + indexFile.string());
    }

    indexOutput
        << "month,file\n";

    std::cout
        << "\nSaving pressure-map dynamics with fictitious history:\n"
        << "  maps dir = "
        << mapsDir
        << "\n"
        << "  months = 1.."
        << finalMonth
        << "\n"
        << "  targets = "
        << targets.size()
        << "\n"
        << "  fictitious sources = "
        << fictitiousSources.size()
        << "\n\n";

    for (int month = 1;
         month <= finalMonth;
         ++month)
    {
        const auto mapStartTime =
            std::chrono::high_resolution_clock::now();

        const auto realDeltaPa =
            computeDirectMonth(
                config,
                wells,
                targets,
                month);

        const auto fictitiousDeltaPa =
            computeFictitiousHistoryOnTargetsDirect(
                config,
                targets,
                fictitiousSources,
                history,
                month);

        std::ostringstream filenameBuilder;

        filenameBuilder
            << "dirichlet_history_pressure_map_month_"
            << std::setw(3)
            << std::setfill('0')
            << month
            << ".csv";

        const fs::path mapFile =
            mapsDir / filenameBuilder.str();

        saveDirichletHistoryPressureMap(
            mapFile,
            config,
            targets,
            realDeltaPa,
            fictitiousDeltaPa,
            month);

        indexOutput
            << month
            << ","
            << mapFile.filename().string()
            << "\n";

        const auto mapEndTime =
            std::chrono::high_resolution_clock::now();

        const double mapTimeSec =
            std::chrono::duration<double>(
                mapEndTime - mapStartTime).count();

        std::cout
            << "  saved month "
            << month
            << " / "
            << finalMonth
            << ": "
            << mapFile
            << ", time = "
            << mapTimeSec
            << " sec\n";
    }

    std::cout
        << "\nSaved pressure-map dynamics index: "
        << indexFile
        << "\n";
}

void validateDirichletMatvecDirect(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    const auto q1 =
        buildRandomFictitiousRates(
            fictitiousSources.size(),
            config.matvecTestRateScale,
            config.matvecTestSeed);

    const auto q2 =
        buildRandomFictitiousRates(
            fictitiousSources.size(),
            config.matvecTestRateScale,
            config.matvecTestSeed + 1);

    const auto qSum =
        addVectors(q1, q2);

    const auto y1 =
        applyDirichletFictitiousMatvecDirect(
            config,
            boundaryPoints,
            fictitiousSources,
            q1);

    const auto y2 =
        applyDirichletFictitiousMatvecDirect(
            config,
            boundaryPoints,
            fictitiousSources,
            q2);

    const auto ySum =
        applyDirichletFictitiousMatvecDirect(
            config,
            boundaryPoints,
            fictitiousSources,
            qSum);

    const auto y1PlusY2 =
        addVectors(y1, y2);

    const auto linearityDiff =
        subtractVectors(
            ySum,
            y1PlusY2);

    const double relativeLinearityError =
        computeL2Norm(linearityDiff)
        / std::max(
            computeL2Norm(ySum),
            1e-300);

    std::cout
        << "Dirichlet direct matvec validation:\n"
        << "  unknowns = "
        << fictitiousSources.size()
        << "\n"
        << "  equations = "
        << boundaryPoints.size()
        << "\n"
        << "  test rate scale = "
        << config.matvecTestRateScale
        << " m^3/day\n"
        << "  ||A*q1||_2 = "
        << computeL2Norm(y1)
        << " Pa\n"
        << "  max|A*q1| = "
        << computeMaxAbs(y1)
        << " Pa\n"
        << "  relative linearity error = "
        << relativeLinearityError
        << "\n\n";
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


template <unsigned int P>
void runDirichletFmmGmresValidation(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<Well>& wells)
{
    const auto reducedSystem =
        buildReducedBoundarySystem(
            boundaryPoints,
            fictitiousSources,
            config.gmresBoundaryStride);

    const auto reducedBoundaryTargets =
        convertBoundaryPointsToTargets(
            reducedSystem.boundaryPoints);

    const auto reducedRealBoundaryDeltaPressurePa =
        computeDirectMonth(
            config,
            wells,
            reducedBoundaryTargets,
            config.targetMonth);

    const auto reducedRhsPa =
        buildDirichletRhs(
            config,
            reducedRealBoundaryDeltaPressurePa);

    std::cout
        << "FMM GMRES validation system:\n"
        << "  full boundary points = "
        << boundaryPoints.size()
        << "\n"
        << "  reduced boundary points = "
        << reducedSystem.boundaryPoints.size()
        << "\n"
        << "  boundary stride = "
        << config.gmresBoundaryStride
        << "\n"
        << "  P = "
        << P
        << "\n"
        << "  max iterations = "
        << config.gmresMaxIterations
        << "\n"
        << "  tolerance = "
        << config.gmresTolerance
        << "\n\n";

    const auto applyOperatorFmm =
        [&](const std::vector<double>& vectorM3Day)
        {
            return applyDirichletFictitiousMatvecFmm<P>(
                config,
                reducedSystem.boundaryPoints,
                reducedSystem.fictitiousSources,
                vectorM3Day);
        };

    const auto startTime =
        std::chrono::high_resolution_clock::now();

    const auto gmresResult =
        runGmres(
            applyOperatorFmm,
            reducedRhsPa,
            config.gmresMaxIterations,
            config.gmresTolerance,
            config.gmresVerbose);

    const auto endTime =
        std::chrono::high_resolution_clock::now();

    const double elapsedSeconds =
        std::chrono::duration<double>(
            endTime - startTime).count();

    const auto aqPa =
        applyOperatorFmm(
            gmresResult.solution);

    const auto residualPa =
        subtractVectors(
            reducedRhsPa,
            aqPa);

    std::cout
        << "\nFMM GMRES result:\n"
        << "  converged = "
        << gmresResult.converged
        << "\n"
        << "  iterations = "
        << gmresResult.iterations
        << "\n"
        << "  initial residual norm = "
        << gmresResult.initialResidualNorm
        << " Pa\n"
        << "  final residual norm = "
        << gmresResult.finalResidualNorm
        << " Pa\n"
        << "  relative residual = "
        << gmresResult.relativeResidual
        << "\n"
        << "  max|residual| = "
        << computeMaxAbs(residualPa)
        << " Pa = "
        << computeMaxAbs(residualPa) / ATM_TO_PA
        << " atm\n"
        << "  elapsed = "
        << elapsedSeconds
        << " sec\n\n";

    const fs::path solutionFile =
        config.outputDir
        / "dirichlet_fictitious_solution_fmm_gmres.csv";

    const fs::path residualFile =
        config.outputDir
        / "dirichlet_boundary_residual_fmm_gmres.csv";

    const fs::path summaryFile =
        config.outputDir
        / "dirichlet_gmres_summary_fmm.csv";

    saveFictitiousSolution(
        solutionFile,
        reducedSystem.fictitiousSources,
        gmresResult.solution);

    saveGmresBoundaryResidual(
        residualFile,
        reducedSystem.boundaryPoints,
        reducedRhsPa,
        aqPa);

    saveGmresSummary(
        summaryFile,
        gmresResult,
        reducedSystem.boundaryPoints.size(),
        config.gmresBoundaryStride);

    std::cout
        << "Saved FMM GMRES fictitious solution: "
        << solutionFile
        << "\n";

    std::cout
        << "Saved FMM GMRES boundary residual: "
        << residualFile
        << "\n";

    std::cout
        << "Saved FMM GMRES summary: "
        << summaryFile
        << "\n";
}


void runDirichletFmmGmresValidationByOrder(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<Well>& wells)
{
    switch (config.pOrder)
    {
    case 8:
        runDirichletFmmGmresValidation<8>(
            config,
            boundaryPoints,
            fictitiousSources,
            wells);
        break;

    case 10:
        runDirichletFmmGmresValidation<10>(
            config,
            boundaryPoints,
            fictitiousSources,
            wells);
        break;

    case 12:
        runDirichletFmmGmresValidation<12>(
            config,
            boundaryPoints,
            fictitiousSources,
            wells);
        break;

    case 16:
        runDirichletFmmGmresValidation<16>(
            config,
            boundaryPoints,
            fictitiousSources,
            wells);
        break;

    default:
        throw std::runtime_error(
            "Unsupported P for FMM GMRES validation. "
            "Allowed values: 8, 10, 12, 16.");
    }
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

        // ======================================================
        // Шаг 1 прототипа Дирихле:
        // строим точки коллокации на реальном контуре
        // и фиктивные источники снаружи области.
        // ======================================================

        const auto boundaryPoints =
            buildBoundaryCollocationPoints(
                contour,
                config.boundarySubdivisionsPerSegment);

        const auto fictitiousSources =
            buildFictitiousSources(
                config,
                contour,
                boundaryPoints);

        validateBoundarySetup(
            contour,
            boundaryPoints,
            fictitiousSources);

        const fs::path boundaryPointsFile =
            config.outputDir
            / "dirichlet_boundary_points.csv";

        const fs::path fictitiousSourcesFile =
            config.outputDir
            / "dirichlet_fictitious_sources.csv";

        saveBoundaryPoints(
            boundaryPointsFile,
            boundaryPoints);

        saveFictitiousSources(
            fictitiousSourcesFile,
            fictitiousSources);

        std::cout
            << "Saved boundary points: "
            << boundaryPointsFile
            << "\n";

        std::cout
            << "Saved fictitious sources: "
            << fictitiousSourcesFile
            << "\n\n";

        if (config.stopAfterBoundary)
        {
            std::cout
                << "Stopped after boundary/fictitious source validation.\n";

            return 0;
        }

        if (config.stopAfterBoundaryMatvec)
        {
            validateDirichletMatvecDirect(
                config,
                boundaryPoints,
                fictitiousSources);

            const auto qTest =
                buildRandomFictitiousRates(
                    fictitiousSources.size(),
                    config.matvecTestRateScale,
                    config.matvecTestSeed);

            const auto boundaryPressurePa =
                applyDirichletFictitiousMatvecDirect(
                    config,
                    boundaryPoints,
                    fictitiousSources,
                    qTest);

            const fs::path matvecFile =
                config.outputDir
                / "dirichlet_boundary_matvec_direct.csv";

            saveBoundaryMatvecResult(
                matvecFile,
                boundaryPoints,
                boundaryPressurePa);

            std::cout
                << "Saved direct boundary matvec result: "
                << matvecFile
                << "\n";

            std::cout
                << "Stopped after direct Dirichlet matvec validation.\n";

            return 0;
        }

        const auto boundaryTargets =
            convertBoundaryPointsToTargets(
                boundaryPoints);

        const auto realBoundaryDeltaPressurePa =
            computeDirectMonth(
                config,
                wells,
                boundaryTargets,
                config.targetMonth);

        const auto dirichletRhsPa =
            buildDirichletRhs(
                config,
                realBoundaryDeltaPressurePa);

        validateDirichletRhs(
            config,
            realBoundaryDeltaPressurePa,
            dirichletRhsPa);

        const fs::path rhsFile =
            config.outputDir
            / "dirichlet_boundary_rhs.csv";

        saveDirichletBoundaryRhs(
            rhsFile,
            config,
            boundaryPoints,
            realBoundaryDeltaPressurePa,
            dirichletRhsPa);

        std::cout
            << "Saved Dirichlet RHS: "
            << rhsFile
            << "\n\n";

        if (config.stopAfterBoundaryRhs)
        {
            std::cout
                << "Stopped after Dirichlet RHS validation.\n";

            return 0;
        }

        if (config.stopAfterFictitiousIntervalValidation)
        {
            const auto reducedSystem =
                buildReducedBoundarySystem(
                    boundaryPoints,
                    fictitiousSources,
                    config.gmresBoundaryStride);

            std::cout
                << "Fictitious interval validation system:\n"
                << "  full boundary points = "
                << boundaryPoints.size()
                << "\n"
                << "  reduced boundary points = "
                << reducedSystem.boundaryPoints.size()
                << "\n"
                << "  boundary stride = "
                << config.gmresBoundaryStride
                << "\n\n";

            validateFictitiousIntervalDirectConsistency(
                config,
                reducedSystem.boundaryPoints,
                reducedSystem.fictitiousSources);

            std::cout
                << "Stopped after fictitious interval validation.\n";

            return 0;
        }

        if (config.stopAfterFictitiousHistoryDirect)
        {
            runFictitiousHistoryDirectValidation(
                config,
                boundaryPoints,
                fictitiousSources,
                wells);

            std::cout
                << "Stopped after direct fictitious history validation.\n";

            return 0;
        }
        
        if (config.stopAfterFictitiousHistoryAllMapsDirect)
        {
            const int finalMonth =
                getHistoryEndMonth(config);

            const auto reducedSystem =
                buildReducedBoundarySystem(
                    boundaryPoints,
                    fictitiousSources,
                    config.gmresBoundaryStride);

            const auto history =
                runFictitiousHistoryDirectValidation(
                    config,
                    boundaryPoints,
                    fictitiousSources,
                    wells);

            const auto targets =
                buildGrid(
                    contour,
                    config.gridSize);

            saveDirichletHistoryPressureMapsDirect(
                config,
                targets,
                wells,
                reducedSystem.fictitiousSources,
                history,
                finalMonth);

            std::cout
                << "Stopped after saving all Dirichlet history pressure maps.\n";

            return 0;
        }
        if (config.stopAfterDirectGmres)
        {
            const auto reducedSystem =
                buildReducedBoundarySystem(
                    boundaryPoints,
                    fictitiousSources,
                    config.gmresBoundaryStride);

            const auto reducedBoundaryTargets =
                convertBoundaryPointsToTargets(
                    reducedSystem.boundaryPoints);

            const auto reducedRealBoundaryDeltaPressurePa =
                computeDirectMonth(
                    config,
                    wells,
                    reducedBoundaryTargets,
                    config.targetMonth);

            const auto reducedRhsPa =
                buildDirichletRhs(
                    config,
                    reducedRealBoundaryDeltaPressurePa);

            std::cout
                << "Direct GMRES validation system:\n"
                << "  full boundary points = "
                << boundaryPoints.size()
                << "\n"
                << "  reduced boundary points = "
                << reducedSystem.boundaryPoints.size()
                << "\n"
                << "  boundary stride = "
                << config.gmresBoundaryStride
                << "\n"
                << "  max iterations = "
                << config.gmresMaxIterations
                << "\n"
                << "  tolerance = "
                << config.gmresTolerance
                << "\n\n";

            const auto applyOperatorDirect =
                [&](const std::vector<double>& vectorM3Day)
                {
                    return applyDirichletFictitiousMatvecDirect(
                        config,
                        reducedSystem.boundaryPoints,
                        reducedSystem.fictitiousSources,
                        vectorM3Day);
                };

            const auto startTime =
                std::chrono::high_resolution_clock::now();

            const auto gmresResult =
                runGmres(
                    applyOperatorDirect,
                    reducedRhsPa,
                    config.gmresMaxIterations,
                    config.gmresTolerance,
                    config.gmresVerbose);

            const auto endTime =
                std::chrono::high_resolution_clock::now();

            const double elapsedSeconds =
                std::chrono::duration<double>(
                    endTime - startTime).count();

            const auto aqPa =
                applyOperatorDirect(
                    gmresResult.solution);

            const auto residualPa =
                subtractVectors(
                    reducedRhsPa,
                    aqPa);

            std::cout
                << "\nDirect GMRES result:\n"
                << "  converged = "
                << gmresResult.converged
                << "\n"
                << "  iterations = "
                << gmresResult.iterations
                << "\n"
                << "  initial residual norm = "
                << gmresResult.initialResidualNorm
                << " Pa\n"
                << "  final residual norm = "
                << gmresResult.finalResidualNorm
                << " Pa\n"
                << "  relative residual = "
                << gmresResult.relativeResidual
                << "\n"
                << "  max|residual| = "
                << computeMaxAbs(residualPa)
                << " Pa = "
                << computeMaxAbs(residualPa) / ATM_TO_PA
                << " atm\n"
                << "  elapsed = "
                << elapsedSeconds
                << " sec\n\n";

            const fs::path solutionFile =
                config.outputDir
                / "dirichlet_fictitious_solution_direct_gmres.csv";

            const fs::path residualFile =
                config.outputDir
                / "dirichlet_boundary_residual_direct_gmres.csv";

            const fs::path summaryFile =
                config.outputDir
                / "dirichlet_gmres_summary_direct.csv";

            saveFictitiousSolution(
                solutionFile,
                reducedSystem.fictitiousSources,
                gmresResult.solution);

            saveGmresBoundaryResidual(
                residualFile,
                reducedSystem.boundaryPoints,
                reducedRhsPa,
                aqPa);

            saveGmresSummary(
                summaryFile,
                gmresResult,
                reducedSystem.boundaryPoints.size(),
                config.gmresBoundaryStride);

            std::cout
                << "Saved direct GMRES fictitious solution: "
                << solutionFile
                << "\n";

            std::cout
                << "Saved direct GMRES boundary residual: "
                << residualFile
                << "\n";

            std::cout
                << "Saved direct GMRES summary: "
                << summaryFile
                << "\n";

            std::cout
                << "Stopped after direct GMRES validation.\n";

            return 0;
        }

        if (config.stopAfterFmmMatvec)
        {
            const auto reducedSystem =
                buildReducedBoundarySystem(
                    boundaryPoints,
                    fictitiousSources,
                    config.gmresBoundaryStride);

            std::cout
                << "FMM matvec validation system:\n"
                << "  full boundary points = "
                << boundaryPoints.size()
                << "\n"
                << "  reduced boundary points = "
                << reducedSystem.boundaryPoints.size()
                << "\n"
                << "  boundary stride = "
                << config.gmresBoundaryStride
                << "\n\n";

            validateDirichletFmmMatvecByOrder(
                config,
                reducedSystem.boundaryPoints,
                reducedSystem.fictitiousSources);

            std::cout
                << "Stopped after FMM matvec validation.\n";

            return 0;
        }
        if (config.stopAfterFmmGmres)
        {
            runDirichletFmmGmresValidationByOrder(
                config,
                boundaryPoints,
                fictitiousSources,
                wells);

            std::cout
                << "Stopped after FMM GMRES validation.\n";

            return 0;
        }
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
