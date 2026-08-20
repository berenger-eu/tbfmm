/**
 * Clean real-case transient Dirichlet prototype.
 *
 * Purpose:
 *   1. Read real wells, monthly well rates and outer contour.
 *   2. Build Dirichlet boundary collocation points and outside fictitious sources.
 *   3. Solve monthly fictitious-source history with Tikhonov regularization.
 *   4. Save pressure-map dynamics for months 1..historyEndMonth.
 *
 * This file intentionally contains no FMM benchmark code and no temporary
 * validation branches. It is a clean direct prototype for debugging the
 * physics and the time history before accelerating the matvec by FMM.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double MD_TO_M2 = 9.869233e-16;
constexpr double CP_TO_PA_S = 1e-3;
constexpr double DAY_TO_SEC = 24.0 * 3600.0;
constexpr double ATM_TO_PA = 101325.0;

// ==========================================================
// 1. Configuration
// ==========================================================

struct Config
{
    fs::path inputDir = "real_case_integrated_input";
    fs::path outputDir = "real_case_dirichlet_output";

    int gridSize = 180;
    int targetMonth = 47;
    int historyEndMonth = -1;
    int timeQuadPoints = 1;

    bool useDeltaQ = false;

    double p0Atm = 268.0;
    double dirichletBoundaryAtm = std::numeric_limits<double>::quiet_NaN();

    double kMd = 24.0;
    double muCp = 1.69;
    double porosity = 0.19;
    double ctPaInv = 2.236e-9;
    double hM = 10.0;
    double dtDays = 30.4375;

    int boundarySubdivisionsPerSegment = 1;
    int gmresBoundaryStride = 20;

    double fictitiousOffsetM = -1.0;
    double fictitiousOffsetFraction = 0.03;

    // Regularization coefficient for:
    // (A^T A + alpha * mean(diag(A^T A)) I) q = A^T rhs.
    double tikhonovAlpha = 1e-8;

    bool saveAllMaps = true;
};

// ==========================================================
// 2. Data structures
// ==========================================================

struct Well
{
    std::string name;
    double x = 0.0;
    double y = 0.0;
    double h = 0.0;

    // rates[month_zero] is q_model in m^3/day.
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
    double nx = 0.0;
    double ny = 0.0;
    double segmentLength = 0.0;
    int boundaryType = 0; // 0 = Dirichlet.
};

struct FictitiousSource
{
    long int sourceId = 0;
    long int boundaryId = 0;
    double x = 0.0;
    double y = 0.0;
    double offsetM = 0.0;
};

struct QuadPoint
{
    double node = 0.0;
    double weight = 0.0;
};

struct FictitiousHistoryLayer
{
    int month = 0;
    std::vector<double> qFicM3Day;

    double relativeResidual = 0.0;
    double maxBoundaryResidualPa = 0.0;
    double qL2M3Day = 0.0;
    double qMaxAbsM3Day = 0.0;
};

using DenseMatrix = std::vector<std::vector<double>>;

// ==========================================================
// 3. Generic helpers
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
    return value == "1" || value == "true" || value == "True" || value == "TRUE";
}

template <class Vector>
double l2Norm(const Vector& vector)
{
    double sum = 0.0;
    for (const double value : vector)
    {
        sum += value * value;
    }
    return std::sqrt(sum);
}

template <class Vector>
double maxAbs(const Vector& vector)
{
    double result = 0.0;
    for (const double value : vector)
    {
        result = std::max(result, std::abs(value));
    }
    return result;
}

std::vector<double> subtractVectors(
    const std::vector<double>& left,
    const std::vector<double>& right)
{
    if (left.size() != right.size())
    {
        throw std::runtime_error("subtractVectors: size mismatch.");
    }

    std::vector<double> result(left.size(), 0.0);
    for (std::size_t index = 0; index < left.size(); ++index)
    {
        result[index] = left[index] - right[index];
    }
    return result;
}

void addInPlace(
    std::vector<double>& target,
    const std::vector<double>& source)
{
    if (target.size() != source.size())
    {
        throw std::runtime_error("addInPlace: size mismatch.");
    }

    for (std::size_t index = 0; index < target.size(); ++index)
    {
        target[index] += source[index];
    }
}

std::string makeMonthString(const int month)
{
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << month;
    return stream.str();
}

// ==========================================================
// 4. Argument parsing
// ==========================================================

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
        else if (arg == "--history-end-month")
        {
            config.historyEndMonth = std::stoi(requireValue(arg));
        }
        else if (arg == "--quad-points")
        {
            config.timeQuadPoints = std::stoi(requireValue(arg));
        }
        else if (arg == "--use-delta-q")
        {
            config.useDeltaQ = parseBool(requireValue(arg));
        }
        else if (arg == "--p0-atm")
        {
            config.p0Atm = std::stod(requireValue(arg));
        }
        else if (arg == "--dirichlet-boundary-atm")
        {
            config.dirichletBoundaryAtm = std::stod(requireValue(arg));
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
        else if (arg == "--boundary-subdivisions")
        {
            config.boundarySubdivisionsPerSegment = std::stoi(requireValue(arg));
        }
        else if (arg == "--gmres-boundary-stride")
        {
            config.gmresBoundaryStride = std::stoi(requireValue(arg));
        }
        else if (arg == "--fictitious-offset-m")
        {
            config.fictitiousOffsetM = std::stod(requireValue(arg));
        }
        else if (arg == "--fictitious-offset-frac")
        {
            config.fictitiousOffsetFraction = std::stod(requireValue(arg));
        }
        else if (arg == "--tikhonov-alpha")
        {
            config.tikhonovAlpha = std::stod(requireValue(arg));
        }
        else if (arg == "--save-all-maps")
        {
            config.saveAllMaps = parseBool(requireValue(arg));
        }
        else if (arg == "--stop-after-fictitious-history-all-maps-direct"
              || arg == "--stop-after-fictitious-history-direct"
              || arg == "--gmres-max-iters"
              || arg == "--gmres-tol"
              || arg == "--gmres-verbose"
              || arg == "--tree-height"
              || arg == "--order"
              || arg == "--block-size")
        {
            // Accepted for compatibility with previous commands, ignored here.
            (void)requireValue(arg);
        }
        else
        {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (config.historyEndMonth <= 0)
    {
        config.historyEndMonth = config.targetMonth;
    }

    if (std::isnan(config.dirichletBoundaryAtm))
    {
        config.dirichletBoundaryAtm = config.p0Atm;
    }

    if (config.targetMonth < 1 || config.historyEndMonth < 1)
    {
        throw std::runtime_error("Month indices must be >= 1.");
    }

    if (config.boundarySubdivisionsPerSegment < 1)
    {
        throw std::runtime_error("--boundary-subdivisions must be >= 1.");
    }

    if (config.gmresBoundaryStride < 1)
    {
        throw std::runtime_error("--gmres-boundary-stride must be >= 1.");
    }

    return config;
}

// ==========================================================
// 5. Physics
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
        return {{0.0, 2.0}};
    }

    if (points == 2)
    {
        const double a = 1.0 / std::sqrt(3.0);
        return {{-a, 1.0}, {a, 1.0}};
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

    throw std::runtime_error("Supported --quad-points values: 1, 2, 4.");
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
// 6. Input CSV
// ==========================================================

std::vector<Well> readWells(
    const fs::path& filename,
    const int maxMonth)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw std::runtime_error("Cannot open wells file: " + filename.string());
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
        throw std::runtime_error("Cannot open rates file: " + filename.string());
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

    struct RateRow
    {
        std::string wellName;
        int monthRaw = 0;
        double qModel = 0.0;
    };

    std::vector<RateRow> rows;
    int minMonth = std::numeric_limits<int>::max();
    int maxRawMonth = std::numeric_limits<int>::min();

    while (std::getline(input, line))
    {
        if (trim(line).empty())
        {
            continue;
        }

        const auto cells = splitCsvLine(line);
        RateRow row;
        row.wellName = cells.at(nameColumn);
        row.monthRaw = std::stoi(cells.at(monthColumn));
        row.qModel = std::stod(cells.at(qColumn));

        minMonth = std::min(minMonth, row.monthRaw);
        maxRawMonth = std::max(maxRawMonth, row.monthRaw);
        rows.push_back(row);
    }

    const bool oneBasedMonths =
        !rows.empty()
        && minMonth == 1
        && maxRawMonth <= maxMonth;

    std::cout
        << "rates month indexing: min=" << minMonth
        << ", max=" << maxRawMonth
        << ", interpreted as "
        << (oneBasedMonths ? "1-based" : "0-based")
        << "\n";

    for (const auto& row : rows)
    {
        const auto iterator = wellIndex.find(row.wellName);
        if (iterator == wellIndex.end())
        {
            throw std::runtime_error("Rate exists for unknown well: " + row.wellName);
        }

        const int monthZero = oneBasedMonths ? row.monthRaw - 1 : row.monthRaw;
        if (monthZero >= 0 && monthZero < maxMonth)
        {
            wells[iterator->second].rates[monthZero] = row.qModel;
        }
    }
}

std::vector<std::array<double, 2>> readOuterContour(
    const fs::path& filename)
{
    std::ifstream input(filename);
    if (!input)
    {
        throw std::runtime_error("Cannot open contour file: " + filename.string());
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
        throw std::runtime_error("Outer contour must contain at least 3 points.");
    }

    return contour;
}

// ==========================================================
// 7. Grid and boundary geometry
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
            && (x < (xj - xi) * (y - yi) /
                    ((yj - yi) + std::numeric_limits<double>::epsilon()) + xi);

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
            minY + (maxY - minY) * static_cast<double>(iy) /
            static_cast<double>(gridSize - 1);

        for (int ix = 0; ix < gridSize; ++ix)
        {
            const double x =
                minX + (maxX - minX) * static_cast<double>(ix) /
                static_cast<double>(gridSize - 1);

            Point point;
            point.targetId = static_cast<long int>(iy) * gridSize + ix;
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

double computePolygonSignedArea(
    const std::vector<std::array<double, 2>>& contour)
{
    double area = 0.0;
    for (std::size_t i = 0; i < contour.size(); ++i)
    {
        const std::size_t j = (i + 1) % contour.size();
        area += contour[i][0] * contour[j][1] - contour[j][0] * contour[i][1];
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
    const double signedArea = computePolygonSignedArea(contour);
    const bool ccw = signedArea > 0.0;

    std::vector<BoundaryPoint> result;
    long int boundaryId = 0;

    for (std::size_t segment = 0; segment < contour.size(); ++segment)
    {
        const std::size_t next = (segment + 1) % contour.size();
        const double x0 = contour[segment][0];
        const double y0 = contour[segment][1];
        const double x1 = contour[next][0];
        const double y1 = contour[next][1];

        const double dx = x1 - x0;
        const double dy = y1 - y0;
        const double length = std::sqrt(dx * dx + dy * dy);

        if (length <= 0.0)
        {
            continue;
        }

        const double tx = dx / length;
        const double ty = dy / length;

        const double nx = ccw ? ty : -ty;
        const double ny = ccw ? -tx : tx;

        for (int local = 0; local < subdivisionsPerSegment; ++local)
        {
            const double s =
                (static_cast<double>(local) + 0.5) /
                static_cast<double>(subdivisionsPerSegment);

            BoundaryPoint point;
            point.boundaryId = boundaryId++;
            point.segmentId = static_cast<long int>(segment);
            point.x = x0 + s * dx;
            point.y = y0 + s * dy;
            point.nx = nx;
            point.ny = ny;
            point.segmentLength = length / static_cast<double>(subdivisionsPerSegment);
            point.boundaryType = 0;

            result.push_back(point);
        }
    }

    return result;
}

std::vector<FictitiousSource> buildFictitiousSources(
    const Config& config,
    const std::vector<std::array<double, 2>>& contour,
    const std::vector<BoundaryPoint>& boundaryPoints)
{
    const double boxSize = computeContourBoxSize(contour);
    const double offset =
        config.fictitiousOffsetM > 0.0
        ? config.fictitiousOffsetM
        : config.fictitiousOffsetFraction * boxSize;

    std::vector<FictitiousSource> sources;
    sources.reserve(boundaryPoints.size());

    for (const auto& point : boundaryPoints)
    {
        FictitiousSource source;
        source.sourceId = static_cast<long int>(sources.size());
        source.boundaryId = point.boundaryId;
        source.x = point.x + offset * point.nx;
        source.y = point.y + offset * point.ny;
        source.offsetM = offset;
        sources.push_back(source);
    }

    return sources;
}

std::vector<BoundaryPoint> reduceBoundaryPoints(
    const std::vector<BoundaryPoint>& points,
    const int stride)
{
    std::vector<BoundaryPoint> result;
    for (std::size_t index = 0; index < points.size(); index += static_cast<std::size_t>(stride))
    {
        result.push_back(points[index]);
    }
    return result;
}

std::vector<FictitiousSource> reduceFictitiousSources(
    const std::vector<FictitiousSource>& sources,
    const int stride)
{
    std::vector<FictitiousSource> result;
    for (std::size_t index = 0; index < sources.size(); index += static_cast<std::size_t>(stride))
    {
        FictitiousSource source = sources[index];
        source.sourceId = static_cast<long int>(result.size());
        result.push_back(source);
    }
    return result;
}

std::vector<Point> convertBoundaryPointsToTargets(
    const std::vector<BoundaryPoint>& boundaryPoints)
{
    std::vector<Point> targets;
    targets.reserve(boundaryPoints.size());

    for (const auto& boundaryPoint : boundaryPoints)
    {
        Point point;
        point.targetId = boundaryPoint.boundaryId;
        point.ix = static_cast<int>(boundaryPoint.boundaryId);
        point.iy = 0;
        point.x = boundaryPoint.x;
        point.y = boundaryPoint.y;
        point.inside = true;
        targets.push_back(point);
    }

    return targets;
}

// ==========================================================
// 8. Direct Gaussian pressure contributions
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
    const auto quadrature = getGaussLegendreRule(config.timeQuadPoints);

    const int targetMonthZero = targetMonthIndex - 1;
    const double finalTime = static_cast<double>(targetMonthIndex) * dtSeconds;

    std::vector<double> pressurePa(targets.size(), 0.0);

    for (int sourceMonthZero = 0; sourceMonthZero <= targetMonthZero; ++sourceMonthZero)
    {
        const double intervalStart = static_cast<double>(sourceMonthZero) * dtSeconds;
        const double intervalEnd = static_cast<double>(sourceMonthZero + 1) * dtSeconds;

        for (const auto& quadPoint : quadrature)
        {
            const double tau =
                0.5 * (intervalEnd - intervalStart) * quadPoint.node
                + 0.5 * (intervalEnd + intervalStart);

            const double weightTime =
                0.5 * (intervalEnd - intervalStart) * quadPoint.weight;

            const double lagSeconds = std::max(finalTime - tau, 1e-12);
            const double delta = 4.0 * kappa * lagSeconds;

            for (const auto& well : wells)
            {
                const double qModel = getSourceQ(well, sourceMonthZero, config.useDeltaQ);
                if (std::abs(qModel) < 1e-14)
                {
                    continue;
                }

                const double qM3S = qModel / DAY_TO_SEC;
                const double sourceWeight = qM3S * prefactor * weightTime / lagSeconds;

                for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex)
                {
                    if (!targets[targetIndex].inside)
                    {
                        continue;
                    }

                    const double dx = targets[targetIndex].x - well.x;
                    const double dy = targets[targetIndex].y - well.y;
                    const double rSquared = dx * dx + dy * dy;

                    pressurePa[targetIndex] += sourceWeight * std::exp(-rSquared / delta);
                }
            }
        }
    }

    return pressurePa;
}

std::vector<double> applyFictitiousIntervalToBoundary(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& qFictitiousM3Day,
    const int finalMonth,
    const int sourceMonth)
{
    if (fictitiousSources.size() != qFictitiousM3Day.size())
    {
        throw std::runtime_error("applyFictitiousIntervalToBoundary: size mismatch.");
    }

    const double kappa = computeKappa(config);
    const double prefactor = computePrefactor(config);
    const double dtSeconds = config.dtDays * DAY_TO_SEC;
    const auto quadrature = getGaussLegendreRule(config.timeQuadPoints);

    const double finalTime = static_cast<double>(finalMonth) * dtSeconds;
    const double intervalStart = static_cast<double>(sourceMonth - 1) * dtSeconds;
    const double intervalEnd = static_cast<double>(sourceMonth) * dtSeconds;

    std::vector<double> pressurePa(boundaryPoints.size(), 0.0);

    for (const auto& quadPoint : quadrature)
    {
        const double tau =
            0.5 * (intervalEnd - intervalStart) * quadPoint.node
            + 0.5 * (intervalEnd + intervalStart);

        const double weightTime =
            0.5 * (intervalEnd - intervalStart) * quadPoint.weight;

        const double lagSeconds = std::max(finalTime - tau, 1e-12);
        const double delta = 4.0 * kappa * lagSeconds;
        const double coefficientForRate = prefactor * weightTime / lagSeconds / DAY_TO_SEC;

        for (std::size_t targetIndex = 0; targetIndex < boundaryPoints.size(); ++targetIndex)
        {
            const auto& target = boundaryPoints[targetIndex];
            double sum = 0.0;

            for (std::size_t sourceIndex = 0; sourceIndex < fictitiousSources.size(); ++sourceIndex)
            {
                const auto& source = fictitiousSources[sourceIndex];
                const double dx = target.x - source.x;
                const double dy = target.y - source.y;
                const double rSquared = dx * dx + dy * dy;

                sum += qFictitiousM3Day[sourceIndex]
                    * coefficientForRate
                    * std::exp(-rSquared / delta);
            }

            pressurePa[targetIndex] += sum;
        }
    }

    return pressurePa;
}

std::vector<double> applyFictitiousIntervalToPoints(
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<double>& qFictitiousM3Day,
    const int finalMonth,
    const int sourceMonth)
{
    if (fictitiousSources.size() != qFictitiousM3Day.size())
    {
        throw std::runtime_error("applyFictitiousIntervalToPoints: size mismatch.");
    }

    const double kappa = computeKappa(config);
    const double prefactor = computePrefactor(config);
    const double dtSeconds = config.dtDays * DAY_TO_SEC;
    const auto quadrature = getGaussLegendreRule(config.timeQuadPoints);

    const double finalTime = static_cast<double>(finalMonth) * dtSeconds;
    const double intervalStart = static_cast<double>(sourceMonth - 1) * dtSeconds;
    const double intervalEnd = static_cast<double>(sourceMonth) * dtSeconds;

    std::vector<double> pressurePa(targets.size(), 0.0);

    for (const auto& quadPoint : quadrature)
    {
        const double tau =
            0.5 * (intervalEnd - intervalStart) * quadPoint.node
            + 0.5 * (intervalEnd + intervalStart);

        const double weightTime =
            0.5 * (intervalEnd - intervalStart) * quadPoint.weight;

        const double lagSeconds = std::max(finalTime - tau, 1e-12);
        const double delta = 4.0 * kappa * lagSeconds;
        const double coefficientForRate = prefactor * weightTime / lagSeconds / DAY_TO_SEC;

        for (std::size_t targetIndex = 0; targetIndex < targets.size(); ++targetIndex)
        {
            if (!targets[targetIndex].inside)
            {
                continue;
            }

            const auto& target = targets[targetIndex];
            double sum = 0.0;

            for (std::size_t sourceIndex = 0; sourceIndex < fictitiousSources.size(); ++sourceIndex)
            {
                const auto& source = fictitiousSources[sourceIndex];
                const double dx = target.x - source.x;
                const double dy = target.y - source.y;
                const double rSquared = dx * dx + dy * dy;

                sum += qFictitiousM3Day[sourceIndex]
                    * coefficientForRate
                    * std::exp(-rSquared / delta);
            }

            pressurePa[targetIndex] += sum;
        }
    }

    return pressurePa;
}

// ==========================================================
// 9. Dense matrix and Tikhonov solver
// ==========================================================

DenseMatrix buildCurrentLayerMatrix(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources)
{
    DenseMatrix matrix(
        boundaryPoints.size(),
        std::vector<double>(fictitiousSources.size(), 0.0));

    std::vector<double> basis(fictitiousSources.size(), 0.0);

    for (std::size_t column = 0; column < fictitiousSources.size(); ++column)
    {
        std::fill(basis.begin(), basis.end(), 0.0);
        basis[column] = 1.0;

        const auto values = applyFictitiousIntervalToBoundary(
            config,
            boundaryPoints,
            fictitiousSources,
            basis,
            1,
            1);

        for (std::size_t row = 0; row < boundaryPoints.size(); ++row)
        {
            matrix[row][column] = values[row];
        }
    }

    return matrix;
}

std::vector<double> multiplyDenseMatrixVector(
    const DenseMatrix& matrix,
    const std::vector<double>& vector)
{
    std::vector<double> result(matrix.size(), 0.0);

    for (std::size_t row = 0; row < matrix.size(); ++row)
    {
        double sum = 0.0;
        for (std::size_t column = 0; column < vector.size(); ++column)
        {
            sum += matrix[row][column] * vector[column];
        }
        result[row] = sum;
    }

    return result;
}

std::vector<double> solveDenseLinearSystem(
    DenseMatrix matrix,
    std::vector<double> rhs)
{
    const std::size_t n = matrix.size();
    if (n == 0 || rhs.size() != n)
    {
        throw std::runtime_error("solveDenseLinearSystem: invalid sizes.");
    }

    for (std::size_t pivot = 0; pivot < n; ++pivot)
    {
        std::size_t best = pivot;
        for (std::size_t row = pivot + 1; row < n; ++row)
        {
            if (std::abs(matrix[row][pivot]) > std::abs(matrix[best][pivot]))
            {
                best = row;
            }
        }

        if (std::abs(matrix[best][pivot]) < 1e-300)
        {
            throw std::runtime_error("solveDenseLinearSystem: singular matrix.");
        }

        if (best != pivot)
        {
            std::swap(matrix[pivot], matrix[best]);
            std::swap(rhs[pivot], rhs[best]);
        }

        const double pivotValue = matrix[pivot][pivot];
        for (std::size_t column = pivot; column < n; ++column)
        {
            matrix[pivot][column] /= pivotValue;
        }
        rhs[pivot] /= pivotValue;

        for (std::size_t row = 0; row < n; ++row)
        {
            if (row == pivot)
            {
                continue;
            }

            const double factor = matrix[row][pivot];
            if (std::abs(factor) < 1e-300)
            {
                continue;
            }

            for (std::size_t column = pivot; column < n; ++column)
            {
                matrix[row][column] -= factor * matrix[pivot][column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    return rhs;
}

std::vector<double> solveTikhonovLeastSquares(
    const DenseMatrix& matrix,
    const std::vector<double>& rhs,
    const double alpha)
{
    if (matrix.empty())
    {
        throw std::runtime_error("solveTikhonovLeastSquares: empty matrix.");
    }

    const std::size_t rowCount = matrix.size();
    const std::size_t columnCount = matrix[0].size();

    DenseMatrix normalMatrix(
        columnCount,
        std::vector<double>(columnCount, 0.0));

    std::vector<double> normalRhs(columnCount, 0.0);

    for (std::size_t row = 0; row < rowCount; ++row)
    {
        for (std::size_t i = 0; i < columnCount; ++i)
        {
            normalRhs[i] += matrix[row][i] * rhs[row];

            for (std::size_t j = 0; j < columnCount; ++j)
            {
                normalMatrix[i][j] += matrix[row][i] * matrix[row][j];
            }
        }
    }

    double meanDiagonal = 0.0;
    for (std::size_t i = 0; i < columnCount; ++i)
    {
        meanDiagonal += normalMatrix[i][i];
    }
    meanDiagonal /= static_cast<double>(columnCount);

    const double regularization = alpha * std::max(meanDiagonal, 1e-300);
    for (std::size_t i = 0; i < columnCount; ++i)
    {
        normalMatrix[i][i] += regularization;
    }

    return solveDenseLinearSystem(normalMatrix, normalRhs);
}

// ==========================================================
// 10. Fictitious-source history
// ==========================================================

std::vector<double> buildDirichletRhs(
    const Config& config,
    const std::vector<double>& realBoundaryPa,
    const std::vector<double>& oldFictitiousBoundaryPa)
{
    if (realBoundaryPa.size() != oldFictitiousBoundaryPa.size())
    {
        throw std::runtime_error("buildDirichletRhs: size mismatch.");
    }

    const double boundaryDeltaPa =
        (config.dirichletBoundaryAtm - config.p0Atm) * ATM_TO_PA;

    std::vector<double> rhs(realBoundaryPa.size(), 0.0);
    for (std::size_t index = 0; index < rhs.size(); ++index)
    {
        rhs[index] = boundaryDeltaPa - realBoundaryPa[index] - oldFictitiousBoundaryPa[index];
    }
    return rhs;
}

std::vector<double> computeOldFictitiousBoundaryContribution(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history,
    const int finalMonth)
{
    std::vector<double> result(boundaryPoints.size(), 0.0);

    for (const auto& layer : history)
    {
        if (layer.month >= finalMonth)
        {
            continue;
        }

        const auto contribution = applyFictitiousIntervalToBoundary(
            config,
            boundaryPoints,
            fictitiousSources,
            layer.qFicM3Day,
            finalMonth,
            layer.month);

        addInPlace(result, contribution);
    }

    return result;
}

std::vector<FictitiousHistoryLayer> solveFictitiousHistory(
    const Config& config,
    const std::vector<BoundaryPoint>& boundaryPoints,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<Well>& wells)
{
    const auto boundaryTargets = convertBoundaryPointsToTargets(boundaryPoints);

    std::cout
        << "Building current-layer dense matrix: "
        << boundaryPoints.size() << " x " << fictitiousSources.size() << "\n";

    const auto matrixStart = std::chrono::high_resolution_clock::now();
    const DenseMatrix currentLayerMatrix = buildCurrentLayerMatrix(
        config,
        boundaryPoints,
        fictitiousSources);
    const auto matrixEnd = std::chrono::high_resolution_clock::now();

    std::cout
        << "Matrix built in "
        << std::chrono::duration<double>(matrixEnd - matrixStart).count()
        << " sec\n";

    std::vector<FictitiousHistoryLayer> history;
    history.reserve(static_cast<std::size_t>(config.historyEndMonth));

    for (int month = 1; month <= config.historyEndMonth; ++month)
    {
        const auto monthStart = std::chrono::high_resolution_clock::now();

        const auto realBoundaryPa = computeDirectMonth(
            config,
            wells,
            boundaryTargets,
            month);

        const auto oldFictitiousBoundaryPa = computeOldFictitiousBoundaryContribution(
            config,
            boundaryPoints,
            fictitiousSources,
            history,
            month);

        const auto rhsPa = buildDirichletRhs(
            config,
            realBoundaryPa,
            oldFictitiousBoundaryPa);

        const auto solution = solveTikhonovLeastSquares(
            currentLayerMatrix,
            rhsPa,
            config.tikhonovAlpha);

        const auto approximationPa = multiplyDenseMatrixVector(
            currentLayerMatrix,
            solution);

        const auto residualPa = subtractVectors(rhsPa, approximationPa);

        FictitiousHistoryLayer layer;
        layer.month = month;
        layer.qFicM3Day = solution;
        layer.maxBoundaryResidualPa = maxAbs(residualPa);
        layer.relativeResidual = l2Norm(residualPa) / std::max(l2Norm(rhsPa), 1e-300);
        layer.qL2M3Day = l2Norm(solution);
        layer.qMaxAbsM3Day = maxAbs(solution);

        history.push_back(layer);

        const auto monthEnd = std::chrono::high_resolution_clock::now();
        const double monthTime = std::chrono::duration<double>(monthEnd - monthStart).count();

        std::cout
            << "month " << month << " / " << config.historyEndMonth
            << ": rel residual = " << layer.relativeResidual
            << ", max boundary residual = " << layer.maxBoundaryResidualPa / ATM_TO_PA << " atm"
            << ", max |q_fic| = " << layer.qMaxAbsM3Day << " m3/day"
            << ", time = " << monthTime << " sec\n";
    }

    return history;
}

// ==========================================================
// 11. Map dynamics
// ==========================================================

std::vector<double> computeFictitiousHistoryOnTargets(
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history,
    const int finalMonth)
{
    std::vector<double> result(targets.size(), 0.0);

    for (const auto& layer : history)
    {
        if (layer.month > finalMonth)
        {
            continue;
        }

        const auto contribution = applyFictitiousIntervalToPoints(
            config,
            targets,
            fictitiousSources,
            layer.qFicM3Day,
            finalMonth,
            layer.month);

        addInPlace(result, contribution);
    }

    return result;
}

void savePressureMap(
    const fs::path& filename,
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<double>& realDeltaPa,
    const std::vector<double>& fictitiousDeltaPa,
    const int month)
{
    if (targets.size() != realDeltaPa.size() || targets.size() != fictitiousDeltaPa.size())
    {
        throw std::runtime_error("savePressureMap: size mismatch.");
    }

    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("Cannot write map: " + filename.string());
    }

    output << std::setprecision(17);
    output
        << "target_id,ix,iy,x,y,inside,month,"
        << "real_delta_pa,fictitious_delta_pa,total_delta_pa,"
        << "real_delta_atm,fictitious_delta_atm,total_delta_atm,"
        << "real_pressure_atm,total_pressure_atm\n";

    for (std::size_t index = 0; index < targets.size(); ++index)
    {
        const double realPa = realDeltaPa[index];
        const double ficPa = fictitiousDeltaPa[index];
        const double totalPa = realPa + ficPa;

        output
            << targets[index].targetId << ","
            << targets[index].ix << ","
            << targets[index].iy << ","
            << targets[index].x << ","
            << targets[index].y << ","
            << static_cast<int>(targets[index].inside) << ","
            << month << ","
            << realPa << ","
            << ficPa << ","
            << totalPa << ","
            << realPa / ATM_TO_PA << ","
            << ficPa / ATM_TO_PA << ","
            << totalPa / ATM_TO_PA << ","
            << config.p0Atm + realPa / ATM_TO_PA << ","
            << config.p0Atm + totalPa / ATM_TO_PA << "\n";
    }
}

void saveAllPressureMaps(
    const Config& config,
    const std::vector<Point>& targets,
    const std::vector<Well>& wells,
    const std::vector<FictitiousSource>& fictitiousSources,
    const std::vector<FictitiousHistoryLayer>& history)
{
    const fs::path mapsDir = config.outputDir / "dirichlet_history_pressure_maps_direct";
    fs::create_directories(mapsDir);

    std::ofstream indexOutput(mapsDir / "index.csv");
    indexOutput << "month,file\n";

    for (int month = 1; month <= config.historyEndMonth; ++month)
    {
        const auto mapStart = std::chrono::high_resolution_clock::now();

        const auto realDeltaPa = computeDirectMonth(config, wells, targets, month);
        const auto fictitiousDeltaPa = computeFictitiousHistoryOnTargets(
            config,
            targets,
            fictitiousSources,
            history,
            month);

        const std::string filename =
            "dirichlet_history_pressure_map_month_" + makeMonthString(month) + ".csv";

        savePressureMap(
            mapsDir / filename,
            config,
            targets,
            realDeltaPa,
            fictitiousDeltaPa,
            month);

        indexOutput << month << "," << filename << "\n";

        const auto mapEnd = std::chrono::high_resolution_clock::now();
        std::cout
            << "saved map " << month << " / " << config.historyEndMonth
            << ": " << filename
            << ", time = "
            << std::chrono::duration<double>(mapEnd - mapStart).count()
            << " sec\n";
    }
}

// ==========================================================
// 12. Output helpers
// ==========================================================

void saveBoundaryPoints(
    const fs::path& filename,
    const std::vector<BoundaryPoint>& boundaryPoints)
{
    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("Cannot write boundary points: " + filename.string());
    }

    output << std::setprecision(17);
    output << "boundary_id,segment_id,x,y,nx,ny,segment_length,boundary_type\n";
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
        throw std::runtime_error("Cannot write fictitious sources: " + filename.string());
    }

    output << std::setprecision(17);
    output << "source_id,boundary_id,x,y,offset_m\n";
    for (const auto& source : sources)
    {
        output
            << source.sourceId << ","
            << source.boundaryId << ","
            << source.x << ","
            << source.y << ","
            << source.offsetM << "\n";
    }
}

void saveFictitiousHistory(
    const fs::path& filename,
    const std::vector<FictitiousSource>& sources,
    const std::vector<FictitiousHistoryLayer>& history)
{
    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("Cannot write fictitious history: " + filename.string());
    }

    output << std::setprecision(17);
    output << "month,source_id,boundary_id,x,y,offset_m,q_fic_m3_day\n";

    for (const auto& layer : history)
    {
        for (std::size_t index = 0; index < sources.size(); ++index)
        {
            output
                << layer.month << ","
                << sources[index].sourceId << ","
                << sources[index].boundaryId << ","
                << sources[index].x << ","
                << sources[index].y << ","
                << sources[index].offsetM << ","
                << layer.qFicM3Day[index] << "\n";
        }
    }
}

void saveFictitiousHistorySummary(
    const fs::path& filename,
    const std::vector<FictitiousHistoryLayer>& history)
{
    std::ofstream output(filename);
    if (!output)
    {
        throw std::runtime_error("Cannot write fictitious history summary: " + filename.string());
    }

    output << std::setprecision(17);
    output
        << "month,relative_residual,max_boundary_residual_pa,"
        << "max_boundary_residual_atm,q_l2_m3_day,q_max_abs_m3_day\n";

    for (const auto& layer : history)
    {
        output
            << layer.month << ","
            << layer.relativeResidual << ","
            << layer.maxBoundaryResidualPa << ","
            << layer.maxBoundaryResidualPa / ATM_TO_PA << ","
            << layer.qL2M3Day << ","
            << layer.qMaxAbsM3Day << "\n";
    }
}

// ==========================================================
// 13. Main pipeline
// ==========================================================

int main(int argc, char** argv)
{
    try
    {
        const Config config = parseArgs(argc, argv);
        fs::create_directories(config.outputDir);

        const fs::path wellsFile = config.inputDir / "real_wells.csv";
        const fs::path ratesFile = config.inputDir / "well_rates_clean.csv";
        const fs::path contourFile = config.inputDir / "contour.csv";

        std::cout
            << "Clean transient Dirichlet fictitious-source prototype\n"
            << "inputDir = " << config.inputDir << "\n"
            << "outputDir = " << config.outputDir << "\n"
            << "gridSize = " << config.gridSize << "\n"
            << "historyEndMonth = " << config.historyEndMonth << "\n"
            << "timeQuadPoints = " << config.timeQuadPoints << "\n"
            << "useDeltaQ = " << config.useDeltaQ << "\n"
            << "p0 = " << config.p0Atm << " atm\n"
            << "Dirichlet boundary = " << config.dirichletBoundaryAtm << " atm\n"
            << "Tikhonov alpha = " << config.tikhonovAlpha << "\n";

        const double kappa = computeKappa(config);
        std::cout
            << "kappa = " << kappa << " m^2/s\n"
            << "sqrt(delta 1 month) = "
            << std::sqrt(4.0 * kappa * config.dtDays * DAY_TO_SEC)
            << " m\n\n";

        auto wells = readWells(wellsFile, config.historyEndMonth);
        readRates(ratesFile, wells, config.historyEndMonth);

        const auto contour = readOuterContour(contourFile);
        const auto targets = buildGrid(contour, config.gridSize);

        const auto fullBoundaryPoints = buildBoundaryCollocationPoints(
            contour,
            config.boundarySubdivisionsPerSegment);

        const auto fullFictitiousSources = buildFictitiousSources(
            config,
            contour,
            fullBoundaryPoints);

        const auto boundaryPoints = reduceBoundaryPoints(
            fullBoundaryPoints,
            config.gmresBoundaryStride);

        const auto fictitiousSources = reduceFictitiousSources(
            fullFictitiousSources,
            config.gmresBoundaryStride);

        const auto insideGridCount = std::count_if(
            targets.begin(),
            targets.end(),
            [](const Point& point) { return point.inside; });

        std::cout
            << "wells = " << wells.size() << "\n"
            << "grid targets = " << targets.size()
            << ", inside = " << insideGridCount << "\n"
            << "full boundary points = " << fullBoundaryPoints.size() << "\n"
            << "reduced boundary points = " << boundaryPoints.size() << "\n"
            << "fictitious sources per month = " << fictitiousSources.size() << "\n"
            << "history unknown count = "
            << static_cast<long long>(fictitiousSources.size()) * config.historyEndMonth
            << "\n\n";

        saveBoundaryPoints(config.outputDir / "dirichlet_boundary_points_full.csv", fullBoundaryPoints);
        saveBoundaryPoints(config.outputDir / "dirichlet_boundary_points_used.csv", boundaryPoints);
        saveFictitiousSources(config.outputDir / "dirichlet_fictitious_sources_used.csv", fictitiousSources);

        const auto historyStart = std::chrono::high_resolution_clock::now();
        const auto history = solveFictitiousHistory(
            config,
            boundaryPoints,
            fictitiousSources,
            wells);
        const auto historyEnd = std::chrono::high_resolution_clock::now();

        std::cout
            << "\nHistory solved in "
            << std::chrono::duration<double>(historyEnd - historyStart).count()
            << " sec\n";

        saveFictitiousHistory(
            config.outputDir / "dirichlet_fictitious_history_direct.csv",
            fictitiousSources,
            history);

        saveFictitiousHistorySummary(
            config.outputDir / "dirichlet_fictitious_history_summary_direct.csv",
            history);

        if (config.saveAllMaps)
        {
            saveAllPressureMaps(
                config,
                targets,
                wells,
                fictitiousSources,
                history);
        }

        std::cout << "\nDone.\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "ERROR: " << error.what() << "\n";
        return 1;
    }
}
