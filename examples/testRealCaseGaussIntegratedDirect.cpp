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

namespace fs = std::filesystem;

constexpr double PI = 3.14159265358979323846;

// ==========================================================
// ПАРАМЕТРЫ ПО УМОЛЧАНИЮ
// ==========================================================

struct Config
{
    fs::path inputDir = "real_case_integrated_input";
    fs::path outputDir = "real_case_integrated_output";

    int gridSize = 180;
    int maxMonth = 47;

    // Начинаем с одной эффективной точки времени на месяц.
    // Позже можно проверить 2 и 4 отдельно.
    int timeQuadPoints = 1;

    // false — использовать текущий месячный дебит q_curr.
    // true  — использовать изменение дебита q_curr - q_prev.
    bool useDeltaQ = false;

    // Сохраняем карту только для итогового месяца.
    int saveMapMonth = 47;

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
        else if (arg == "--output-dir")
        {
            config.outputDir = requireValue(arg);
        }
        else if (arg == "--grid-size")
        {
            config.gridSize = std::stoi(requireValue(arg));
        }
        else if (arg == "--max-month")
        {
            config.maxMonth = std::stoi(requireValue(arg));
        }
        else if (arg == "--quad-points")
        {
            config.timeQuadPoints = std::stoi(requireValue(arg));
        }
        else if (arg == "--use-delta-q")
        {
            config.useDeltaQ = parseBool(requireValue(arg));
        }
        else if (arg == "--save-map-month")
        {
            config.saveMapMonth = std::stoi(requireValue(arg));
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
// MAIN
// ==========================================================

int main(int argc, char** argv)
{
    try
    {
        const Config config = parseArgs(argc, argv);

        fs::create_directories(config.outputDir);

        const fs::path wellsFile =
            config.inputDir / "real_wells.csv";

        const fs::path ratesFile =
            config.inputDir / "well_rates_clean.csv";

        const fs::path contourFile =
            config.inputDir / "contour.csv";

        std::cout << "Integrated real-case direct benchmark\n";
        std::cout << "inputDir = " << config.inputDir << "\n";
        std::cout << "outputDir = " << config.outputDir << "\n";
        std::cout << "gridSize = " << config.gridSize << "\n";
        std::cout << "maxMonth = " << config.maxMonth << "\n";
        std::cout << "timeQuadPoints = " << config.timeQuadPoints << "\n";
        std::cout << "useDeltaQ = " << config.useDeltaQ << "\n";

        const double kappa = computeKappa(config);

        std::cout << "kappa = " << kappa << " m^2/s\n";
        std::cout
            << "sqrt(delta 1 month) = "
            << std::sqrt(
                4.0 * kappa * config.dtDays * DAY_TO_SEC)
            << " m\n\n";

        auto wells =
            readWells(wellsFile, config.maxMonth);

        readRates(ratesFile, wells, config.maxMonth);

        const auto contour =
            readOuterContour(contourFile);

        const auto targets =
            buildGrid(contour, config.gridSize);

        const auto insideCount =
            std::count_if(
                targets.begin(),
                targets.end(),
                [](const Point& point)
                {
                    return point.inside;
                });

        std::cout << "wells = " << wells.size() << "\n";
        std::cout << "targets = " << targets.size() << "\n";
        std::cout << "inside targets = " << insideCount << "\n\n";

        const fs::path timingFile =
            config.outputDir / "integrated_direct_timing.csv";

        std::ofstream timingOutput(timingFile);

        if (!timingOutput)
        {
            throw std::runtime_error(
                "Cannot write timing file: "
                + timingFile.string());
        }

        timingOutput << std::setprecision(17);

        timingOutput
            << "month_index,active_wells,quad_points,"
            << "direct_time_sec,min_pressure_atm,max_pressure_atm\n";

        for (int monthIndex = 1;
             monthIndex <= config.maxMonth;
             ++monthIndex)
        {
            std::cout
                << "Month "
                << monthIndex
                << " / "
                << config.maxMonth
                << " ... "
                << std::flush;

            const auto start =
                std::chrono::steady_clock::now();

            const auto pressureDeltaPa =
                computeDirectMonth(
                    config,
                    wells,
                    targets,
                    monthIndex);

            const auto end =
                std::chrono::steady_clock::now();

            const double elapsedSeconds =
                std::chrono::duration<double>(
                    end - start).count();

            double minPressureAtm =
                std::numeric_limits<double>::max();

            double maxPressureAtm =
                std::numeric_limits<double>::lowest();

            for (std::size_t index = 0;
                 index < targets.size();
                 ++index)
            {
                if (!targets[index].inside)
                {
                    continue;
                }

                const double pressureAtm =
                    config.p0Atm
                    + pressureDeltaPa[index] / ATM_TO_PA;

                minPressureAtm =
                    std::min(minPressureAtm, pressureAtm);

                maxPressureAtm =
                    std::max(maxPressureAtm, pressureAtm);
            }

            const int activeWells =
                countCurrentActiveWells(
                    wells,
                    monthIndex - 1);

            timingOutput
                << monthIndex << ","
                << activeWells << ","
                << config.timeQuadPoints << ","
                << elapsedSeconds << ","
                << minPressureAtm << ","
                << maxPressureAtm << "\n";

            std::cout
                << elapsedSeconds
                << " sec"
                << " | active wells = "
                << activeWells
                << "\n";

            if (monthIndex == config.saveMapMonth)
            {
                const fs::path mapFile =
                    config.outputDir
                    / (
                        "direct_map_month_"
                        + std::to_string(monthIndex)
                        + ".csv"
                    );

                saveMap(
                    mapFile,
                    config,
                    targets,
                    pressureDeltaPa,
                    monthIndex);

                std::cout
                    << "Saved map: "
                    << mapFile
                    << "\n";
            }
        }

        std::cout
            << "\nSaved timing: "
            << timingFile
            << "\n";

        std::cout << "Done.\n";
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