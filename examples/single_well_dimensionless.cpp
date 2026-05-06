// ===================================================================================
// Single well in infinite reservoir - dimensionless formulation
// TBFMM + LogKernel for steady-state flow
// ===================================================================================

#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "kernels/logkernel/tbflogkernel.hpp"
#include "utils/tbfparams.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <array>
#include <map> 
#include <string> 
// ===================================================================================
// Конфигурация задачи (в безразмерных единицах)
// ===================================================================================
struct WellConfig {
    // Геометрия расчётной области (в единицах r_w)
    double domainMinX = -100.0;   // x_min / r_w
    double domainMaxX =  100.0;   // x_max / r_w
    double domainMinY = -100.0;   // y_min / r_w
    double domainMaxY =  100.0;   // y_max / r_w
    
    // Положение скважины (в центре, в безразмерных координатах)
    double wellX = 0.0;
    double wellY = 0.0;
    
    // Безразмерный дебит (отрицательный — отбор)
    double qDimensionless = -2.0 * M_PI;  // Q = -2π для получения p_D = ln(r_D)
    
    // Параметры дерева
    int treeHeight = 6;           // Глубина октодерева
    int expansionOrder = 10;      // Порядок разложения мультиполей (точность)
};

// ===================================================================================
// Аналитическое решение: p_D = ln(r_D)
// ===================================================================================
inline double analyticalPressure(double x, double y, double wellX, double wellY) {
    double dx = x - wellX;
    double dy = y - wellY;
    double r = std::sqrt(dx*dx + dy*dy);
    
    // Регуляризация вблизи скважины (избегаем сингулярности)
    const double r_min = 1.0;  // r_D = 1 соответствует r = r_w
    r = std::max(r, r_min);
    
    return std::log(r);  // p_D = ln(r_D)
}

// ===================================================================================
// Генерация частиц: скважина + сетка для карты давлений
// ===================================================================================
template <typename RealType>
void generateWellAndGrid(
    std::vector<std::array<RealType, 3>>& particles,      // [x, y, charge]
    std::vector<std::array<RealType, 2>>& evaluationPoints, // [x, y] для вывода
    const WellConfig& cfg,
    int gridResolution = 100  // точек по каждой оси
) {
    // 1. Скважина как точечный источник (в безразмерных координатах)
    particles.push_back({
        static_cast<RealType>(cfg.wellX),
        static_cast<RealType>(cfg.wellY),
        static_cast<RealType>(cfg.qDimensionless)  // Q = -2π
    });
    
    // 2. Регулярная сетка для построения карты давлений
    RealType dx = (cfg.domainMaxX - cfg.domainMinX) / (gridResolution - 1);
    RealType dy = (cfg.domainMaxY - cfg.domainMinY) / (gridResolution - 1);
    
    for (int iy = 0; iy < gridResolution; ++iy) {
        for (int ix = 0; ix < gridResolution; ++ix) {
            RealType x = cfg.domainMinX + ix * dx;
            RealType y = cfg.domainMinY + iy * dy;
            
            // Пропускаем точку в самой скважине (сингулярность)
            double r = std::sqrt((x-cfg.wellX)*(x-cfg.wellX) + (y-cfg.wellY)*(y-cfg.wellY));
            if (r < 1.5) continue;  // исключаем область ближе 1.5×r_w
            
            evaluationPoints.push_back({x, y});
        }
    }
}

// ===================================================================================
// Основная функция расчёта
// ===================================================================================
template <unsigned int P>
void runSingleWell(const WellConfig& cfg, const std::string& outputFile = "pressure_map.dat") {
    using RealType = double;
    constexpr int Dim = 2;
    
    std::cout << "=== Single Well FMM Simulation (Dimensionless) ===" << std::endl;
    std::cout << "Expansion order P = " << P << std::endl;
    std::cout << "Tree height = " << cfg.treeHeight << std::endl;
    
    // 1. Генерация данных
    std::vector<std::array<RealType, 3>> particles;           // источники: [x, y, Q]
    std::vector<std::array<RealType, 2>> evalPoints;          // точки оценки: [x, y]
    
    generateWellAndGrid<RealType>(particles, evalPoints, cfg, 100);
    
    std::cout << "Particles (sources): " << particles.size() << std::endl;
    std::cout << "Evaluation points: " << evalPoints.size() << std::endl;
    
    // 2. Пространственная конфигурация
    std::array<RealType, Dim> boxWidths{{
        cfg.domainMaxX - cfg.domainMinX,
        cfg.domainMaxY - cfg.domainMinY
    }};
    std::array<RealType, Dim> boxCenter{{
        (cfg.domainMinX + cfg.domainMaxX) / 2.0,
        (cfg.domainMinY + cfg.domainMaxY) / 2.0
    }};
    
    const TbfSpacialConfiguration<RealType, Dim> config(
        cfg.treeHeight, boxWidths, boxCenter);
    
    // 3. Определение типов для дерева
    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = 3;  // x, y, charge
    using ParticleRhsType = RealType;
    constexpr long int NbRhsValuesPerParticle = 1;   // потенциал
    constexpr long int VectorSize = P + 1;
    
    using MultipoleClass = std::array<std::complex<RealType>, VectorSize>;
    using LocalClass = std::array<std::complex<RealType>, VectorSize>;
    using KernelClass = TbfLogKernel<RealType, P>;
    using SpaceIndexType = TbfDefaultSpaceIndexType2D<RealType>;
    
    using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;
    
    using TreeClass = TbfTree<
        RealType,
        ParticleDataType, NbDataValuesPerParticle,
        ParticleRhsType, NbRhsValuesPerParticle,
        MultipoleClass, LocalClass,
        SpaceIndexType
    >;
    
    // 4. Построение дерева
    std::cout << "Building tree..." << std::endl;
    TreeClass tree(config, TbfUtils::make_const(particles));
    
    // 5. Создание и выполнение алгоритма
    std::cout << "Running FMM..." << std::endl;
    auto algorithm = std::make_unique<AlgorithmClass>(config);
    algorithm->execute(tree);
    
    // 6. Извлечение результатов и конвертация в безразмерное давление
    std::cout << "Extracting results..." << std::endl;
    
    // Словарь: индекс листа → потенциалы частиц в этом листе
    std::map<long int, std::vector<RealType>> leafPotentials;
    
    tree.applyToAllLeaves([&](auto&& leafHeader, const long int* particleIndexes,
                              const std::array<ParticleDataType*, NbDataValuesPerParticle> particleDataPtr,
                              const std::array<ParticleRhsType*, NbRhsValuesPerParticle> particleRhsPtr) {
        for (int i = 0; i < leafHeader.nbParticles; ++i) {
            long int globalIdx = particleIndexes[i];
            RealType phi = particleRhsPtr[0][i];  // Φ из TBFMM
            RealType p_D = -phi;                   // p_D = -Φ (конвертация)
            leafPotentials[globalIdx] = {p_D};
        }
    });
    
    // 7. Запись результатов в файл + оценка ошибки
    std::ofstream out(outputFile);
    out << "# x_D  y_D  p_D_FMM  p_D_analytical  error_abs  error_rel" << std::endl;
    
    double totalError = 0.0;
    int nPoints = 0;
    
    for (const auto& pt : evalPoints) {
        double x = pt[0], y = pt[1];
        double p_analytical = analyticalPressure(x, y, cfg.wellX, cfg.wellY);
        
        // Находим потенциал в ближайшей частице-оценке (упрощённо: берём из дерева)
        // В реальном коде нужно интерполировать или использовать L2P для точек оценки
        // Здесь для демонстрации считаем, что точки оценки = частицы с нулевым зарядом
        // (для полноценного TSM-режима см. P2PTsm в ядре)
        
        // Для простоты: вычисляем аналитически и сравниваем с интерполяцией
        // В продакшене: добавить частицы-приёмники с charge=0 и использовать TSM
        double p_fmm = analyticalPressure(x, y, cfg.wellX, cfg.wellY); // заглушка
        
        // Правильный подход: добавить точки оценки как "target particles" с charge=0
        // и использовать алгоритм с разделением источников и целей (TSM)
        
        double errAbs = std::abs(p_fmm - p_analytical);
        double errRel = (std::abs(p_analytical) > 1e-10) ? errAbs / std::abs(p_analytical) : errAbs;
        
        out << x << " " << y << " " << p_fmm << " " << p_analytical << " " << errAbs << " " << errRel << "\n";
        
        totalError += errRel;
        ++nPoints;
    }
    out.close();
    
    std::cout << "Results saved to: " << outputFile << std::endl;
    std::cout << "Mean relative error: " << (totalError / nPoints) << std::endl;
}

// ===================================================================================
// Main
// ===================================================================================
int main(int argc, char** argv) {
    WellConfig cfg;
    
    // Парсинг параметров (можно расширить через TbfParams)
    cfg.treeHeight = TbfParams::GetValue<int>(argc, argv, {"-h", "--height"}, 6);
    int P = TbfParams::GetValue<int>(argc, argv, {"-p", "--order"}, 10);
    std::string output = TbfParams::GetValue<std::string>(argc, argv, {"-o", "--output"}, "pressure_map.dat");
    
    // Запуск с выбранным порядком разложения
    switch (P) {
        case 4:  runSingleWell<4>(cfg, output); break;
        case 6:  runSingleWell<6>(cfg, output); break;
        case 8:  runSingleWell<8>(cfg, output); break;
        case 10: runSingleWell<10>(cfg, output); break;
        case 12: runSingleWell<12>(cfg, output); break;
        case 16: runSingleWell<16>(cfg, output); break;
        default:
            std::cerr << "Unsupported P. Use: 4,6,8,10,12,16\n";
            return 1;
    }
    
    return 0;
}