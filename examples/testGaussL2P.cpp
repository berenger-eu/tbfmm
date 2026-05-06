#include "kernels/gausskernel/tbfgausskernel.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

template <int P>
constexpr int expansionIndex(const int bx, const int by)
{
    return bx * (P + 1) + by;
}

int main()
{
    using RealType = double;

    constexpr int Dim = 2;
    constexpr int P = 4;
    constexpr int OrderSize = P + 1;
    constexpr int ExpansionSize = OrderSize * OrderSize;

    const RealType delta = 0.1;
    const RealType coefficient = 2.0;

    const long int treeHeight = 2;

    const std::array<RealType, Dim> boxWidths{{1.0, 1.0}};
    const std::array<RealType, Dim> boxCenter{{0.5, 0.5}};

    using SpaceIndexType = TbfDefaultSpaceIndexType2D<RealType>;
    using KernelClass = TbfGaussKernel<RealType, P, SpaceIndexType>;

    const TbfSpacialConfiguration<RealType, Dim> configuration(
        treeHeight,
        boxWidths,
        boxCenter);

    KernelClass kernel(configuration, delta, coefficient);

    struct FakeLeafIndex
    {
        std::array<long int, Dim> boxCoord;
    };

    FakeLeafIndex leafIndex{{0, 0}};

    std::array<RealType, ExpansionSize> local{};

    for (int bx = 0; bx < OrderSize; ++bx)
    {
        for (int by = 0; by < OrderSize; ++by)
        {
            local[expansionIndex<P>(bx, by)] =
                RealType(0.13) * RealType(bx + 1)
                - RealType(0.05) * RealType(by + 1)
                + RealType(0.02) * RealType((bx + 1) * (by + 1));
        }
    }

    constexpr long int nbParticles = 4;

    std::array<std::vector<RealType>, 3> particles;
    std::array<std::vector<RealType>, 1> particlesRhs;
    std::array<std::vector<RealType>, 1> referenceRhs;

    for (auto& values : particles)
    {
        values.resize(nbParticles);
    }

    particlesRhs[0].resize(nbParticles, RealType(0));
    referenceRhs[0].resize(nbParticles, RealType(0));

    particles[0] = {0.10, 0.20, 0.35, 0.45};
    particles[1] = {0.15, 0.40, 0.20, 0.35};
    particles[2] = {1.0, -2.0, 0.5, 3.0}; // not used by L2P

    const long int* particleIndexes = nullptr;

    kernel.L2P(
        leafIndex,
        local,
        particleIndexes,
        particles,
        particlesRhs,
        nbParticles);

    const auto leafWidths = configuration.getLeafWidths();
    const auto boxCorner = configuration.getBoxCorner();

    const std::array<RealType, Dim> leafCenter{
        boxCorner[0] + RealType(0.5) * leafWidths[0],
        boxCorner[1] + RealType(0.5) * leafWidths[1]};

    const RealType invSqrtDelta = RealType(1) / std::sqrt(delta);

    for (long int idxPart = 0; idxPart < nbParticles; ++idxPart)
    {
        const RealType dx =
            (particles[0][idxPart] - leafCenter[0]) * invSqrtDelta;

        const RealType dy =
            (particles[1][idxPart] - leafCenter[1]) * invSqrtDelta;

        RealType powersX[OrderSize];
        RealType powersY[OrderSize];

        powersX[0] = RealType(1);
        powersY[0] = RealType(1);

        for (int k = 1; k < OrderSize; ++k)
        {
            powersX[k] = powersX[k - 1] * dx;
            powersY[k] = powersY[k - 1] * dy;
        }

        RealType sum = RealType(0);

        for (int bx = 0; bx < OrderSize; ++bx)
        {
            for (int by = 0; by < OrderSize; ++by)
            {
                sum +=
                    local[expansionIndex<P>(bx, by)]
                    * powersX[bx]
                    * powersY[by];
            }
        }

        referenceRhs[0][idxPart] += sum;
    }

    RealType maxAbsError = RealType(0);

    std::cout << "Gauss L2P test\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n";
    std::cout << "P = " << P << "\n\n";

    for (long int idxPart = 0; idxPart < nbParticles; ++idxPart)
    {
        const RealType error =
            std::abs(particlesRhs[0][idxPart] - referenceRhs[0][idxPart]);

        maxAbsError = std::max(maxAbsError, error);

        std::cout
            << "particle " << idxPart
            << " | computed = " << particlesRhs[0][idxPart]
            << " | reference = " << referenceRhs[0][idxPart]
            << " | error = " << error
            << "\n";
    }

    std::cout << "\nmax abs error = " << maxAbsError << "\n";

    const RealType tolerance = 1e-14;

    if (maxAbsError < tolerance)
    {
        std::cout << "TEST PASSED\n";
        return 0;
    }

    std::cout << "TEST FAILED\n";
    return 1;
}