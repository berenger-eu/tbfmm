#include "kernels/gausskernel/tbfgausskernel.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

template <class RealType>
RealType factorial(const int n)
{
    RealType result = RealType(1);

    for (int i = 2; i <= n; ++i)
    {
        result *= RealType(i);
    }

    return result;
}

template <int P>
constexpr int expansionIndex(const int ax, const int ay)
{
    return ax * (P + 1) + ay;
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

    /**
     * Leaf coordinate.
     *
     * treeHeight = 2 means leaf level has 2^(treeHeight-1) = 2 boxes
     * per direction in many TBFMM configurations.
     *
     * We use boxCoord = {0, 0}; for domain [0,1]^2 its center is (0.25,0.25)
     * if leaf width is 0.5.
     */
    struct FakeLeafIndex
    {
        std::array<long int, Dim> boxCoord;
    };

    FakeLeafIndex leafIndex{{0, 0}};

    constexpr long int nbParticles = 3;

    std::array<std::vector<RealType>, 3> particles;

    for (auto& values : particles)
    {
        values.resize(nbParticles);
    }

    particles[0] = {0.10, 0.20, 0.35};   // x
    particles[1] = {0.15, 0.40, 0.20};   // y
    particles[2] = {1.0, -2.0, 0.5};     // q

    std::array<RealType, ExpansionSize> multipole{};
    std::array<RealType, ExpansionSize> reference{};

    const long int* particleIndexes = nullptr;

    kernel.P2M(
        leafIndex,
        particleIndexes,
        particles,
        nbParticles,
        multipole);

    const auto leafWidths = configuration.getLeafWidths();
    const auto boxCorner = configuration.getBoxCorner();

    const std::array<RealType, Dim> leafCenter{
        boxCorner[0] + RealType(0.5) * leafWidths[0],
        boxCorner[1] + RealType(0.5) * leafWidths[1]};

    const RealType invSqrtDelta = RealType(1) / std::sqrt(delta);

    for (long int idxPart = 0; idxPart < nbParticles; ++idxPart)
    {
        const RealType q = coefficient * particles[2][idxPart];

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

        for (int ax = 0; ax < OrderSize; ++ax)
        {
            for (int ay = 0; ay < OrderSize; ++ay)
            {
                reference[expansionIndex<P>(ax, ay)] +=
                    q
                    * powersX[ax]
                    * powersY[ay]
                    / (factorial<RealType>(ax) * factorial<RealType>(ay));
            }
        }
    }

    RealType maxAbsError = RealType(0);

    std::cout << "Gauss P2M test\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n";
    std::cout << "P = " << P << "\n\n";

    for (int ax = 0; ax < OrderSize; ++ax)
    {
        for (int ay = 0; ay < OrderSize; ++ay)
        {
            const int idx = expansionIndex<P>(ax, ay);

            const RealType error =
                std::abs(multipole[idx] - reference[idx]);

            maxAbsError = std::max(maxAbsError, error);

            std::cout
                << "M(" << ax << "," << ay << ")"
                << " | computed = " << multipole[idx]
                << " | reference = " << reference[idx]
                << " | error = " << error
                << "\n";
        }
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