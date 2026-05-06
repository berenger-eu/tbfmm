#include "kernels/gausskernel/tbfgausskernel.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <array>
#include <cmath>
#include <iostream>
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

template <class RealType>
void computeHermiteValues(
    const RealType x,
    RealType* const h,
    const int size)
{
    h[0] = std::exp(-x * x);

    if (size <= 1)
    {
        return;
    }

    h[1] = RealType(2) * x * h[0];

    for (int n = 1; n < size - 1; ++n)
    {
        h[n + 1] =
            RealType(2) * x * h[n]
            - RealType(2 * n) * h[n - 1];
    }
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
    constexpr int HermiteSize = 2 * P + 1;

    const RealType delta = 0.1;
    const RealType coefficient = 2.0;

    const long int treeHeight = 3;
    const long int level = 2;

    const std::array<RealType, Dim> boxWidths{{1.0, 1.0}};
    const std::array<RealType, Dim> boxCenter{{0.5, 0.5}};

    using SpaceIndexType = TbfDefaultSpaceIndexType2D<RealType>;
    using KernelClass = TbfGaussKernel<RealType, P, SpaceIndexType>;

    const TbfSpacialConfiguration<RealType, Dim> configuration(
        treeHeight,
        boxWidths,
        boxCenter);

    KernelClass kernel(configuration, delta, coefficient);

    struct FakeCellIndex
    {
        typename SpaceIndexType::IndexType spaceIndex;
        std::array<long int, Dim> boxCoord;
    };

    /**
     * Target cell C at level 2 with coordinate (1,1).
     * Source interaction cell B is represented through relative position.
     */
    FakeCellIndex targetIndex{
        SpaceIndexType(configuration).getIndexFromBoxPos({1, 1}),
        {1, 1}};

    constexpr long int nbNeighbors = 1;

    /*
     * We choose sourceCoord = targetCoord + relativePos.
     * relativePos = (2, 0) means source box coordinate is (3, 1).
     *
     * Instead of relying on library interaction-index encoding in the test,
     * we will use the same encoding function indirectly only if available.
     *
     * For this test, neighPos must be a valid interaction index.
     * In this library, getRelativePosFromInteractionIndex is used inside M2L,
     * so we need to provide an index that maps to relative position (2,0).
     *
     * If this fails in your build, we will replace this test by direct wrapper
     * or inspect the space-index encoding.
     */
    const std::array<long int, Dim> wantedRelativePos{{2, 0}};

    long int neighPos[nbNeighbors] = {
        SpaceIndexType(configuration).getInteractionIndexFromRelativePos(
            wantedRelativePos)
    };

    std::array<std::array<RealType, ExpansionSize>, nbNeighbors> sourceMultipoles{};
    std::array<RealType, ExpansionSize> local{};
    std::array<RealType, ExpansionSize> reference{};

    for (int ax = 0; ax < OrderSize; ++ax)
    {
        for (int ay = 0; ay < OrderSize; ++ay)
        {
            const int idx = expansionIndex<P>(ax, ay);

            sourceMultipoles[0][idx] =
                RealType(0.2) * RealType(ax + 1)
                - RealType(0.07) * RealType(ay + 1)
                + RealType(0.03) * RealType((ax + 1) * (ay + 1));
        }
    }

    kernel.M2L(
        targetIndex,
        level,
        sourceMultipoles,
        neighPos,
        nbNeighbors,
        local);

    const RealType invSqrtDelta = RealType(1) / std::sqrt(delta);

    const auto boxCorner = configuration.getBoxCorner();
    const RealType widthAtLevel =
        boxWidths[0] * std::pow(RealType(0.5), RealType(level));

    const std::array<RealType, Dim> targetCenter{
        boxCorner[0] + (RealType(targetIndex.boxCoord[0]) + RealType(0.5)) * widthAtLevel,
        boxCorner[1] + (RealType(targetIndex.boxCoord[1]) + RealType(0.5)) * widthAtLevel};

    const std::array<long int, Dim> sourceCoord{
        targetIndex.boxCoord[0] + wantedRelativePos[0],
        targetIndex.boxCoord[1] + wantedRelativePos[1]};

    const std::array<RealType, Dim> sourceCenter{
        boxCorner[0] + (RealType(sourceCoord[0]) + RealType(0.5)) * widthAtLevel,
        boxCorner[1] + (RealType(sourceCoord[1]) + RealType(0.5)) * widthAtLevel};

    const RealType dx =
        (sourceCenter[0] - targetCenter[0]) * invSqrtDelta;

    const RealType dy =
        (sourceCenter[1] - targetCenter[1]) * invSqrtDelta;

    RealType hermiteX[HermiteSize];
    RealType hermiteY[HermiteSize];

    computeHermiteValues(dx, hermiteX, HermiteSize);
    computeHermiteValues(dy, hermiteY, HermiteSize);

    for (int bx = 0; bx < OrderSize; ++bx)
    {
        for (int by = 0; by < OrderSize; ++by)
        {
            RealType sum = RealType(0);

            for (int ax = 0; ax < OrderSize; ++ax)
            {
                for (int ay = 0; ay < OrderSize; ++ay)
                {
                    sum +=
                        sourceMultipoles[0][expansionIndex<P>(ax, ay)]
                        * hermiteX[ax + bx]
                        * hermiteY[ay + by];
                }
            }

            if ((bx + by) % 2 != 0)
            {
                sum = -sum;
            }

            sum /= factorial<RealType>(bx) * factorial<RealType>(by);

            reference[expansionIndex<P>(bx, by)] += sum;
        }
    }

    RealType maxAbsError = RealType(0);

    std::cout << "Gauss M2L test\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n";
    std::cout << "P = " << P << "\n\n";

    for (int bx = 0; bx < OrderSize; ++bx)
    {
        for (int by = 0; by < OrderSize; ++by)
        {
            const int idx = expansionIndex<P>(bx, by);

            const RealType error =
                std::abs(local[idx] - reference[idx]);

            maxAbsError = std::max(maxAbsError, error);

            std::cout
                << "L(" << bx << "," << by << ")"
                << " | computed = " << local[idx]
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