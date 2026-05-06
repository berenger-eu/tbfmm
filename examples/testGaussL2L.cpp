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
RealType binomial(const int n, const int k)
{
    if (k < 0 || k > n)
    {
        return RealType(0);
    }

    return factorial<RealType>(n)
         / (factorial<RealType>(k) * factorial<RealType>(n - k));
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

    const long int treeHeight = 3;
    const long int parentLevel = 1;

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

    FakeCellIndex parentIndex{
        SpaceIndexType(configuration).getIndexFromBoxPos({0, 0}),
        {0, 0}};

    constexpr long int nbChildren = 4;
    long int childrenPos[nbChildren] = {0, 1, 2, 3};

    std::array<RealType, ExpansionSize> parentLocal{};
    std::array<std::array<RealType, ExpansionSize>, nbChildren> childLocal{};
    std::array<std::array<RealType, ExpansionSize>, nbChildren> reference{};

    for (int ax = 0; ax < OrderSize; ++ax)
    {
        for (int ay = 0; ay < OrderSize; ++ay)
        {
            parentLocal[expansionIndex<P>(ax, ay)] =
                RealType(0.15) * RealType(ax + 1)
                - RealType(0.04) * RealType(ay + 1)
                + RealType(0.02) * RealType((ax + 1) * (ay + 1));
        }
    }

    kernel.L2L(
        parentIndex,
        parentLevel,
        parentLocal,
        childLocal,
        childrenPos,
        nbChildren);

    const RealType invSqrtDelta = RealType(1) / std::sqrt(delta);

    const auto boxCorner = configuration.getBoxCorner();

    const RealType parentWidth =
        boxWidths[0] * std::pow(RealType(0.5), RealType(parentLevel));

    const std::array<RealType, Dim> parentCenter{
        boxCorner[0] + (RealType(parentIndex.boxCoord[0]) + RealType(0.5)) * parentWidth,
        boxCorner[1] + (RealType(parentIndex.boxCoord[1]) + RealType(0.5)) * parentWidth};

    const long int childLevel = parentLevel + 1;

    const RealType childWidth =
        boxWidths[0] * std::pow(RealType(0.5), RealType(childLevel));

    for (long int child = 0; child < nbChildren; ++child)
    {
        const auto childIndex =
            SpaceIndexType(configuration).getChildIndexFromParent(
                parentIndex.spaceIndex,
                childrenPos[child]);

        const std::array<long int, Dim> childCoord =
            SpaceIndexType(configuration).getBoxPosFromIndex(childIndex);

        const std::array<RealType, Dim> childCenter{
            boxCorner[0] + (RealType(childCoord[0]) + RealType(0.5)) * childWidth,
            boxCorner[1] + (RealType(childCoord[1]) + RealType(0.5)) * childWidth};

        const RealType dx =
            (childCenter[0] - parentCenter[0]) * invSqrtDelta;

        const RealType dy =
            (childCenter[1] - parentCenter[1]) * invSqrtDelta;

        RealType powersX[OrderSize];
        RealType powersY[OrderSize];

        powersX[0] = RealType(1);
        powersY[0] = RealType(1);

        for (int k = 1; k < OrderSize; ++k)
        {
            powersX[k] = powersX[k - 1] * dx;
            powersY[k] = powersY[k - 1] * dy;
        }

        for (int bx = 0; bx < OrderSize; ++bx)
        {
            for (int by = 0; by < OrderSize; ++by)
            {
                RealType sum = RealType(0);

                for (int ax = bx; ax < OrderSize; ++ax)
                {
                    for (int ay = by; ay < OrderSize; ++ay)
                    {
                        sum +=
                            parentLocal[expansionIndex<P>(ax, ay)]
                            * binomial<RealType>(ax, bx)
                            * binomial<RealType>(ay, by)
                            * powersX[ax - bx]
                            * powersY[ay - by];
                    }
                }

                reference[child][expansionIndex<P>(bx, by)] = sum;
            }
        }
    }

    RealType maxAbsError = RealType(0);

    std::cout << "Gauss L2L test\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n";
    std::cout << "P = " << P << "\n\n";

    for (long int child = 0; child < nbChildren; ++child)
    {
        std::cout << "child " << child << "\n";

        for (int bx = 0; bx < OrderSize; ++bx)
        {
            for (int by = 0; by < OrderSize; ++by)
            {
                const int idx = expansionIndex<P>(bx, by);

                const RealType error =
                    std::abs(childLocal[child][idx] - reference[child][idx]);

                maxAbsError = std::max(maxAbsError, error);

                std::cout
                    << "L(" << bx << "," << by << ")"
                    << " | computed = " << childLocal[child][idx]
                    << " | reference = " << reference[child][idx]
                    << " | error = " << error
                    << "\n";
            }
        }

        std::cout << "\n";
    }

    std::cout << "max abs error = " << maxAbsError << "\n";

    const RealType tolerance = 1e-14;

    if (maxAbsError < tolerance)
    {
        std::cout << "TEST PASSED\n";
        return 0;
    }

    std::cout << "TEST FAILED\n";
    return 1;
}