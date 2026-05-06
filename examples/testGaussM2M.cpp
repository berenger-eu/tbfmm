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

    const long int treeHeight = 3;

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
     * Parent at level 1, coordinate (0,0).
     * Its children at level 2 have local positions 0,1,2,3.
     */
    FakeCellIndex parentIndex{
        SpaceIndexType(configuration).getIndexFromBoxPos({0, 0}),
        {0, 0}};

    constexpr long int nbChildren = 4;
    long int childrenPos[nbChildren] = {0, 1, 2, 3};

    std::array<std::array<RealType, ExpansionSize>, nbChildren> childMultipoles{};
    std::array<RealType, ExpansionSize> parentMultipole{};
    std::array<RealType, ExpansionSize> reference{};

    /**
     * Fill child multipoles with deterministic nontrivial values.
     */
    for (long int child = 0; child < nbChildren; ++child)
    {
        for (int ax = 0; ax < OrderSize; ++ax)
        {
            for (int ay = 0; ay < OrderSize; ++ay)
            {
                const int idx = expansionIndex<P>(ax, ay);

                childMultipoles[child][idx] =
                    RealType(0.1)
                    * RealType(child + 1)
                    * RealType(ax + 1)
                    - RealType(0.03)
                    * RealType(ay + 1);
            }
        }
    }

    kernel.M2M(
        parentIndex,
        1,
        childMultipoles,
        parentMultipole,
        childrenPos,
        nbChildren);

    const RealType invSqrtDelta = RealType(1) / std::sqrt(delta);

    const auto boxCorner = configuration.getBoxCorner();

    const RealType parentWidth = boxWidths[0] * std::pow(RealType(0.5), RealType(1));
    const std::array<RealType, Dim> parentCenter{
        boxCorner[0] + RealType(0.5) * parentWidth,
        boxCorner[1] + RealType(0.5) * parentWidth};

    const RealType childWidth = boxWidths[0] * std::pow(RealType(0.5), RealType(2));

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

        const RealType dx = (childCenter[0] - parentCenter[0]) * invSqrtDelta;
        const RealType dy = (childCenter[1] - parentCenter[1]) * invSqrtDelta;

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
                RealType accumulator = RealType(0);

                for (int bx = 0; bx <= ax; ++bx)
                {
                    for (int by = 0; by <= ay; ++by)
                    {
                        accumulator +=
                            childMultipoles[child][expansionIndex<P>(bx, by)]
                            * powersX[ax - bx]
                            * powersY[ay - by]
                            / (factorial<RealType>(ax - bx)
                               * factorial<RealType>(ay - by));
                    }
                }

                reference[expansionIndex<P>(ax, ay)] += accumulator;
            }
        }
    }

    RealType maxAbsError = RealType(0);

    std::cout << "Gauss M2M test\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n";
    std::cout << "P = " << P << "\n\n";

    for (int ax = 0; ax < OrderSize; ++ax)
    {
        for (int ay = 0; ay < OrderSize; ++ay)
        {
            const int idx = expansionIndex<P>(ax, ay);

            const RealType error =
                std::abs(parentMultipole[idx] - reference[idx]);

            maxAbsError = std::max(maxAbsError, error);

            std::cout
                << "M(" << ax << "," << ay << ")"
                << " | computed = " << parentMultipole[idx]
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