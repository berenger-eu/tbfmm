#include "kernels/gausskernel/tbfgausskernel.hpp"
#include "kernels/P2P/FP2PGauss.hpp"

#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

template <int P>
constexpr int expansionIndex(const int ax, const int ay)
{
    return ax * (P + 1) + ay;
}

int main()
{
    using RealType = double;

    constexpr int Dim = 2;
    constexpr int P = 12;
    constexpr int OrderSize = P + 1;
    constexpr int ExpansionSize = OrderSize * OrderSize;

    const RealType delta = 0.05;
    const RealType coefficient = 1.0;

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

    /*
     * Source cell B = (0,0), target cell C = (3,3) at level 2.
     * They are well-separated.
     */
    FakeCellIndex sourceIndex{
        SpaceIndexType(configuration).getIndexFromBoxPos({0, 0}),
        {0, 0}};

    FakeCellIndex targetIndex{
        SpaceIndexType(configuration).getIndexFromBoxPos({3, 3}),
        {3, 3}};

    constexpr long int nbSources = 4;
    constexpr long int nbTargets = 4;

    std::array<std::vector<RealType>, 3> sources;
    std::array<std::vector<RealType>, 3> targets;

    std::array<std::vector<RealType>, 1> fmmRhs;
    std::array<std::vector<RealType>, 1> directRhs;

    for (auto& values : sources)
    {
        values.resize(nbSources);
    }

    for (auto& values : targets)
    {
        values.resize(nbTargets);
    }

    fmmRhs[0].resize(nbTargets, RealType(0));
    directRhs[0].resize(nbTargets, RealType(0));

    /*
     * Source points inside cell (0,0), roughly [0,0.25] x [0,0.25]
     */
    sources[0] = {0.05, 0.10, 0.15, 0.20};
    sources[1] = {0.06, 0.18, 0.12, 0.22};
    sources[2] = {1.0, -0.7, 0.5, 1.3};

    /*
     * Target points inside cell (3,3), roughly [0.75,1] x [0.75,1]
     */
    targets[0] = {0.80, 0.88, 0.92, 0.97};
    targets[1] = {0.82, 0.90, 0.78, 0.95};
    targets[2] = {0.0, 0.0, 0.0, 0.0};

    std::array<RealType, ExpansionSize> sourceMultipole{};
    std::array<RealType, ExpansionSize> targetLocal{};

    const long int* indexes = nullptr;

    kernel.P2M(
        sourceIndex,
        indexes,
        sources,
        nbSources,
        sourceMultipole);

    /*
     * M2L expects interaction-list encoding.
     * relative position = sourceCoord - targetCoord = (0,0) - (3,3) = (-3,-3)
     */
    constexpr long int nbNeighbors = 1;

    const std::array<long int, Dim> relativePos{{-3, -3}};

    long int neighPos[nbNeighbors] = {
        SpaceIndexType(configuration).getInteractionIndexFromRelativePos(
            relativePos)
    };

    std::array<std::array<RealType, ExpansionSize>, nbNeighbors> interactingCells{};
    interactingCells[0] = sourceMultipole;

    kernel.M2L(
        targetIndex,
        level,
        interactingCells,
        neighPos,
        nbNeighbors,
        targetLocal);

    kernel.L2P(
        targetIndex,
        targetLocal,
        indexes,
        targets,
        fmmRhs,
        nbTargets);

    FP2PGauss::template GenericFullRemote<RealType>(
        sources,
        nbSources,
        targets,
        directRhs,
        nbTargets,
        delta,
        coefficient);

    RealType maxAbsError = RealType(0);
    RealType maxRelError = RealType(0);

    std::cout << "Gauss P2M -> M2L -> L2P test\n";
    std::cout << "delta = " << delta << "\n";
    std::cout << "coefficient = " << coefficient << "\n";
    std::cout << "P = " << P << "\n\n";

    for (long int i = 0; i < nbTargets; ++i)
    {
        const RealType absError =
            std::abs(fmmRhs[0][i] - directRhs[0][i]);

        const RealType relError =
            absError / std::max(std::abs(directRhs[0][i]), RealType(1e-30));

        maxAbsError = std::max(maxAbsError, absError);
        maxRelError = std::max(maxRelError, relError);

        std::cout
            << "target " << i
            << " | fmm = " << fmmRhs[0][i]
            << " | direct = " << directRhs[0][i]
            << " | abs error = " << absError
            << " | rel error = " << relError
            << "\n";
    }

    std::cout << "\nmax abs error = " << maxAbsError << "\n";
    std::cout << "max rel error = " << maxRelError << "\n";

    return 0;
}