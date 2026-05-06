#include "kernels/P2P/FP2PGauss.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>

template <class RealType>
RealType referenceKernel(
    const RealType sourceX,
    const RealType sourceY,
    const RealType targetX,
    const RealType targetY,
    const RealType delta,
    const RealType coefficient)
{
    const RealType dx = targetX - sourceX;
    const RealType dy = targetY - sourceY;
    const RealType r2 = dx * dx + dy * dy;

    return coefficient * std::exp(-r2 / delta);
}

template <class RealType>
bool testGenericInner()
{
    constexpr long int nbParticles = 5;

    const RealType delta = 0.1;
    const RealType coefficient = 1.0;
    const RealType tolerance = 1e-14;

    std::array<std::vector<RealType>, 3> particles;
    std::array<std::vector<RealType>, 1> particlesRhs;
    std::array<std::vector<RealType>, 1> referenceRhs;

    for (auto& values : particles)
    {
        values.resize(nbParticles);
    }

    particlesRhs[0].resize(nbParticles, RealType(0));
    referenceRhs[0].resize(nbParticles, RealType(0));

    particles[0] = {0.1, 0.3, 0.5, 0.7, 0.9};
    particles[1] = {0.2, 0.8, 0.4, 0.6, 0.1};
    particles[2] = {1.0, 2.0, -1.0, 0.5, 3.0};

    FP2PGauss::template GenericInner<RealType>(
        particles,
        particlesRhs,
        nbParticles,
        delta,
        coefficient);

    for (long int i = 0; i < nbParticles; ++i)
    {
        for (long int j = i + 1; j < nbParticles; ++j)
        {
            const RealType kernelValue = referenceKernel(
                particles[0][j],
                particles[1][j],
                particles[0][i],
                particles[1][i],
                delta,
                coefficient);

            referenceRhs[0][i] += kernelValue * particles[2][j];
            referenceRhs[0][j] += kernelValue * particles[2][i];
        }
    }

    RealType maxAbsError = 0;

    std::cout << "Gauss P2P GenericInner test\n";

    for (long int i = 0; i < nbParticles; ++i)
    {
        const RealType error =
            std::abs(particlesRhs[0][i] - referenceRhs[0][i]);

        maxAbsError = std::max(maxAbsError, error);

        std::cout
            << "particle " << i
            << " | computed = " << particlesRhs[0][i]
            << " | reference = " << referenceRhs[0][i]
            << " | error = " << error
            << "\n";
    }

    std::cout << "max abs error = " << maxAbsError << "\n";

    const bool passed = maxAbsError < tolerance;
    std::cout << (passed ? "GenericInner PASSED\n\n" : "GenericInner FAILED\n\n");

    return passed;
}

template <class RealType>
bool testFullMutual()
{
    constexpr long int nbSources = 3;
    constexpr long int nbTargets = 4;

    const RealType delta = 0.1;
    const RealType coefficient = 1.0;
    const RealType tolerance = 1e-14;

    std::array<std::vector<RealType>, 3> sources;
    std::array<std::vector<RealType>, 3> targets;

    std::array<std::vector<RealType>, 1> sourcesRhs;
    std::array<std::vector<RealType>, 1> targetsRhs;

    std::array<std::vector<RealType>, 1> referenceSourcesRhs;
    std::array<std::vector<RealType>, 1> referenceTargetsRhs;

    for (auto& values : sources)
    {
        values.resize(nbSources);
    }

    for (auto& values : targets)
    {
        values.resize(nbTargets);
    }

    sourcesRhs[0].resize(nbSources, RealType(0));
    targetsRhs[0].resize(nbTargets, RealType(0));

    referenceSourcesRhs[0].resize(nbSources, RealType(0));
    referenceTargetsRhs[0].resize(nbTargets, RealType(0));

    sources[0] = {0.1, 0.2, 0.4};
    sources[1] = {0.1, 0.7, 0.3};
    sources[2] = {1.0, -2.0, 0.5};

    targets[0] = {0.6, 0.8, 0.9, 0.15};
    targets[1] = {0.2, 0.5, 0.9, 0.85};
    targets[2] = {3.0, 1.0, -1.0, 2.0};

    FP2PGauss::template FullMutual<RealType>(
        sources,
        sourcesRhs,
        nbSources,
        targets,
        targetsRhs,
        nbTargets,
        delta,
        coefficient);

    for (long int i = 0; i < nbTargets; ++i)
    {
        for (long int j = 0; j < nbSources; ++j)
        {
            const RealType kernelValue = referenceKernel(
                sources[0][j],
                sources[1][j],
                targets[0][i],
                targets[1][i],
                delta,
                coefficient);

            referenceTargetsRhs[0][i] += kernelValue * sources[2][j];
            referenceSourcesRhs[0][j] += kernelValue * targets[2][i];
        }
    }

    RealType maxAbsError = 0;

    std::cout << "Gauss P2P FullMutual test\n";

    for (long int i = 0; i < nbTargets; ++i)
    {
        const RealType error =
            std::abs(targetsRhs[0][i] - referenceTargetsRhs[0][i]);

        maxAbsError = std::max(maxAbsError, error);

        std::cout
            << "target " << i
            << " | computed = " << targetsRhs[0][i]
            << " | reference = " << referenceTargetsRhs[0][i]
            << " | error = " << error
            << "\n";
    }

    for (long int j = 0; j < nbSources; ++j)
    {
        const RealType error =
            std::abs(sourcesRhs[0][j] - referenceSourcesRhs[0][j]);

        maxAbsError = std::max(maxAbsError, error);

        std::cout
            << "source " << j
            << " | computed = " << sourcesRhs[0][j]
            << " | reference = " << referenceSourcesRhs[0][j]
            << " | error = " << error
            << "\n";
    }

    std::cout << "max abs error = " << maxAbsError << "\n";

    const bool passed = maxAbsError < tolerance;
    std::cout << (passed ? "FullMutual PASSED\n\n" : "FullMutual FAILED\n\n");

    return passed;
}

template <class RealType>
bool testGenericFullRemote()
{
    constexpr long int nbSources = 3;
    constexpr long int nbTargets = 4;

    const RealType delta = 0.1;
    const RealType coefficient = 1.0;
    const RealType tolerance = 1e-14;

    std::array<std::vector<RealType>, 3> sources;
    std::array<std::vector<RealType>, 3> targets;

    std::array<std::vector<RealType>, 1> targetsRhs;
    std::array<std::vector<RealType>, 1> referenceTargetsRhs;

    for (auto& values : sources)
    {
        values.resize(nbSources);
    }

    for (auto& values : targets)
    {
        values.resize(nbTargets);
    }

    targetsRhs[0].resize(nbTargets, RealType(0));
    referenceTargetsRhs[0].resize(nbTargets, RealType(0));

    sources[0] = {0.1, 0.2, 0.4};
    sources[1] = {0.1, 0.7, 0.3};
    sources[2] = {1.0, -2.0, 0.5};

    targets[0] = {0.6, 0.8, 0.9, 0.15};
    targets[1] = {0.2, 0.5, 0.9, 0.85};
    targets[2] = {3.0, 1.0, -1.0, 2.0};

    FP2PGauss::template GenericFullRemote<RealType>(
        sources,
        nbSources,
        targets,
        targetsRhs,
        nbTargets,
        delta,
        coefficient);

    for (long int i = 0; i < nbTargets; ++i)
    {
        for (long int j = 0; j < nbSources; ++j)
        {
            const RealType kernelValue = referenceKernel(
                sources[0][j],
                sources[1][j],
                targets[0][i],
                targets[1][i],
                delta,
                coefficient);

            referenceTargetsRhs[0][i] += kernelValue * sources[2][j];
        }
    }

    RealType maxAbsError = 0;

    std::cout << "Gauss P2P GenericFullRemote test\n";

    for (long int i = 0; i < nbTargets; ++i)
    {
        const RealType error =
            std::abs(targetsRhs[0][i] - referenceTargetsRhs[0][i]);

        maxAbsError = std::max(maxAbsError, error);

        std::cout
            << "target " << i
            << " | computed = " << targetsRhs[0][i]
            << " | reference = " << referenceTargetsRhs[0][i]
            << " | error = " << error
            << "\n";
    }

    std::cout << "max abs error = " << maxAbsError << "\n";

    const bool passed = maxAbsError < tolerance;
    std::cout << (passed ? "GenericFullRemote PASSED\n\n" : "GenericFullRemote FAILED\n\n");

    return passed;
}

int main()
{
    using RealType = double;

    bool ok = true;

    ok = testGenericInner<RealType>() && ok;
    ok = testFullMutual<RealType>() && ok;
    ok = testGenericFullRemote<RealType>() && ok;
    if (ok)
    {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }

    std::cout << "TEST FAILED\n";
    return 1;
}