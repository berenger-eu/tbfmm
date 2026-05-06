#ifndef FP2PGAUSS_HPP
#define FP2PGAUSS_HPP

#include <cmath>

/**
 * @brief Direct P2P routines for the 2D Gaussian kernel.
 *
 * Kernel:
 *      K(x, y) = coefficient * exp(-|x - y|^2 / delta)
 *
 * Expected particle data layout:
 *      inParticles[0] -> x coordinate
 *      inParticles[1] -> y coordinate
 *      inParticles[2] -> physical value / source weight q
 *
 * Expected RHS layout:
 *      inParticlesRhs[0] -> accumulated potential
 */
namespace FP2PGauss
{
    template <class FRealContainer>
    auto GetPtr(FRealContainer& ptr)
    {
        return &ptr[0];
    }

    template <class FRealContainer>
    auto GetPtr(const FRealContainer& ptr)
    {
        return &ptr[0];
    }

    /**
     * @brief Evaluate Gaussian kernel between one source and one target.
     */
    template <class FReal>
    inline FReal EvaluateKernel(
        const FReal sourceX,
        const FReal sourceY,
        const FReal targetX,
        const FReal targetY,
        const FReal delta,
        const FReal coefficient)
    {
        const FReal dx = targetX - sourceX;
        const FReal dy = targetY - sourceY;
        const FReal r2 = dx * dx + dy * dy;

        return coefficient * std::exp(-r2 / delta);
    }

    /**
     * @brief Direct interactions inside one leaf.
     *
     * This function skips i == j.
     *
     * It computes all pairs:
     *      i < j
     *
     * and updates both particles because the Gaussian kernel is symmetric.
     */
    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void GenericInnerScalar(
        const ParticlesClassValues& inTargets,
        ParticlesClassRhs& inTargetsRhs,
        const long int nbParticlesTargets,
        const FReal delta,
        const FReal coefficient)
    {
        const FReal* const targetsX = GetPtr(inTargets[0]);
        const FReal* const targetsY = GetPtr(inTargets[1]);
        const FReal* const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal* const targetsPotentials = GetPtr(inTargetsRhs[0]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const FReal targetX = targetsX[idxTarget];
            const FReal targetY = targetsY[idxTarget];
            const FReal targetPhysicalValue = targetsPhysicalValues[idxTarget];

            for (long int idxSource = idxTarget + 1; idxSource < nbParticlesTargets; ++idxSource)
            {
                const FReal kernelValue = EvaluateKernel(
                    targetsX[idxSource],
                    targetsY[idxSource],
                    targetX,
                    targetY,
                    delta,
                    coefficient);

                targetsPotentials[idxTarget] += kernelValue * targetsPhysicalValues[idxSource];
                targetsPotentials[idxSource] += kernelValue * targetPhysicalValue;
            }
        }
    }

    /**
     * @brief Direct interactions between two different leaves, mutual version.
     *
     * Updates both:
     *      targets receive sources contribution
     *      sources receive targets contribution
     */
    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void FullMutualScalar(
        const ParticlesClassValues& inSources,
        ParticlesClassRhs& inSourcesRhs,
        const long int nbParticlesSources,
        const ParticlesClassValues& inTargets,
        ParticlesClassRhs& inTargetsRhs,
        const long int nbParticlesTargets,
        const FReal delta,
        const FReal coefficient)
    {
        const FReal* const targetsX = GetPtr(inTargets[0]);
        const FReal* const targetsY = GetPtr(inTargets[1]);
        const FReal* const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal* const targetsPotentials = GetPtr(inTargetsRhs[0]);

        const FReal* const sourcesX = GetPtr(inSources[0]);
        const FReal* const sourcesY = GetPtr(inSources[1]);
        const FReal* const sourcesPhysicalValues = GetPtr(inSources[2]);
        FReal* const sourcesPotentials = GetPtr(inSourcesRhs[0]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const FReal targetX = targetsX[idxTarget];
            const FReal targetY = targetsY[idxTarget];
            const FReal targetPhysicalValue = targetsPhysicalValues[idxTarget];

            FReal targetPotentialIncrement = FReal(0);

            for (long int idxSource = 0; idxSource < nbParticlesSources; ++idxSource)
            {
                const FReal kernelValue = EvaluateKernel(
                    sourcesX[idxSource],
                    sourcesY[idxSource],
                    targetX,
                    targetY,
                    delta,
                    coefficient);

                targetPotentialIncrement += kernelValue * sourcesPhysicalValues[idxSource];
                sourcesPotentials[idxSource] += kernelValue * targetPhysicalValue;
            }

            targetsPotentials[idxTarget] += targetPotentialIncrement;
        }
    }

    /**
     * @brief Direct interactions between source leaf and target leaf, non-mutual version.
     *
     * Updates only target particles.
     *
     * This is useful for task-based / parallel algorithms where writing into
     * source RHS may be unsafe or undesired.
     */
    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void GenericFullRemoteScalar(
        const ParticlesClassValues& inSources,
        const long int nbParticlesSources,
        const ParticlesClassValues& inTargets,
        ParticlesClassRhs& inTargetsRhs,
        const long int nbParticlesTargets,
        const FReal delta,
        const FReal coefficient)
    {
        const FReal* const targetsX = GetPtr(inTargets[0]);
        const FReal* const targetsY = GetPtr(inTargets[1]);
        FReal* const targetsPotentials = GetPtr(inTargetsRhs[0]);

        const FReal* const sourcesX = GetPtr(inSources[0]);
        const FReal* const sourcesY = GetPtr(inSources[1]);
        const FReal* const sourcesPhysicalValues = GetPtr(inSources[2]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const FReal targetX = targetsX[idxTarget];
            const FReal targetY = targetsY[idxTarget];

            FReal targetPotentialIncrement = FReal(0);

            for (long int idxSource = 0; idxSource < nbParticlesSources; ++idxSource)
            {
                const FReal kernelValue = EvaluateKernel(
                    sourcesX[idxSource],
                    sourcesY[idxSource],
                    targetX,
                    targetY,
                    delta,
                    coefficient);

                targetPotentialIncrement += kernelValue * sourcesPhysicalValues[idxSource];
            }

            targetsPotentials[idxTarget] += targetPotentialIncrement;
        }
    }

#define GenericInner GenericInnerScalar
#define GenericFullRemote GenericFullRemoteScalar
#define FullMutual FullMutualScalar

} // namespace FP2PGauss

#endif // FP2PGAUSS_HPP