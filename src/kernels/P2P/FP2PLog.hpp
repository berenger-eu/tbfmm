// ===================================================================================
// Copyright SCALFmm 2011 INRIA, Olivier Coulaud, Bérenger Bramas, Matthias Messner
// olivier.coulaud@inria.fr, berenger.bramas@inria.fr
// This software is a computer program whose purpose is to compute the FMM.
//
// This software is governed by the CeCILL-C and LGPL licenses and
// abiding by the rules of distribution of free software.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public and CeCILL-C Licenses for more details.
// "http://www.cecill.info".
// "http://www.gnu.org/licenses".
// ===================================================================================
#ifndef FP2PLog_HPP
#define FP2PLog_HPP
#include <complex>
// #include "kernels/unifkernel/FMath.hpp"

#ifdef TBF_USE_INASTEMP
#include "InastempGlobal.h"
#endif

/**
 * @brief The FP2PLog namespace
 */
namespace FP2PLog
{

    template <class FRealContainer>
    auto GetPtr(FRealContainer &ptr)
    {
        return &ptr[0];
    }

    template <class FReal>
    inline void MutualParticles(const FReal sourceX, const FReal sourceY, const FReal sourceZ, const FReal sourcePhysicalValue,
                                FReal *sourceForceX, FReal *sourceForceY, FReal *sourceForceZ, FReal *sourcePotential,
                                const FReal targetX, const FReal targetY, const FReal targetZ, const FReal targetPhysicalValue,
                                FReal *targetForceX, FReal *targetForceY, FReal *targetForceZ, FReal *targetPotential)
    {
        std::complex<FReal> target_point{targetX, targetY};
        std::complex<FReal> source_point{sourceX, sourceY};
        std::complex<FReal> distance{target_point - source_point};
        std::complex<FReal> ln_distance{distance};

        *targetPotential += (ln_distance.real() * sourcePhysicalValue);
        *sourcePotential += (ln_distance.real() * targetPhysicalValue);
    }

    template <class FReal>
    inline void NonMutualParticles(const FReal sourceX, const FReal sourceY, const FReal sourceZ, const FReal sourcePhysicalValue,
                                   const FReal targetX, const FReal targetY, const FReal targetZ, const FReal targetPhysicalValue,
                                   FReal *targetForceX, FReal *targetForceY, FReal *targetForceZ, FReal *targetPotential)
    {
        std::complex<FReal> target_point{targetX, targetY};
        std::complex<FReal> source_point{sourceX, sourceY};
        std::complex<FReal> distance{target_point - source_point};
        std::complex<FReal> ln_distance{distance};

        *targetPotential += (ln_distance.real() * sourcePhysicalValue);
    }

    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void FullMutualScalar(const ParticlesClassValues &inNeighbors, ParticlesClassRhs &inNeighborsRhs, const long int nbParticlesSources,
                                 const ParticlesClassValues &inTargets, ParticlesClassRhs &inTargetsRhs, const long int nbParticlesTargets)
    {

        const FReal *const targetsX = GetPtr(inTargets[0]);
        const FReal *const targetsY = GetPtr(inTargets[1]);
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[0]);

        const FReal *const sourcesX = GetPtr(inNeighbors[0]);
        const FReal *const sourcesY = GetPtr(inNeighbors[1]);
        const FReal *const sourcesPhysicalValues = GetPtr(inNeighbors[2]);
        FReal *const sourcesPotentials = GetPtr(inNeighborsRhs[0]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            std::complex<FReal> target_point{targetsX[idxTarget], targetsY[idxTarget]};
            FReal tpo = 0;
            const FReal tv = targetsPhysicalValues[idxTarget];

            for (long int idxSource = 0; idxSource < nbParticlesSources; ++idxSource)
            {
                std::complex<FReal> source_point{targetsX[idxSource], targetsY[idxSource]};

                std::complex<FReal> distance{target_point - source_point};
                std::complex<FReal> ln_distance{distance};

                tpo += ln_distance.real() * sourcesPhysicalValues[idxSource];

                sourcesPotentials[idxSource] += ln_distance.real() * tv;
            }

            targetsPotentials[idxTarget] += (tpo);
        }
    }

#ifdef TBF_USE_INASTEMP
    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void FullMutual(const ParticlesClassValues &inNeighbors, ParticlesClassRhs &inNeighborsRhs, const long int nbParticlesSources,
                           const ParticlesClassValues &inTargets, ParticlesClassRhs &inTargetsRhs, const long int nbParticlesTargets)
    {
        using VecType = InaVecBestType<FReal>;

        const FReal *const targetsX = GetPtr(inTargets[0]);
        const FReal *const targetsY = GetPtr(inTargets[1]);
        const FReal *const targetsZ = GetPtr(inTargets[2]);
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[3]);
        FReal *const targetsForcesX = GetPtr(inTargetsRhs[0]);
        FReal *const targetsForcesY = GetPtr(inTargetsRhs[1]);
        FReal *const targetsForcesZ = GetPtr(inTargetsRhs[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[3]);

        const FReal *const sourcesX = GetPtr(inNeighbors[0]);
        const FReal *const sourcesY = GetPtr(inNeighbors[1]);
        const FReal *const sourcesZ = GetPtr(inNeighbors[2]);
        const FReal *const sourcesPhysicalValues = GetPtr(inNeighbors[3]);
        FReal *const sourcesForcesX = GetPtr(inNeighborsRhs[0]);
        FReal *const sourcesForcesY = GetPtr(inNeighborsRhs[1]);
        FReal *const sourcesForcesZ = GetPtr(inNeighborsRhs[2]);
        FReal *const sourcesPotentials = GetPtr(inNeighborsRhs[3]);

        const VecType mOne = VecType(1);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const long int nbVectorizedInteractions = (nbParticlesSources / VecType::GetVecLength()) * VecType::GetVecLength();
            {
                const VecType tx = VecType(targetsX[idxTarget]);
                const VecType ty = VecType(targetsY[idxTarget]);
                const VecType tz = VecType(targetsZ[idxTarget]);
                const VecType tv = VecType(targetsPhysicalValues[idxTarget]);
                VecType tfx = VecType::GetZero();
                VecType tfy = VecType::GetZero();
                VecType tfz = VecType::GetZero();
                VecType tpo = VecType::GetZero();

                for (long int idxSource = 0; idxSource < nbVectorizedInteractions; idxSource += VecType::GetVecLength())
                {
                    VecType dx = VecType(&sourcesX[idxSource]) - tx;
                    VecType dy = VecType(&sourcesY[idxSource]) - ty;
                    VecType dz = VecType(&sourcesZ[idxSource]) - tz;

                    VecType inv_square_distance = mOne / (dx * dx + dy * dy + dz * dz);
                    const VecType inv_distance = inv_square_distance.sqrt();

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * VecType(&sourcesPhysicalValues[idxSource]);

                    dx *= inv_square_distance;
                    dy *= inv_square_distance;
                    dz *= inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * VecType(&sourcesPhysicalValues[idxSource]);

                    (VecType(&sourcesForcesX[idxSource]) - dx).storeInArray(&sourcesForcesX[idxSource]);
                    (VecType(&sourcesForcesY[idxSource]) - dy).storeInArray(&sourcesForcesY[idxSource]);
                    (VecType(&sourcesForcesZ[idxSource]) - dz).storeInArray(&sourcesForcesZ[idxSource]);
                    (VecType(&sourcesPotentials[idxSource]) + inv_distance * tv).storeInArray(&sourcesPotentials[idxSource]);
                }

                targetsForcesX[idxTarget] += (tfx.horizontalSum());
                targetsForcesY[idxTarget] += (tfy.horizontalSum());
                targetsForcesZ[idxTarget] += (tfz.horizontalSum());
                targetsPotentials[idxTarget] += (tpo.horizontalSum());
            }
            {
                const FReal tx = targetsX[idxTarget];
                const FReal ty = targetsY[idxTarget];
                const FReal tz = targetsZ[idxTarget];
                const FReal tv = targetsPhysicalValues[idxTarget];
                FReal tfx = 0;
                FReal tfy = 0;
                FReal tfz = 0;
                FReal tpo = 0;

                for (long int idxSource = nbVectorizedInteractions; idxSource < nbParticlesSources; ++idxSource)
                {
                    FReal dx = sourcesX[idxSource] - tx;
                    FReal dy = sourcesY[idxSource] - ty;
                    FReal dz = sourcesZ[idxSource] - tz;

                    FReal inv_square_distance = FReal(1.0) / (dx * dx + dy * dy + dz * dz);
                    const FReal inv_distance = FMath::Sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * sourcesPhysicalValues[idxSource];

                    dx *= inv_square_distance;
                    dy *= inv_square_distance;
                    dz *= inv_square_distance;

                    sourcesForcesX[idxSource] -= dx;
                    sourcesForcesY[idxSource] -= dy;
                    sourcesForcesZ[idxSource] -= dz;
                    sourcesPotentials[idxSource] += inv_distance * tv;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * sourcesPhysicalValues[idxSource];
                }

                targetsForcesX[idxTarget] += (tfx);
                targetsForcesY[idxTarget] += (tfy);
                targetsForcesZ[idxTarget] += (tfz);
                targetsPotentials[idxTarget] += (tpo);
            }
        }
    }
#else
#define FullMutual FullMutualScalar
#endif

    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void GenericInnerScalar(const ParticlesClassValues &inTargets,
                                   ParticlesClassRhs &inTargetsRhs, const long int nbParticlesTargets)
    {
        const FReal *const targetsX = GetPtr(inTargets[0]);
        const FReal *const targetsY = GetPtr(inTargets[1]);
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[0]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {

            std::complex<FReal> target_point{targetsX[idxTarget], targetsY[idxTarget]};
            for (long int idxSource = idxTarget + 1; idxSource < nbParticlesTargets; ++idxSource)
            {
                std::complex<FReal> source_point{targetsX[idxSource], targetsY[idxSource]};

                std::complex<FReal> distance{target_point - source_point};
                std::complex<FReal> ln_distance{distance};

                targetsPotentials[idxTarget] += ln_distance.real() * targetsPhysicalValues[idxSource];

                targetsPotentials[idxSource] += ln_distance.real() * targetsPhysicalValues[idxTarget];
            }
        }
    }

#ifdef TBF_USE_INASTEMP

    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void GenericInner(const ParticlesClassValues &inTargets,
                             ParticlesClassRhs &inTargetsRhs, const long int nbParticlesTargets)
    {
        using VecType = InaVecBestType<FReal>;

        const FReal *const targetsX = GetPtr(inTargets[0]);
        const FReal *const targetsY = GetPtr(inTargets[1]);
        const FReal *const targetsZ = GetPtr(inTargets[2]);
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[3]);
        FReal *const targetsForcesX = GetPtr(inTargetsRhs[0]);
        FReal *const targetsForcesY = GetPtr(inTargetsRhs[1]);
        FReal *const targetsForcesZ = GetPtr(inTargetsRhs[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[3]);

        const VecType mOne = VecType(1);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const long int nbVectorizedInteractions = ((nbParticlesTargets - (idxTarget + 1)) / VecType::GetVecLength()) * VecType::GetVecLength() + (idxTarget + 1);
            {
                const VecType tx = VecType(targetsX[idxTarget]);
                const VecType ty = VecType(targetsY[idxTarget]);
                const VecType tz = VecType(targetsZ[idxTarget]);
                const VecType tv = VecType(targetsPhysicalValues[idxTarget]);
                VecType tfx = VecType::GetZero();
                VecType tfy = VecType::GetZero();
                VecType tfz = VecType::GetZero();
                VecType tpo = VecType::GetZero();

                for (long int idxSource = idxTarget + 1; idxSource < nbVectorizedInteractions; idxSource += VecType::GetVecLength())
                {
                    VecType dx = VecType(&targetsX[idxSource]) - tx;
                    VecType dy = VecType(&targetsY[idxSource]) - ty;
                    VecType dz = VecType(&targetsZ[idxSource]) - tz;

                    VecType inv_square_distance = mOne / (dx * dx + dy * dy + dz * dz);
                    const VecType inv_distance = inv_square_distance.sqrt();

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * VecType(&targetsPhysicalValues[idxSource]);

                    dx *= inv_square_distance;
                    dy *= inv_square_distance;
                    dz *= inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * VecType(&targetsPhysicalValues[idxSource]);

                    (VecType(&targetsForcesX[idxSource]) - dx).storeInArray(&targetsForcesX[idxSource]);
                    (VecType(&targetsForcesY[idxSource]) - dy).storeInArray(&targetsForcesY[idxSource]);
                    (VecType(&targetsForcesZ[idxSource]) - dz).storeInArray(&targetsForcesZ[idxSource]);
                    (VecType(&targetsPotentials[idxSource]) + inv_distance * tv).storeInArray(&targetsPotentials[idxSource]);
                }

                targetsForcesX[idxTarget] += (tfx.horizontalSum());
                targetsForcesY[idxTarget] += (tfy.horizontalSum());
                targetsForcesZ[idxTarget] += (tfz.horizontalSum());
                targetsPotentials[idxTarget] += (tpo.horizontalSum());
            }
            {
                const FReal tx = targetsX[idxTarget];
                const FReal ty = targetsY[idxTarget];
                const FReal tz = targetsZ[idxTarget];
                const FReal tv = targetsPhysicalValues[idxTarget];
                FReal tfx = 0;
                FReal tfy = 0;
                FReal tfz = 0;
                FReal tpo = 0;

                for (long int idxSource = nbVectorizedInteractions; idxSource < nbParticlesTargets; ++idxSource)
                {
                    FReal dx = targetsX[idxSource] - tx;
                    FReal dy = targetsY[idxSource] - ty;
                    FReal dz = targetsZ[idxSource] - tz;

                    FReal inv_square_distance = FReal(1.0) / (dx * dx + dy * dy + dz * dz);
                    const FReal inv_distance = FMath::Sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * targetsPhysicalValues[idxSource];

                    dx *= inv_square_distance;
                    dy *= inv_square_distance;
                    dz *= inv_square_distance;

                    targetsForcesX[idxSource] -= dx;
                    targetsForcesY[idxSource] -= dy;
                    targetsForcesZ[idxSource] -= dz;
                    targetsPotentials[idxSource] += inv_distance * tv;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * targetsPhysicalValues[idxSource];
                }

                targetsForcesX[idxTarget] += (tfx);
                targetsForcesY[idxTarget] += (tfy);
                targetsForcesZ[idxTarget] += (tfz);
                targetsPotentials[idxTarget] += (tpo);
            }
        }
    }
#else
#define GenericInner GenericInnerScalar
#endif

    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void GenericFullRemoteScalar(const ParticlesClassValues &inNeighbors, const long int nbParticlesSources,
                                        const ParticlesClassValues &inTargets, ParticlesClassRhs &inTargetsRhs, const long int nbParticlesTargets)
    {
        const FReal *const targetsX = GetPtr(inTargets[0]);
        const FReal *const targetsY = GetPtr(inTargets[1]);
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[0]);

        const FReal *const sourcesX = GetPtr(inNeighbors[0]);
        const FReal *const sourcesY = GetPtr(inNeighbors[1]);
        const FReal *const sourcesPhysicalValues = GetPtr(inNeighbors[2]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            std::complex<FReal> target_point{targetsX[idxTarget], targetsY[idxTarget]};
            const FReal tv = targetsPhysicalValues[idxTarget];
            FReal tpo = 0;

            for (long int idxSource = 0; idxSource < nbParticlesSources; ++idxSource)
            {

                std::complex<FReal> source_point{sourcesX[idxSource], sourcesY[idxSource]};
                std::complex<FReal> distance{target_point - source_point};
                std::complex<FReal> ln_distance{distance};

                tpo += ln_distance.real() * sourcesPhysicalValues[idxSource];
            }

            targetsPotentials[idxTarget] += (tpo);
        }
    }

#ifdef TBF_USE_INASTEMP
    template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
    static void GenericFullRemote(const ParticlesClassValues &inNeighbors, const long int nbParticlesSources,
                                  const ParticlesClassValues &inTargets, ParticlesClassRhs &inTargetsRhs, const long int nbParticlesTargets)
    {
        using VecType = InaVecBestType<FReal>;

        const FReal *const targetsX = GetPtr(inTargets[0]);
        const FReal *const targetsY = GetPtr(inTargets[1]);
        const FReal *const targetsZ = GetPtr(inTargets[2]);
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[3]);
        FReal *const targetsForcesX = GetPtr(inTargetsRhs[0]);
        FReal *const targetsForcesY = GetPtr(inTargetsRhs[1]);
        FReal *const targetsForcesZ = GetPtr(inTargetsRhs[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[3]);

        const VecType mOne = VecType(1);

        const FReal *const sourcesX = GetPtr(inNeighbors[0]);
        const FReal *const sourcesY = GetPtr(inNeighbors[1]);
        const FReal *const sourcesZ = GetPtr(inNeighbors[2]);
        const FReal *const sourcesPhysicalValues = GetPtr(inNeighbors[3]);

        const long int nbVectorizedInteractionsSource = (nbParticlesSources / VecType::GetVecLength()) * VecType::GetVecLength();

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            {
                const VecType tx = VecType(targetsX[idxTarget]);
                const VecType ty = VecType(targetsY[idxTarget]);
                const VecType tz = VecType(targetsZ[idxTarget]);
                const VecType tv = VecType(targetsPhysicalValues[idxTarget]);
                VecType tfx = VecType::GetZero();
                VecType tfy = VecType::GetZero();
                VecType tfz = VecType::GetZero();
                VecType tpo = VecType::GetZero();

                for (long int idxSource = 0; idxSource < nbVectorizedInteractionsSource; idxSource += VecType::GetVecLength())
                {
                    VecType dx = VecType(&sourcesX[idxSource]) - tx;
                    VecType dy = VecType(&sourcesY[idxSource]) - ty;
                    VecType dz = VecType(&sourcesZ[idxSource]) - tz;

                    VecType inv_square_distance = mOne / (dx * dx + dy * dy + dz * dz);
                    const VecType inv_distance = inv_square_distance.sqrt();

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * VecType(&sourcesPhysicalValues[idxSource]);

                    dx *= inv_square_distance;
                    dy *= inv_square_distance;
                    dz *= inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * VecType(&sourcesPhysicalValues[idxSource]);
                }

                targetsForcesX[idxTarget] += (tfx.horizontalSum());
                targetsForcesY[idxTarget] += (tfy.horizontalSum());
                targetsForcesZ[idxTarget] += (tfz.horizontalSum());
                targetsPotentials[idxTarget] += (tpo.horizontalSum());
            }
            {
                const FReal tx = targetsX[idxTarget];
                const FReal ty = targetsY[idxTarget];
                const FReal tz = targetsZ[idxTarget];
                const FReal tv = targetsPhysicalValues[idxTarget];
                FReal tfx = 0;
                FReal tfy = 0;
                FReal tfz = 0;
                FReal tpo = 0;

                for (long int idxSource = nbVectorizedInteractionsSource; idxSource < nbParticlesSources; ++idxSource)
                {
                    FReal dx = sourcesX[idxSource] - tx;
                    FReal dy = sourcesY[idxSource] - ty;
                    FReal dz = sourcesZ[idxSource] - tz;

                    FReal inv_square_distance = FReal(1) / (dx * dx + dy * dy + dz * dz);
                    const FReal inv_distance = FMath::Sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * sourcesPhysicalValues[idxSource];

                    dx *= inv_square_distance;
                    dy *= inv_square_distance;
                    dz *= inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * sourcesPhysicalValues[idxSource];
                }

                targetsForcesX[idxTarget] += (tfx);
                targetsForcesY[idxTarget] += (tfy);
                targetsForcesZ[idxTarget] += (tfz);
                targetsPotentials[idxTarget] += (tpo);
            }
        }
    }
#else
#define GenericFullRemote GenericFullRemoteScalar
#endif

} // End namespace

#endif // FP2PR_HPP
