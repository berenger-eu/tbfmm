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
#include "kernels/unifkernel/FMath.hpp"

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
        FReal ln_distance{std::log(distance).real() * FMath::FOneDiv2Pi<FReal>()};

        *targetPotential += (ln_distance * sourcePhysicalValue);
        *sourcePotential += (ln_distance * targetPhysicalValue);
    }

    template <class FReal>
    inline void NonMutualParticles(const FReal sourceX, const FReal sourceY, const FReal sourceZ, const FReal sourcePhysicalValue,
                                   const FReal targetX, const FReal targetY, const FReal targetZ, const FReal targetPhysicalValue,
                                   FReal *targetForceX, FReal *targetForceY, FReal *targetForceZ, FReal *targetPotential)
    {
        std::complex<FReal> target_point{targetX, targetY};
        std::complex<FReal> source_point{sourceX, sourceY};
        std::complex<FReal> distance{target_point - source_point};
        FReal ln_distance{std::log(distance).real() * FMath::FOneDiv2Pi<FReal>()};

        *targetPotential += (ln_distance * sourcePhysicalValue);
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
                std::complex<FReal> source_point{sourcesX[idxSource], sourcesY[idxSource]};

                std::complex<FReal> distance{target_point - source_point};
                FReal ln_distance{std::log(distance).real() * FMath::FOneDiv2Pi<FReal>()};

                tpo += ln_distance * sourcesPhysicalValues[idxSource];

                sourcesPotentials[idxSource] += ln_distance * tv;
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
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[0]);

        const FReal *const sourcesX = GetPtr(inNeighbors[0]);
        const FReal *const sourcesY = GetPtr(inNeighbors[1]);
        const FReal *const sourcesPhysicalValues = GetPtr(inNeighbors[2]);
        FReal *const sourcesPotentials = GetPtr(inNeighborsRhs[0]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const long int nbVectorizedInteractions = (nbParticlesSources / VecType::GetVecLength()) * VecType::GetVecLength();
            {
                const std::complex<VecType> target_point{VecType(targetsX[idxTarget]), VecType(targetsY[idxTarget])};
                const VecType tv = VecType(targetsPhysicalValues[idxTarget]);
                VecType tpo = VecType::GetZero();

                for (long int idxSource = 0; idxSource < nbVectorizedInteractions; idxSource += VecType::GetVecLength())
                {

                    std::complex<VecType> source_point{VecType(&sourcesX[idxSource]), VecType(&sourcesY[idxSource])};
                    std::complex<VecType> distance{target_point - source_point};
                    std::complex<VecType> ln_distance{distance};

                    tpo += ln_distance.real() * VecType(&sourcesPhysicalValues[idxSource]);

                    (VecType(&sourcesPotentials[idxSource]) + ln_distance.real() * tv).storeInArray(&sourcesPotentials[idxSource]);
                }

                targetsPotentials[idxTarget] += (tpo.horizontalSum());
            }
            {
                std::complex<FReal> target_point{targetsX[idxTarget], targetsY[idxTarget]};
                const FReal tv = targetsPhysicalValues[idxTarget];
                FReal tpo = 0;

                for (long int idxSource = nbVectorizedInteractions; idxSource < nbParticlesSources; ++idxSource)
                {
                    std::complex<FReal> source_point{sourcesX[idxSource], sourcesY[idxSource]};

                    std::complex<FReal> distance{target_point - source_point};
                    std::complex<FReal> ln_distance{distance};

                    tpo += ln_distance.real() * sourcesPhysicalValues[idxSource];

                    sourcesPotentials[idxSource] += ln_distance.real() * tv;
                }

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
                FReal ln_distance{std::log(distance).real() * FMath::FOneDiv2Pi<FReal>()};

                targetsPotentials[idxTarget] += ln_distance * targetsPhysicalValues[idxSource];

                targetsPotentials[idxSource] += ln_distance * targetsPhysicalValues[idxTarget];
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
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[0]);

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            const long int nbVectorizedInteractions = ((nbParticlesTargets - (idxTarget + 1)) / VecType::GetVecLength()) * VecType::GetVecLength() + (idxTarget + 1);
            {
                const std::complex<VecType> target_point{VecType(targetsX[idxTarget]), VecType(targetsY[idxTarget])};
                const VecType tv = VecType(targetsPhysicalValues[idxTarget]);
                VecType tpo = VecType::GetZero();

                for (long int idxSource = idxTarget + 1; idxSource < nbVectorizedInteractions; idxSource += VecType::GetVecLength())
                {

                    std::complex<VecType> source_point{VecType(&targetsX[idxSource]), VecType(&targetsY[idxSource])};

                    std::complex<VecType> distance{target_point - source_point};
                    std::complex<VecType> ln_distance{distance};

                    tpo += ln_distance.real() * VecType(&targetsPhysicalValues[idxSource]);

                    (VecType(&targetsPotentials[idxSource]) + ln_distance.real() * tv).storeInArray(&targetsPotentials[idxSource]);
                }

                targetsPotentials[idxTarget] += (tpo.horizontalSum());
            }
            {

                std::complex<FReal> target_point{targetsX[idxTarget], targetsY[idxTarget]};
                FReal tpo = 0;

                for (long int idxSource = nbVectorizedInteractions; idxSource < nbParticlesTargets; ++idxSource)
                {
                    std::complex<FReal> source_point{targetsX[idxSource], targetsY[idxSource]};
                    std::complex<FReal> distance{target_point - source_point};
                    std::complex<FReal> ln_distance{distance};

                    tpo += ln_distance.real() * targetsPhysicalValues[idxSource];
                }

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
                FReal ln_distance{std::log(distance).real() * FMath::FOneDiv2Pi<FReal>()};

                tpo += ln_distance * sourcesPhysicalValues[idxSource];
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
        const FReal *const targetsPhysicalValues = GetPtr(inTargets[2]);
        FReal *const targetsPotentials = GetPtr(inTargetsRhs[0]);

        const FReal *const sourcesX = GetPtr(inNeighbors[0]);
        const FReal *const sourcesY = GetPtr(inNeighbors[1]);
        const FReal *const sourcesPhysicalValues = GetPtr(inNeighbors[2]);

        const long int nbVectorizedInteractionsSource = (nbParticlesSources / VecType::GetVecLength()) * VecType::GetVecLength();

        for (long int idxTarget = 0; idxTarget < nbParticlesTargets; ++idxTarget)
        {
            {
                const std::complex<VecType> target_point{VecType(targetsX[idxTarget]), VecType(targetsY[idxTarget])};
                const VecType tv = VecType(targetsPhysicalValues[idxTarget]);
                VecType tpo = VecType::GetZero();

                for (long int idxSource = 0; idxSource < nbVectorizedInteractionsSource; idxSource += VecType::GetVecLength())
                {
                    std::complex<VecType> source_point{VecType(&sourcesX[idxSource]), VecType(&sourcesY[idxSource])};
                    std::complex<VecType> distance{target_point - source_point};
                    std::complex<VecType> ln_distance{distance};

                    tpo += ln_distance.real() * VecType(&sourcesPhysicalValues[idxSource]);
                }

                targetsPotentials[idxTarget] += (tpo.horizontalSum());
            }
            {
                std::complex<FReal> target_point{targetsX[idxTarget], targetsY[idxTarget]};
                const FReal tv = targetsPhysicalValues[idxTarget];
                FReal tpo = 0;

                for (long int idxSource = nbVectorizedInteractionsSource; idxSource < nbParticlesSources; ++idxSource)
                {
                    std::complex<FReal> source_point{sourcesX[idxSource], sourcesY[idxSource]};
                    std::complex<FReal> distance{target_point - source_point};
                    std::complex<FReal> ln_distance{distance};

                    tpo += ln_distance.real() * sourcesPhysicalValues[idxSource];
                }

                targetsPotentials[idxTarget] += (tpo);
            }
        }
    }
#else
#define GenericFullRemote GenericFullRemoteScalar
#endif

} // End namespace

#endif // FP2PR_HPP
