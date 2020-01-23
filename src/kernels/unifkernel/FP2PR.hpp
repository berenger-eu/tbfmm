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
#ifndef FP2PR_HPP
#define FP2PR_HPP

#include "FMath.hpp"

/**
 * @brief The FP2PR namespace
 */
namespace FP2PR{
template <class FReal>
inline void MutualParticles(const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal sourcePhysicalValue,
                            FReal* sourceForceX, FReal* sourceForceY, FReal* sourceForceZ, FReal* sourcePotential,
                            const FReal targetX,const FReal targetY,const FReal targetZ, const FReal targetPhysicalValue,
                            FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential
                            ){
    FReal dx = sourceX - targetX;
    FReal dy = sourceY - targetY;
    FReal dz = sourceZ - targetZ;

    FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
    FReal inv_distance = FMath::Sqrt(inv_square_distance);

    inv_square_distance *= inv_distance;
    inv_square_distance *= targetPhysicalValue * sourcePhysicalValue;

    dx *= inv_square_distance;
    dy *= inv_square_distance;
    dz *= inv_square_distance;

    *targetForceX += dx;
    *targetForceY += dy;
    *targetForceZ += dz;
    *targetPotential += ( inv_distance * sourcePhysicalValue );

    *sourceForceX -= dx;
    *sourceForceY -= dy;
    *sourceForceZ -= dz;
    *sourcePotential += ( inv_distance * targetPhysicalValue );
}

template <class FReal>
inline void NonMutualParticles(const FReal sourceX,const FReal sourceY,const FReal sourceZ, const FReal sourcePhysicalValue,
                               const FReal targetX,const FReal targetY,const FReal targetZ, const FReal targetPhysicalValue,
                               FReal* targetForceX, FReal* targetForceY, FReal* targetForceZ, FReal* targetPotential){
    FReal dx = sourceX - targetX;
    FReal dy = sourceY - targetY;
    FReal dz = sourceZ - targetZ;

    FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
    FReal inv_distance = FMath::Sqrt(inv_square_distance);

    inv_square_distance *= inv_distance;
    inv_square_distance *= targetPhysicalValue * sourcePhysicalValue;

    dx *= inv_square_distance;
    dy *= inv_square_distance;
    dz *= inv_square_distance;

    *targetForceX += dx;
    *targetForceY += dy;
    *targetForceZ += dz;
    *targetPotential += ( inv_distance * sourcePhysicalValue );
}


template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
static void FullMutual(const ParticlesClassValues& inNeighbors, ParticlesClassRhs& inNeighborsRhs, const long int nbParticlesSources,
                      const ParticlesClassValues& inTargets, ParticlesClassRhs& inTargetsRhs, const long int nbParticlesTargets){

    const FReal*const targetsX = inTargets[0];
    const FReal*const targetsY = inTargets[1];
    const FReal*const targetsZ = inTargets[2];
    const FReal*const targetsPhysicalValues = inTargets[3];
    FReal*const targetsForcesX = inTargetsRhs[0];
    FReal*const targetsForcesY = inTargetsRhs[1];
    FReal*const targetsForcesZ = inTargetsRhs[2];
    FReal*const targetsPotentials = inTargetsRhs[3];

    const FReal mOne = 1;

    const FReal*const sourcesX = inNeighbors[0];
    const FReal*const sourcesY = inNeighbors[1];
    const FReal*const sourcesZ = inNeighbors[2];
    const FReal*const sourcesPhysicalValues = inNeighbors[3];
    FReal*const sourcesForcesX = inNeighborsRhs[0];
    FReal*const sourcesForcesY = inNeighborsRhs[1];
    FReal*const sourcesForcesZ = inNeighborsRhs[2];
    FReal*const sourcesPotentials = inNeighborsRhs[3];

    for(long int idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        const FReal tx = targetsX[idxTarget];
        const FReal ty = targetsY[idxTarget];
        const FReal tz = targetsZ[idxTarget];
        const FReal tv = targetsPhysicalValues[idxTarget];
        FReal  tfx = 0;
        FReal  tfy = 0;
        FReal  tfz = 0;
        FReal  tpo = 0;

        for(long int idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){
            FReal dx = sourcesX[idxSource] - tx;
            FReal dy = sourcesY[idxSource] - ty;
            FReal dz = sourcesZ[idxSource] - tz;

            FReal inv_square_distance = mOne / (dx*dx + dy*dy + dz*dz);
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

            sourcesForcesX[idxSource] -= dx;
            sourcesForcesY[idxSource] -= dy;
            sourcesForcesZ[idxSource] -= dz;
            sourcesPotentials[idxSource] += inv_distance * tv;
        }

        targetsForcesX[idxTarget] += (tfx);
        targetsForcesY[idxTarget] += (tfy);
        targetsForcesZ[idxTarget] += (tfz);
        targetsPotentials[idxTarget] += (tpo);
    }
}

template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
static void GenericInner(const ParticlesClassValues& inTargets,
                         ParticlesClassRhs& inTargetsRhs, const long int nbParticlesTargets){

    const FReal*const targetsX = inTargets[0];
    const FReal*const targetsY = inTargets[1];
    const FReal*const targetsZ = inTargets[2];
    const FReal*const targetsPhysicalValues = inTargets[3];
    FReal*const targetsForcesX = inTargetsRhs[0];
    FReal*const targetsForcesY = inTargetsRhs[1];
    FReal*const targetsForcesZ = inTargetsRhs[2];
    FReal*const targetsPotentials = inTargetsRhs[3];

    for(long int idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        for(long int idxSource = idxTarget+1 ; idxSource < nbParticlesTargets ; ++idxSource){
            FReal dx = targetsX[idxSource] - targetsX[idxTarget];
            FReal dy = targetsY[idxSource] - targetsY[idxTarget];
            FReal dz = targetsZ[idxSource] - targetsZ[idxTarget];

            FReal inv_square_distance = FReal(1.0) / (dx*dx + dy*dy + dz*dz);
            const FReal inv_distance = FMath::Sqrt(inv_square_distance);

            inv_square_distance *= inv_distance;
            inv_square_distance *= targetsPhysicalValues[idxTarget] * targetsPhysicalValues[idxSource];

            dx *= inv_square_distance;
            dy *= inv_square_distance;
            dz *= inv_square_distance;

            targetsForcesX[idxTarget] += dx;
            targetsForcesY[idxTarget] += dy;
            targetsForcesZ[idxTarget] += dz;
            targetsPotentials[idxTarget] += inv_distance * targetsPhysicalValues[idxSource];

            targetsForcesX[idxSource] -= dx;
            targetsForcesY[idxSource] -= dy;
            targetsForcesZ[idxSource] -= dz;
            targetsPotentials[idxSource] += inv_distance * targetsPhysicalValues[idxTarget];
        }
    }
}

template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
static void GenericFullRemote(const ParticlesClassValues&& inNeighbors, const long int nbParticlesSources,
                              const ParticlesClassValues&& inTargets, ParticlesClassRhs&& inTargetsRhs, const long int nbParticlesTargets){

    const FReal*const targetsX = inTargets[0];
    const FReal*const targetsY = inTargets[1];
    const FReal*const targetsZ = inTargets[2];
    const FReal*const targetsPhysicalValues = inTargets[3];
    FReal*const targetsForcesX = inTargetsRhs[0];
    FReal*const targetsForcesY = inTargetsRhs[1];
    FReal*const targetsForcesZ = inTargetsRhs[2];
    FReal*const targetsPotentials = inTargetsRhs[3];

    const FReal mOne = 1;

    const FReal*const sourcesX = inNeighbors[0];
    const FReal*const sourcesY = inNeighbors[1];
    const FReal*const sourcesZ = inNeighbors[2];
    const FReal*const sourcesPhysicalValues = inNeighbors[3];

    for(long int idxTarget = 0 ; idxTarget < nbParticlesTargets ; ++idxTarget){
        const FReal tx = targetsX[idxTarget];
        const FReal ty = targetsY[idxTarget];
        const FReal tz = targetsZ[idxTarget];
        const FReal tv = targetsPhysicalValues[idxTarget];
        FReal  tfx = 0;
        FReal  tfy = 0;
        FReal  tfz = 0;
        FReal  tpo = 0;

        for(long int idxSource = 0 ; idxSource < nbParticlesSources ; ++idxSource){
            FReal dx = sourcesX[idxSource] - tx;
            FReal dy = sourcesY[idxSource] - ty;
            FReal dz = sourcesZ[idxSource] - tz;

            FReal inv_square_distance = mOne / (dx*dx + dy*dy + dz*dz);
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

} // End namespace




#endif // FP2PR_HPP
