#ifndef TBFP2PCUDA_HPP
#define TBFP2PCUDA_HPP


namespace TbfP2PCuda{

__device__ auto CuMin(const std::size_t& v1, const std::size_t& v2){
    return (v1 < v2 ? v1 : v2);
}

template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
__device__ static void GenericFullRemote(const ParticlesClassValues& inNeighbors,
                                    const long int nbParticlesSources,
                                    const ParticlesClassValues& inTargets, ParticlesClassRhs& inTargetsRhs,
                                    const long int nbParticlesTargets){
    const FReal*const targetsX = (inTargets[0]);
    const FReal*const targetsY = (inTargets[1]);
    const FReal*const targetsZ = (inTargets[2]);
    const FReal*const targetsPhysicalValues = (inTargets[3]);
    FReal*const targetsForcesX = (inTargetsRhs[0]);
    FReal*const targetsForcesY = (inTargetsRhs[1]);
    FReal*const targetsForcesZ = (inTargetsRhs[2]);
    FReal*const targetsPotentials = (inTargetsRhs[3]);

    const FReal*const valuesSrcX = (inNeighbors[0]);
    const FReal*const valuesSrcY = (inNeighbors[1]);
    const FReal*const valuesSrcZ = (inNeighbors[2]);
    const FReal*const valuesSrcPhysicalValues = (inNeighbors[3]);

    constexpr std::size_t SHARED_MEMORY_SIZE = 256;
    const std::size_t nbIterations = ((nbParticlesTargets+blockDim.x-1)/blockDim.x)*blockDim.x;

    for(std::size_t idxTarget = threadIdx.x ; idxTarget < nbIterations ; idxTarget += blockDim.x){
        const bool threadCompute = (idxTarget<nbParticlesTargets);

        double tx;
        double ty;
        double tz;
        double tv;

        if(threadCompute){
            tx = double(targetsX[idxTarget]);
            ty = double(targetsY[idxTarget]);
            tz = double(targetsZ[idxTarget]);
            tv = double(targetsPhysicalValues[idxTarget]);
        }

        double  tfx = double(0.);
        double  tfy = double(0.);
        double  tfz = double(0.);
        double  tpo = double(0.);

        for(std::size_t idxCopy = 0 ; idxCopy < nbParticlesSources ; idxCopy += SHARED_MEMORY_SIZE){
            __shared__ double sourcesX[SHARED_MEMORY_SIZE];
            __shared__ double sourcesY[SHARED_MEMORY_SIZE];
            __shared__ double sourcesZ[SHARED_MEMORY_SIZE];
            __shared__ double sourcesPhys[SHARED_MEMORY_SIZE];

            const std::size_t nbCopies = CuMin(SHARED_MEMORY_SIZE, nbParticlesSources-idxCopy);
            for(std::size_t idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                sourcesX[idx] = valuesSrcX[idx+idxCopy];
                sourcesY[idx] = valuesSrcY[idx+idxCopy];
                sourcesZ[idx] = valuesSrcZ[idx+idxCopy];
                sourcesPhys[idx] = valuesSrcPhysicalValues[idx+idxCopy];
            }

            __syncthreads();

            if(threadCompute){
                for(std::size_t otherIndex = 0; otherIndex < nbCopies; ++otherIndex) {
                    double dx = tx - sourcesX[otherIndex];
                    double dy = ty - sourcesY[otherIndex];
                    double dz = tz - sourcesZ[otherIndex];

                    double inv_square_distance = double(1) / (dx*dx + dy*dy + dz*dz);
                    const double inv_distance = sqrt(inv_square_distance);

                    inv_square_distance *= inv_distance;
                    inv_square_distance *= tv * sourcesPhys[otherIndex];

                    dx *= - inv_square_distance;
                    dy *= - inv_square_distance;
                    dz *= - inv_square_distance;

                    tfx += dx;
                    tfy += dy;
                    tfz += dz;
                    tpo += inv_distance * sourcesPhys[otherIndex];
                }
            }

            __syncthreads();
        }

        if( threadCompute ){
            targetsForcesX[idxTarget] += tfx;
            targetsForcesY[idxTarget] += tfy;
            targetsForcesZ[idxTarget] += tfz;
            targetsPotentials[idxTarget] += tpo;
        }

        __syncthreads();
    }
}


template <class FReal, class ParticlesClassValues, class ParticlesClassRhs>
__device__ static void GenericInner(const ParticlesClassValues& inTargets,
                               ParticlesClassRhs& inTargetsRhs, const long int nbParticlesTargets){
    const FReal*const targetsX = (inTargets[0]);
    const FReal*const targetsY = (inTargets[1]);
    const FReal*const targetsZ = (inTargets[2]);
    const FReal*const targetsPhysicalValues = (inTargets[3]);
    FReal*const targetsForcesX = (inTargetsRhs[0]);
    FReal*const targetsForcesY = (inTargetsRhs[1]);
    FReal*const targetsForcesZ = (inTargetsRhs[2]);
    FReal*const targetsPotentials = (inTargetsRhs[3]);

    constexpr std::size_t SHARED_MEMORY_SIZE = 256;
    const std::size_t nbIterations = ((nbParticlesTargets+blockDim.x-1)/blockDim.x)*blockDim.x;

    for(std::size_t idxTarget = threadIdx.x ; idxTarget < nbIterations ; idxTarget += blockDim.x){
        const bool threadCompute = (idxTarget<nbParticlesTargets);

        FReal tx;
        FReal ty;
        FReal tz;
        FReal tv;

        if(threadCompute){
            tx = FReal(targetsX[idxTarget]);
            ty = FReal(targetsY[idxTarget]);
            tz = FReal(targetsZ[idxTarget]);
            tv = FReal(targetsPhysicalValues[idxTarget]);
        }

        FReal  tfx = FReal(0.);
        FReal  tfy = FReal(0.);
        FReal  tfz = FReal(0.);
        FReal  tpo = FReal(0.);

        for(std::size_t idxCopy = 0 ; idxCopy < nbParticlesTargets ; idxCopy += SHARED_MEMORY_SIZE){
            __shared__ FReal sourcesX[SHARED_MEMORY_SIZE];
            __shared__ FReal sourcesY[SHARED_MEMORY_SIZE];
            __shared__ FReal sourcesZ[SHARED_MEMORY_SIZE];
            __shared__ FReal sourcesPhys[SHARED_MEMORY_SIZE];

            const std::size_t nbCopies = CuMin(SHARED_MEMORY_SIZE, nbParticlesTargets-idxCopy);
            for(std::size_t idx = threadIdx.x ; idx < nbCopies ; idx += blockDim.x){
                sourcesX[idx] = targetsX[idx+idxCopy];
                sourcesY[idx] = targetsY[idx+idxCopy];
                sourcesZ[idx] = targetsZ[idx+idxCopy];
                sourcesPhys[idx] = targetsPhysicalValues[idx+idxCopy];
            }

            __syncthreads();

            if(threadCompute){
                for(std::size_t otherIndex = 0; otherIndex < nbCopies; ++otherIndex) {
                    if(idxCopy + otherIndex != idxTarget){
                        FReal dx = tx - sourcesX[otherIndex];
                        FReal dy = ty - sourcesY[otherIndex];
                        FReal dz = tz - sourcesZ[otherIndex];

                        FReal inv_square_distance = FReal(1) / (dx*dx + dy*dy + dz*dz);
                        const FReal inv_distance = sqrt(inv_square_distance);

                        inv_square_distance *= inv_distance;
                        inv_square_distance *= tv * sourcesPhys[otherIndex];

                        dx *= - inv_square_distance;
                        dy *= - inv_square_distance;
                        dz *= - inv_square_distance;

                        tfx += dx;
                        tfy += dy;
                        tfz += dz;
                        tpo += inv_distance * sourcesPhys[otherIndex];
                    }
                }
            }

            __syncthreads();
        }

        if( threadCompute ){
            targetsForcesX[idxTarget] += tfx;
            targetsForcesY[idxTarget] += tfy;
            targetsForcesZ[idxTarget] += tfz;
            targetsPotentials[idxTarget] += tpo;
        }

        __syncthreads();
    }
}


}

#endif
