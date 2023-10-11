#ifndef TBFSMSTARPUCALLBACKSCUDA_HPP
#define TBFSMSTARPUCALLBACKSCUDA_HPP

#include <cstddef>
#include <starpu.h>


class TbfSmStarpuCallbacksCuda{
public:
    template<class ThisClass, class CellContainerClass, class ParticleContainerClass>
    static void P2MCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        unsigned char* groupCellsData;
        size_t groupCellsDataSize;
        unsigned char* groupParticlesData;
        size_t groupParticlesDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &groupCellsData, &groupCellsDataSize, &groupParticlesData, &groupParticlesDataSize);

        unsigned char* particleData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* leafData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t leafDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* leafMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t leafMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        const CellContainerClass leafGroupObjCpu(groupCellsData, groupCellsDataSize,
                                           nullptr, 0,
                                            nullptr, 0);
        const ParticleContainerClass particleGroupObjCpu(groupParticlesData, groupParticlesDataSize,
                                                          nullptr, 0);

        CellContainerClass leafGroupObjCuda(leafData, leafDataSize,
                                        leafMultipole, leafMultipoleSize,
                                        nullptr, 0, false);
        const ParticleContainerClass particleGroupObjCuda(particleData, particleDataSize,
                                                      nullptr, 0, false);

        thisptr->kernelWrapperCuda.P2M(starpu_cuda_get_local_stream(),
                                       thisptr->kernels[starpu_worker_get_id()],
                                       particleGroupObjCpu, leafGroupObjCpu,
                                       particleGroupObjCuda, leafGroupObjCuda);
    }

    template<class ThisClass, class ParticleContainerClass>
    static void P2PBetweenLeavesCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        typename ThisClass::VecOfIndexes* indexesForGroup_first;
        unsigned char* srcDataCpu;
        size_t srcDataSizeCpu;
        unsigned char* tgtDataCpu;
        size_t tgtDataSizeCpu;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first,
                                   &srcDataCpu, &srcDataSizeCpu, &tgtDataCpu, &tgtDataSizeCpu);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* tgtRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t tgtRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const ParticleContainerClass groupSrcCpu(srcDataCpu, srcDataSizeCpu,
                                        nullptr, 0, true);
        const ParticleContainerClass groupTargetCpu(tgtDataCpu, tgtDataSizeCpu,
                                           nullptr, 0, true);

        ParticleContainerClass groupSrc(srcData, srcDataSize,
                                        srcRhs, srcRhsSize, false);
        ParticleContainerClass groupTarget(tgtData, tgtDataSize,
                                           tgtRhs, tgtRhsSize, false);

        thisptr->kernelWrapperCuda.P2PBetweenGroups(starpu_cuda_get_local_stream(),
                                                    thisptr->kernels[starpu_worker_get_id()],
                                                    groupSrcCpu, groupTargetCpu,
                                                    groupSrc, groupTarget, *indexesForGroup_first);
    }

    template<class ThisClass, class ParticleContainerClass>
    static void P2POneLeafCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        typename ThisClass::VecOfIndexes* indexesForGroup_first;
        unsigned char* particleDataCpu;
        size_t particleDataSizeCpu;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first, &particleDataCpu, &particleDataSizeCpu);

        unsigned char* particleData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* particleRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t particleRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        const ParticleContainerClass currentGroupCpu(particleDataCpu, particleDataSizeCpu,
                                            nullptr, 0, true);

        ParticleContainerClass currentGroup(particleData, particleDataSize,
                                            particleRhs, particleRhsSize, false);

        thisptr->kernelWrapperCuda.P2PInGroup(starpu_cuda_get_local_stream(),
                                              thisptr->kernels[starpu_worker_get_id()],
                                              currentGroupCpu,
                                              currentGroup, *indexesForGroup_first);
        thisptr->kernelWrapperCuda.P2PInner(starpu_cuda_get_local_stream(),
                                            thisptr->kernels[starpu_worker_get_id()], currentGroupCpu, currentGroup);
    }

    template<class ThisClass, class CellContainerClass>
    static void M2MCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        long int idxLevel;
        unsigned char* groupCellsLowerData;
        size_t groupCellsLowerDataSize;
        unsigned char* groupCellsUpperData;
        size_t groupCellsUpperDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &groupCellsLowerData, &groupCellsLowerDataSize,
                                   &groupCellsUpperData, &groupCellsUpperDataSize);

        unsigned char* lowerData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t lowerDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* lowerMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t lowerMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* upperData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t upperDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* upperMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t upperMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass lowerGroupObjCpu(groupCellsLowerData, groupCellsLowerDataSize, nullptr, 0,
                                               nullptr, 0);
        const CellContainerClass upperGroupObjCpu(groupCellsUpperData, groupCellsUpperDataSize, nullptr, 0,
                                         nullptr, 0);

        const CellContainerClass lowerGroupObjCuda(lowerData, lowerDataSize, lowerMultipole, lowerMultipoleSize,
                                               nullptr, 0, false);
        CellContainerClass upperGroupObjCuda(upperData, upperDataSize, upperMultipole, upperMultipoleSize,
                                         nullptr, 0, false);

        thisptr->kernelWrapperCuda.M2M(starpu_cuda_get_local_stream(),
                                       idxLevel, thisptr->kernels[starpu_worker_get_id()],
                                       lowerGroupObjCpu, upperGroupObjCpu,
                                       lowerGroupObjCuda, upperGroupObjCuda);
    }

    template<class ThisClass, class CellContainerClass>
    static void M2LCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        typename ThisClass::VecOfIndexes* indexesForGroup_first;
        unsigned char* groupCellsSrcData;
        size_t groupCellsSrcDataSize;
        unsigned char* groupCellsTgtData;
        size_t groupCellsTgtDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first,
                                   &groupCellsSrcData, &groupCellsSrcDataSize, &groupCellsTgtData, &groupCellsTgtDataSize);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* tgtLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t tgtLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass groupSrcCpu(groupCellsSrcData, groupCellsSrcDataSize, nullptr, 0, nullptr, 0);
        const CellContainerClass groupTargetCpu(groupCellsTgtData, groupCellsTgtDataSize, nullptr, 0, nullptr, 0);

        const CellContainerClass groupSrcCuda(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                          nullptr, 0, false);
        CellContainerClass groupTargetCuda(tgtData, tgtDataSize, nullptr, 0, tgtLocal, tgtLocalSize, false);

        thisptr->kernelWrapperCuda.M2LBetweenGroups(starpu_cuda_get_local_stream(),
                                                    idxLevel, thisptr->kernels[starpu_worker_get_id()],
                                                    groupTargetCpu, groupSrcCpu,
                                                    groupTargetCuda, groupSrcCuda, *indexesForGroup_first);
    }

    template<class ThisClass, class CellContainerClass>
    static void M2LInnerCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        typename ThisClass::VecOfIndexes* indexesForGroup_first;        
        unsigned char* groupCellsData;
        size_t groupCellsDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first, &groupCellsData, &groupCellsDataSize);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* srcLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t srcLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        const CellContainerClass currentGroupCpu(srcData, srcDataSize, nullptr, 0, nullptr, 0);

        CellContainerClass currentGroupCuda(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                            srcLocal, srcLocalSize, false);

        thisptr->kernelWrapperCuda.M2LInGroup(starpu_cuda_get_local_stream(),
                                              idxLevel, thisptr->kernels[starpu_worker_get_id()],
                                              currentGroupCpu, currentGroupCuda, *indexesForGroup_first);
    }

    template<class ThisClass, class CellContainerClass>
    static void L2LCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        long int idxLevel;
        unsigned char* groupCellsLowerData;
        size_t groupCellsLowerDataSize;
        unsigned char* groupCellsUpperData;
        size_t groupCellsUpperDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &groupCellsUpperData, &groupCellsUpperDataSize,
                                   &groupCellsLowerData, &groupCellsLowerDataSize);

        unsigned char* upperData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t upperDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* upperLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t upperLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* lowerData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t lowerDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* lowerLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t lowerLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass upperGroupObjCpu(groupCellsUpperData, groupCellsUpperDataSize, nullptr, 0, nullptr, 0);
        const CellContainerClass lowerGroupObjCpu(groupCellsLowerData, groupCellsLowerDataSize, nullptr, 0, nullptr, 0);

        const CellContainerClass upperGroupObjCuda(upperData, upperDataSize, nullptr, 0, upperLocal, upperLocalSize, false);
        CellContainerClass lowerGroupObjCuda(lowerData, lowerDataSize, nullptr, 0, lowerLocal, lowerLocalSize, false);

        thisptr->kernelWrapperCuda.L2L(starpu_cuda_get_local_stream(),
                                       idxLevel, thisptr->kernels[starpu_worker_get_id()],
                                       upperGroupObjCpu, lowerGroupObjCpu,
                                       upperGroupObjCuda, lowerGroupObjCuda);
    }

    template<class ThisClass, class CellContainerClass, class ParticleContainerClass>
    static void L2PCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        unsigned char* groupCellsData;
        size_t groupCellsDataSize;
        unsigned char* groupParticlesData;
        size_t groupParticlesDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr,  &groupCellsData, &groupCellsDataSize, &groupParticlesData, &groupParticlesDataSize);

        unsigned char* leafData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t leafDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* leafLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t leafLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* particleData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t particleDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* particleRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t particleRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClass leafGroupObjCpu(leafData, leafDataSize, nullptr, 0, nullptr, 0);
        const ParticleContainerClass particleGroupObjCpu(particleData, particleDataSize, nullptr, 0);

        const CellContainerClass leafGroupObjCuda(leafData, leafDataSize, nullptr, 0, leafLocal, leafLocalSize, false);
        ParticleContainerClass particleGroupObjCuda(particleData, particleDataSize, particleRhs, particleRhsSize, false);

        thisptr->kernelWrapperCuda.L2P(starpu_cuda_get_local_stream(),
                                       thisptr->kernels[starpu_worker_get_id()],
                                       leafGroupObjCpu, particleGroupObjCpu,
                                       leafGroupObjCuda, particleGroupObjCuda);
    }


    template<class ThisClass, class CellContainerClassSource, class CellContainerClassTarget>
    static void M2LTsmCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        int idxLevel;
        typename ThisClass::VecOfIndexes* indexesForGroup_first;
        unsigned char* groupCellsSrcData;
        size_t groupCellsSrcDataSize;
        unsigned char* groupCellsTgtData;
        size_t groupCellsTgtDataSize;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &idxLevel, &indexesForGroup_first,
                                   &groupCellsSrcData, &groupCellsSrcDataSize, &groupCellsTgtData, &groupCellsTgtDataSize);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* srcMultipole = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcMultipoleSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        unsigned char* tgtLocal = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t tgtLocalSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        const CellContainerClassSource groupSrcCpu(groupCellsSrcData, groupCellsSrcDataSize, nullptr, 0, nullptr, 0);
        const CellContainerClassTarget groupTargetCpu(groupCellsTgtData, groupCellsTgtDataSize, nullptr, 0, nullptr, 0);

        const CellContainerClassSource groupSrcCuda(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                                nullptr, 0, false);
        CellContainerClassTarget groupTargetCuda(tgtData, tgtDataSize, nullptr, 0, tgtLocal, tgtLocalSize, false);

        thisptr->kernelWrapperCuda.M2LBetweenGroups(starpu_cuda_get_local_stream(),
                                                    idxLevel, thisptr->kernels[starpu_worker_get_id()],
                                                    groupTargetCpu, groupSrcCpu,
                                                    groupTargetCuda, groupSrcCuda,
                                                    *indexesForGroup_first);
    }

    template<class ThisClass, class ParticleContainerClassSource, class ParticleContainerClassTarget>
    static void P2PTsmBetweenLeavesCallback(void *buffers[], void *cl_arg){
        ThisClass* thisptr;
        typename ThisClass::VecOfIndexes* indexesForGroup_first;
        unsigned char* srcDataCpu;
        size_t srcDataSizeCpu;
        unsigned char* tgtDataCpu;
        size_t tgtDataSizeCpu;
        starpu_codelet_unpack_args(cl_arg, &thisptr, &indexesForGroup_first,
                                   &srcDataCpu, &srcDataSizeCpu, &tgtDataCpu, &tgtDataSizeCpu);

        unsigned char* srcData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[0]);
        size_t srcDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[0]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        const ParticleContainerClassSource groupSrcCpu(srcDataCpu, srcDataSizeCpu,
                                           nullptr, 0, true);
        const ParticleContainerClassTarget groupTargetCpu(tgtDataCpu, tgtDataSizeCpu,
                                              nullptr, 0, true);

        const ParticleContainerClassSource groupSrc(srcData, srcDataSize,
                                                    nullptr, 0, false);
        ParticleContainerClassTarget groupTarget(tgtData, tgtDataSize,
                                                 tgtRhs, tgtRhsSize, false);

        thisptr->kernelWrapperCuda.P2PBetweenGroupsTsm(starpu_cuda_get_local_stream(),
                                                       thisptr->kernels[starpu_worker_get_id()],
                                                       groupSrcCpu, groupTargetCpu,
                                                       groupSrc, groupTarget, *indexesForGroup_first);
    }
};

#endif
