#ifndef TBFSMSTARPUCALLBACKS_HPP
#define TBFSMSTARPUCALLBACKS_HPP

#include <starpu.h>


class TbfSmStarpuCallbacks{
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

        CellContainerClass leafGroupObj(leafData, leafDataSize,
                                        leafMultipole, leafMultipoleSize,
                                        nullptr, 0);
        const ParticleContainerClass particleGroupObj(particleData, particleDataSize,
                                                      nullptr, 0);

        thisptr->kernelWrapper.P2M(thisptr->kernels[starpu_worker_get_id()], particleGroupObj, leafGroupObj);
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
        assert(srcDataCpu == srcData);
        assert(srcDataSize == srcDataSizeCpu);

        unsigned char* srcRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t srcRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);
        assert(tgtDataCpu == tgtData);
        assert(tgtDataSizeCpu == tgtDataSize);

        unsigned char* tgtRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[3]);
        size_t tgtRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[3]);

        ParticleContainerClass groupSrc(srcData, srcDataSize,
                                        srcRhs, srcRhsSize);
        ParticleContainerClass groupTarget(tgtData, tgtDataSize,
                                           tgtRhs, tgtRhsSize);

        thisptr->kernelWrapper.P2PBetweenGroups(thisptr->kernels[starpu_worker_get_id()], groupSrc, groupTarget, *indexesForGroup_first);
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
        assert(particleDataCpu == particleData);
        assert(particleDataSize == particleDataSizeCpu);

        unsigned char* particleRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t particleRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);

        ParticleContainerClass currentGroup(particleData, particleDataSize,
                                            particleRhs, particleRhsSize);

        thisptr->kernelWrapper.P2PInGroup(thisptr->kernels[starpu_worker_get_id()], currentGroup, *indexesForGroup_first);
        thisptr->kernelWrapper.P2PInner(thisptr->kernels[starpu_worker_get_id()], currentGroup);
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

        const CellContainerClass lowerGroupObj(lowerData, lowerDataSize, lowerMultipole, lowerMultipoleSize,
                                               nullptr, 0);
        CellContainerClass upperGroupObj(upperData, upperDataSize, upperMultipole, upperMultipoleSize,
                                         nullptr, 0);

        thisptr->kernelWrapper.M2M(idxLevel, thisptr->kernels[starpu_worker_get_id()], lowerGroupObj, upperGroupObj);
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

        const CellContainerClass groupSrc(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                          nullptr, 0);
        CellContainerClass groupTarget(tgtData, tgtDataSize, nullptr, 0, tgtLocal, tgtLocalSize);

        thisptr->kernelWrapper.M2LBetweenGroups(idxLevel, thisptr->kernels[starpu_worker_get_id()], groupTarget, groupSrc, *indexesForGroup_first);
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

        CellContainerClass currentGroup(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                        srcLocal, srcLocalSize);

        thisptr->kernelWrapper.M2LInGroup(idxLevel, thisptr->kernels[starpu_worker_get_id()], currentGroup, *indexesForGroup_first);
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

        const CellContainerClass upperGroupObj(upperData, upperDataSize, nullptr, 0, upperLocal, upperLocalSize);
        CellContainerClass lowerGroupObj(lowerData, lowerDataSize, nullptr, 0, lowerLocal, lowerLocalSize);

        thisptr->kernelWrapper.L2L(idxLevel, thisptr->kernels[starpu_worker_get_id()], upperGroupObj, lowerGroupObj);
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

        const CellContainerClass leafGroupObj(leafData, leafDataSize, nullptr, 0, leafLocal, leafLocalSize);
        ParticleContainerClass particleGroupObj(particleData, particleDataSize, particleRhs, particleRhsSize);

        thisptr->kernelWrapper.L2P(thisptr->kernels[starpu_worker_get_id()], leafGroupObj, particleGroupObj);
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

        const CellContainerClassSource groupSrc(srcData, srcDataSize, srcMultipole, srcMultipoleSize,
                                          nullptr, 0);
        CellContainerClassTarget groupTarget(tgtData, tgtDataSize, nullptr, 0, tgtLocal, tgtLocalSize);

        thisptr->kernelWrapper.M2LBetweenGroups(idxLevel, thisptr->kernels[starpu_worker_get_id()], groupTarget, groupSrc, *indexesForGroup_first);
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
        assert(srcDataCpu == srcData);
        assert(srcDataSize == srcDataSizeCpu);

        unsigned char* tgtData = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[1]);
        size_t tgtDataSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[1]);
        assert(tgtDataCpu == tgtData);
        assert(tgtDataSizeCpu == tgtDataSize);

        unsigned char* tgtRhs = (unsigned char*)STARPU_VARIABLE_GET_PTR(buffers[2]);
        size_t tgtRhsSize = STARPU_VARIABLE_GET_ELEMSIZE(buffers[2]);

        const ParticleContainerClassSource groupSrc(srcData, srcDataSize,
                                        nullptr, 0);
        ParticleContainerClassTarget groupTarget(tgtData, tgtDataSize,
                                           tgtRhs, tgtRhsSize);

        thisptr->kernelWrapper.P2PBetweenGroupsTsm(thisptr->kernels[starpu_worker_get_id()],
                                                   groupSrc, groupTarget, *indexesForGroup_first);
    }
};

#endif
