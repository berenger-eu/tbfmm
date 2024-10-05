#ifndef TBFALGORITHMSELECTER_HPP
#define TBFALGORITHMSELECTER_HPP

#include "tbfglobal.hpp"

#include "algorithms/sequential/tbfalgorithm.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"
#ifdef TBF_USE_SPECX
#include "algorithms/smspecx/tbfsmspecxalgorithm.hpp"
#include "algorithms/smspecx/tbfsmspecxalgorithmtsm.hpp"
#ifdef TBF_USE_CUDA
#include "algorithms/smspecx/tbfsmspecxalgorithmcuda.hpp"
#endif
#endif
#ifdef TBF_USE_OPENMP
#include "algorithms/openmp/tbfopenmpalgorithm.hpp"
#include "algorithms/openmp/tbfopenmpalgorithmtsm.hpp"
#endif
#ifdef TBF_USE_STARPU
#include "algorithms/smstarpu/tbfsmstarpualgorithm.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithmtsm.hpp"
#ifdef TBF_USE_CUDA
#include "algorithms/smstarpu/tbfsmstarpualgorithmcuda.hpp"
#endif
#endif

struct TbfAlgorithmSelecter{
    template<typename RealType, class KernelClass, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType>>
#ifdef TBF_USE_STARPU
#ifndef TBF_USE_CUDA
    using type = TbfSmStarpuAlgorithm<RealType, KernelClass, SpaceIndexType>;
#else
    using type = TbfSmStarpuAlgorithmCuda<RealType, KernelClass, SpaceIndexType>;
#endif
//#elif defined(TBF_USE_SPECX)
//#ifndef TBF_USE_CUDA
//    using type = TbfSmSpecxAlgorithm<RealType, KernelClass, SpaceIndexType>;
//#else
//    using type = TbfSmSpecxAlgorithmCuda<RealType, KernelClass, SpaceIndexType>;
//#endif
#elif defined(TBF_USE_OPENMP)
    using type = TbfOpenmpAlgorithm<RealType, KernelClass, SpaceIndexType>;
#else
    using type = TbfAlgorithm<RealType, KernelClass, SpaceIndexType>;
#endif
};

struct TbfAlgorithmSelecterTsm{
    template<typename RealType, class KernelClass, class SpaceIndexType = TbfDefaultSpaceIndexType<RealType>>
#ifdef TBF_USE_STARPU
    using type = TbfSmStarpuAlgorithmTsm<RealType, KernelClass, SpaceIndexType>;
//#elif defined(TBF_USE_SPECX)
//    using type = TbfSmSpecxAlgorithmTsm<RealType, KernelClass, SpaceIndexType>;
#elif defined(TBF_USE_OPENMP)
    using type = TbfOpenmpAlgorithmTsm<RealType, KernelClass, SpaceIndexType>;
#else
    using type = TbfAlgorithmTsm<RealType, KernelClass, SpaceIndexType>;
#endif
};

#endif

