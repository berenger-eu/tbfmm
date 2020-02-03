#ifndef TBFALGORITHMSELECTER_HPP
#define TBFALGORITHMSELECTER_HPP

#include "tbfglobal.hpp"

#include "algorithms/sequential/tbfalgorithm.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"
#ifdef TBF_USE_SPETABARU
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#endif
#ifdef TBF_USE_OPENMP
#include "algorithms/openmp/tbfopenmpalgorithm.hpp"
#endif

struct TbfAlgorithmSelecter{
    template<typename RealType, class KernelClass>
#ifdef TBF_USE_SPETABARU
    using type = TbfSmSpetabaruAlgorithm<RealType, KernelClass>;
#elif defined(TBF_USE_OPENMP)
    using type = TbfOpenmpAlgorithm<RealType, KernelClass>;
#else
    using type = TbfAlgorithm<RealType, KernelClass>;
#endif
};

struct TbfAlgorithmSelecterTsm{
    template<typename RealType, class KernelClass>
    using type = TbfAlgorithmTsm<RealType, KernelClass>;
};

#endif

