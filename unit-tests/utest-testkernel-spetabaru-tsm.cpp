#include "testkernel-core-tsm.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithmtsm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_SPETABARU
// -- END --

// You must do this
using AlgoTestClass = TestTestKernelTsm<TbfSmSpetabaruAlgorithmTsm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)


