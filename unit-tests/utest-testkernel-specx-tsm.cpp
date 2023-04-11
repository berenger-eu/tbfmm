#include "testkernel-core-tsm.hpp"
#include "algorithms/smspecx/tbfsmspecxalgorithmtsm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_SPECX
// -- END --

// You must do this
using AlgoTestClass = TestTestKernelTsm<TbfSmSpecxAlgorithmTsm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)


