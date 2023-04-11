#include "testkernel-core.hpp"
#include "algorithms/smspecx/tbfsmspecxalgorithm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_SPECX
// -- END --

// You must do this
using AlgoTestClass = TestTestKernel<TbfSmSpecxAlgorithm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)
