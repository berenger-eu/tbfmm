#include "testkernel-core.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_SPETABARU
// -- END --

// You must do this
using AlgoTestClass = TestTestKernel<TbfSmSpetabaruAlgorithm<float, TbfTestKernel<float>>>;
TestClass(AlgoTestClass)
