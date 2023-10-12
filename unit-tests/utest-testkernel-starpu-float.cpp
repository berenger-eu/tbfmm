#include "testkernel-core.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// -- END --

using AlgoTestClass = TestTestKernel<TbfSmStarpuAlgorithm<float, TbfTestKernel<float>>>;
TestClass(AlgoTestClass)
