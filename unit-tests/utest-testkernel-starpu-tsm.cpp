#include "testkernel-core-tsm.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// -- END --

using AlgoTestClass = TestTestKernelTsm<TbfSmStarpuAlgorithm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)

