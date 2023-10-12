#include "testkernel-core-tsm.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithmtsm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// -- END --

using AlgoTestClass = TestTestKernelTsm<TbfSmStarpuAlgorithmTsm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)

