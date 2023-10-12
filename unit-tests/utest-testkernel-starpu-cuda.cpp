#include "testkernel-core.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithmcuda.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// @TBF_USE_CUDA
// -- END --

using AlgoTestClass = TestTestKernel<TbfSmStarpuAlgorithmCuda<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)
