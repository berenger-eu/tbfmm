#include "rotationkernel-core.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithmcuda.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"


// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// @TBF_USE_CUDA
// -- END --

// You must do this
using AlgoTestClass = TestRotationKernel<double, TbfSmStarpuAlgorithmCuda>;
TestClass(AlgoTestClass)


