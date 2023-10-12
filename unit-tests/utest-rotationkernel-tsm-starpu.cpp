#include "rotationkernel-core-tsm.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithm.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// -- END --

// You must do this
using AlgoTestClass = TestRotationKernelTsm<double, TbfSmStarpuAlgorithm>;
TestClass(AlgoTestClass)


