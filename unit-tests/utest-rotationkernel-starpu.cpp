#include "rotationkernel-core.hpp"
#include "algorithms/smstarpu/tbfsmstarpualgorithm.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"


// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_STARPU
// -- END --

// You must do this
using AlgoTestClass = TestRotationKernel<double, TbfSmStarpuAlgorithm>;
TestClass(AlgoTestClass)


