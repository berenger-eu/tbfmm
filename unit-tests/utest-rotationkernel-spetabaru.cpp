#include "rotationkernel-core.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_SPETABARU
// -- END --

// You must do this
using AlgoTestClass = TestRotationKernel<double, TbfSmSpetabaruAlgorithm>;
TestClass(AlgoTestClass)
