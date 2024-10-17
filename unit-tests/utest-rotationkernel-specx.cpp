#include "rotationkernel-core.hpp"
#include "algorithms/smspecx/tbfsmspecxalgorithm.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_SPECX
// -- END --

// You must do this
using AlgoTestClass = TestRotationKernel<double, TbfSmSpecxAlgorithm>;
TestClass(AlgoTestClass)
