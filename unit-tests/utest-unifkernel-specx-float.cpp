#include "unifkernel-core.hpp"
#include "algorithms/smspecx/tbfsmspecxalgorithm.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_FFTW
// @TBF_USE_SPECX
// -- END --

// You must do this
using AlgoTestClass = TestUnifKernel<float, TbfSmSpecxAlgorithm>;
TestClass(AlgoTestClass)
