#include "unifkernel-core.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_FFTW
// -- END --

// You must do this
using AlgoTestClass = TestUnifKernel<float, TbfAlgorithm>;
TestClass(AlgoTestClass)


