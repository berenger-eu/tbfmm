#include "unifkernel-core-tsm.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_FFTW
// -- END --

// You must do this
using AlgoTestClass = TestUnifKernelTsm<double, TbfAlgorithmTsm>;
TestClass(AlgoTestClass)


