#include "unifkernel-core.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_FFTW
// @TBF_USE_SPETABARU
// -- END --

// You must do this
using AlgoTestClass = TestUnifKernel<float, TbfSmSpetabaruAlgorithm>;
TestClass(AlgoTestClass)
