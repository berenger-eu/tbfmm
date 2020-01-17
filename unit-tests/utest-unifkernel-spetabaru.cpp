#include "unifkernel-core.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

// You must do this
using AlgoTestClass = TestUnifKernel<double, TbfSmSpetabaruAlgorithm>;
TestClass(AlgoTestClass)
