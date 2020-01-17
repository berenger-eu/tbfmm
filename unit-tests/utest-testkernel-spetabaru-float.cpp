#include "testkernel-core.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"

// You must do this
using AlgoTestClass = TestTestKernel<TbfSmSpetabaruAlgorithm<float, TbfTestKernel<float>>>;
TestClass(AlgoTestClass)
