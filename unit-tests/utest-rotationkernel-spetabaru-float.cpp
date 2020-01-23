#include "rotationkernel-core.hpp"
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"

// You must do this
using AlgoTestClass = TestRotationKernel<float, TbfSmSpetabaruAlgorithm>;
TestClass(AlgoTestClass)
