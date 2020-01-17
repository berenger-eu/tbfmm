#include "unifkernel-core.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "kernels/unifkernel/FUnifKernel.hpp"

// You must do this
using AlgoTestClass = TestUnifKernel<double, TbfAlgorithm>;
TestClass(AlgoTestClass)


