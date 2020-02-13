#include "testkernel-core-periodic.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "kernels/counterkernels/tbfinteractionprinter.hpp"

// You must do this
using AlgoTestClass = TestTestKernelPeriodic<TbfAlgorithm<double, TbfInteractionPrinter<TbfTestKernel<double, TbfDefaultSpaceIndexTypePeriodic<double>>>,
TbfDefaultSpaceIndexTypePeriodic<double>>>;
TestClass(AlgoTestClass)


