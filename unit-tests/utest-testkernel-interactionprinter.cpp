#include "testkernel-core.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "kernels/counterkernels/tbfinteractionprinter.hpp"

// You must do this
using AlgoTestClass = TestTestKernel<TbfAlgorithm<double, TbfInteractionPrinter<TbfTestKernel<double>>>>;
TestClass(AlgoTestClass)


