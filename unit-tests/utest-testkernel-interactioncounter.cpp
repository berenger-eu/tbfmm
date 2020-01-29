#include "testkernel-core.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "kernels/counterkernels/tbfinteractioncounter.hpp"

// You must do this
using AlgoTestClass = TestTestKernel<TbfAlgorithm<double, TbfInteractionCounter<TbfTestKernel<double>>>>;
TestClass(AlgoTestClass)


