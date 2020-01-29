#include "testkernel-core.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "kernels/counterkernels/tbfinteractiontimer.hpp"

// You must do this
using AlgoTestClass = TestTestKernel<TbfAlgorithm<double, TbfInteractionTimer<TbfTestKernel<double>>>>;
TestClass(AlgoTestClass)


