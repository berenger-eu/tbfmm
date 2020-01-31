#include "testkernel-core-tsm.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"

// You must do this
using AlgoTestClass = TestTestKernelTsm<TbfAlgorithmTsm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)


