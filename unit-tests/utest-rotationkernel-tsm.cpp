#include "rotationkernel-core-tsm.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"

// You must do this
using AlgoTestClass = TestRotationKernelTsm<double, TbfAlgorithmTsm>;
TestClass(AlgoTestClass)


