#include "testkernel-core-periodic-tsm.hpp"
#include "algorithms/sequential/tbfalgorithmtsm.hpp"

// You must do this
using AlgoTestClass = TestTestKernelPeriodicTsm<
                                        TbfAlgorithmTsm<double,
                                                    TbfTestKernel<double, TbfDefaultSpaceIndexTypePeriodic<double>>,
                                                    TbfDefaultSpaceIndexTypePeriodic<double>
                                        >
                        >;
TestClass(AlgoTestClass)


