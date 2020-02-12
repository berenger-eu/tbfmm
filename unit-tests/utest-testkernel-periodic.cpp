#include "testkernel-core-periodic.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"

// You must do this
using AlgoTestClass = TestTestKernelPeriodic<
                                        TbfAlgorithm<double,
                                                    TbfTestKernel<double, TbfDefaultSpaceIndexTypePeriodic<double>>,
                                                    TbfDefaultSpaceIndexTypePeriodic<double>
                                        >
                        >;
TestClass(AlgoTestClass)


