#include "testkernel-core-tsm.hpp"
#include "algorithms/openmp/tbfopenmpalgorithmtsm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_OPENMP
// -- END --

#ifdef __GNUC__
int main(){}
#else
// You must do this
using AlgoTestClass = TestTestKernelTsm<TbfOpenmpAlgorithmTsm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)
#endif

