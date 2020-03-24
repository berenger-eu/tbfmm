#include "testkernel-core.hpp"
#include "algorithms/openmp/tbfopenmpalgorithm.hpp"

// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_OPENMP
// -- END --

#ifdef __GNUC__
int main(){}
#else
// You must do this
using AlgoTestClass = TestTestKernel<TbfOpenmpAlgorithm<double, TbfTestKernel<double>>>;
TestClass(AlgoTestClass)
#endif
