#include <iostream>
#include <vector>
#include <iterator>
#include <memory>

#if __cplusplus >= 201703L
#if _MSC_VER
#pragma warning C++17 seems supported
#else
#warning C++17 seems supported
#endif
#else
#if _MSC_VER
#pragma warning C++17 seems NOT supported
#else
#warning C++17 seems NOT supported
#endif
#endif

int main() 
{
    std::vector<int> v;
    std::cout << std::size(v) << '\n';
    
    std::unique_ptr<int[]> ptr(new int[2]);
    ptr.reset(nullptr);
    
    return 0;
}
