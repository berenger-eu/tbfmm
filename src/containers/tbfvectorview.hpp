#ifndef TBFVECTORVIEW_HPP
#define TBFVECTORVIEW_HPP

#include "tbfglobal.hpp"

#include "tbfmemorydim.hpp"

#include <vector>
#include <cassert>

template <class ElementType_T>
class TbfVectorView{
public:
    using ElementType = ElementType_T;

private:
    const std::vector<ElementType_T>& vector;
    const long int offset;
    const long int length;
public:
#ifdef __NVCC__
    __device__ __host__
#endif
    TbfVectorView(const std::vector<ElementType_T>& inVector, const long int inOffset, const long int inLength)
        : vector(inVector), offset(inOffset), length(inLength){
        assert(offset+length <= static_cast<long int>(vector.size()));
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    long int size() const{
        return length;
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    const ElementType& operator[](const long int inIndex) const{
        assert(inIndex < length);
        return vector[offset+inIndex];
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    explicit operator std::vector<ElementType_T>() const{
        return toStdVector();
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    std::vector<ElementType_T> toStdVector() const{
        return std::vector<ElementType_T>(vector.begin() + offset,
                                   vector.begin() + offset + length);
    }
};

template <class ElementType_T>
#ifdef __NVCC__
__device__ __host__
#endif
auto TbfMakeVectorView(const std::vector<ElementType_T>& inVector, const long int inOffset, const long int inLength){
    return TbfVectorView<ElementType_T>(inVector, inOffset, inLength);
}

#endif
