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
    TbfVectorView(const std::vector<ElementType_T>& inVector, const long int inOffset, const long int inLength)
        : vector(inVector), offset(inOffset), length(inLength){
        assert(offset+length <= static_cast<long int>(vector.size()));
    }

    long int size() const{
        return length;
    }

    const ElementType& operator[](const long int inIndex) const{
        assert(inIndex < length);
        return vector[offset+inIndex];
    }
};

template <class ElementType_T>
auto TbfMakeVectorView(const std::vector<ElementType_T>& inVector, const long int inOffset, const long int inLength){
    return TbfVectorView<ElementType_T>(inVector, inOffset, inLength);
}

#endif
