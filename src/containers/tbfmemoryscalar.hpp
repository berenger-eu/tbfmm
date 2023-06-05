#ifndef TBFMEMORYSCALAR_HPP
#define TBFMEMORYSCALAR_HPP

#include "tbfglobal.hpp"

#include "tbfmemorydim.hpp"

#include <cassert>

template <class DataType_T, long int MemoryAlignementBytes = TbfDefaultMemoryAlignement>
class TbfMemoryScalar{
public:
    using DataType = DataType_T;
    using Width = TbfMemoryDim::Scalar;
    using Height = TbfMemoryDim::Scalar;
    using Major = TbfMemoryDim::Unused;
    constexpr static long int BlockMemoryAlignementBytes = MemoryAlignementBytes;

    template <class FuncType>
    static void ApplyToAllElements(DataType* inPtrToData, const long int inNbItems, FuncType&& inFunc){
        assert(inNbItems == 1);
        (void)inNbItems;
        inFunc(*inPtrToData);
    }

    template <class FuncType>
    static void ApplyToAllElementsConst(const DataType* inPtrToData, const long int inNbItems, FuncType&& inFunc){
        assert(inNbItems == 1);
        (void)inNbItems;
        inFunc(*inPtrToData);
    }

    static long int GetMemorySizeFromNbItems(const long int inNbItems){
        assert(inNbItems == 1);
        (void)inNbItems;
        const long int leadingDim = TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes);
        return  leadingDim;
    }

    class Viewer{
        DataType* ptrToData;
    public:
        __device__ __host__ explicit Viewer(DataType* inPtrToData, const long int inNbItems) : ptrToData(inPtrToData){
            assert(inNbItems == 1);
            (void)inNbItems;
        }

        __device__ __host__ DataType& getItem(){
            return *ptrToData;
        }
    };

    class ViewerConst{
        const DataType* ptrToData;
    public:
        __device__ __host__ explicit ViewerConst(const DataType* inPtrToData, const long int inNbItems) : ptrToData(inPtrToData){
            assert(inNbItems == 1);
            (void)inNbItems;
        }

        __device__ __host__ const DataType& getItem() const{
            return *ptrToData;
        }
    };
};

#endif

