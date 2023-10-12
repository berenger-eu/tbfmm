#ifndef TBFMEMORYVECTOR_HPP
#define TBFMEMORYVECTOR_HPP

#include "tbfglobal.hpp"

#include "tbfmemorydim.hpp"

template <class DataType_T, long int MemoryAlignementBytes = TbfDefaultMemoryAlignement>
class TbfMemoryVector{
public:
    using DataType = DataType_T;
    using Width = TbfMemoryDim::Variable;
    using Height = TbfMemoryDim::Scalar;
    using Major = TbfMemoryDim::Unused;
    constexpr static long int BlockMemoryAlignementBytes = MemoryAlignementBytes;

    template <class FuncType>
    static void ApplyToAllElements(DataType* inPtrToData, const long int inNbItems, FuncType&& inFunc){
        for(long int idx = 0 ; idx < inNbItems ; ++idx){
            inFunc(inPtrToData[idx]);
        }
    }

    template <class FuncType>
    static void ApplyToAllElementsConst(const DataType* inPtrToData, const long int inNbItems, FuncType&& inFunc){
        for(long int idx = 0 ; idx < inNbItems ; ++idx){
            inFunc(inPtrToData[idx]);
        }
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    static long int GetMemorySizeFromNbItems(const long int inNbItems){
        const long int leadingDim = TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes);
        return leadingDim;
    }

    class Viewer{
        DataType* ptrToData;
        const long int nbItems;
    public:
        #ifdef __NVCC__
                    __device__ __host__
        #endif
        explicit Viewer(DataType* inPtrToData, const long int inNbItems)
            : ptrToData(inPtrToData), nbItems(inNbItems){}
        #ifdef __NVCC__
                __device__ __host__
        #endif
        DataType& getItem(const long int inIdx){
            return ptrToData[inIdx];
        }
        #ifdef __NVCC__
                __device__ __host__
        #endif
        long int getNbItems() const{
            return nbItems;
        }
        #ifdef __NVCC__
                        __device__ __host__
        #endif
        long int size() const{
            return getNbItems();
        }
        #ifdef __NVCC__
                __device__ __host__
        #endif
        DataType& front(){
            return ptrToData[0];
        }
        #ifdef __NVCC__
                __device__ __host__
        #endif
        DataType& back(){
            return ptrToData[getNbItems()-1];
        }
    };


    class ViewerConst{
        const DataType* ptrToData;
        const long int nbItems;
    public:
        #ifdef __NVCC__
                __device__ __host__
        #endif
        explicit ViewerConst(const DataType* inPtrToData, const long int inNbItems)
            : ptrToData(inPtrToData), nbItems(inNbItems){}

        #ifdef __NVCC__
                __device__ __host__
        #endif
        const DataType& getItem(const long int inIdx){
            return ptrToData[inIdx];
        }

        #ifdef __NVCC__
                __device__ __host__
        #endif
        long int getNbItems() const{
            return nbItems;
        }

        #ifdef __NVCC__
                __device__ __host__
        #endif
        long int size() const{
            return getNbItems();
        }
        #ifdef __NVCC__
                __device__ __host__
        #endif
        const DataType& front(){
            return ptrToData[0];
        }
        #ifdef __NVCC__
        __device__ __host__
        #endif
        const DataType& back(){
            return ptrToData[getNbItems()-1];
        }
    };
};

#endif

