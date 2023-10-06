#ifndef TBFMEMORYMULTIRVECTOR_HPP
#define TBFMEMORYMULTIRVECTOR_HPP

#include "tbfglobal.hpp"

#include "tbfmemorydim.hpp"

template <class DataType_T, long int NbRows, long int MemoryAlignementBytes = TbfDefaultMemoryAlignement>
class TbfMemoryMultiRVector{
public:
    static_assert(((sizeof(DataType_T)/TbfDefaultMemoryAlignement)*TbfDefaultMemoryAlignement == sizeof(DataType_T))
                  || ((TbfDefaultMemoryAlignement/sizeof(DataType_T))*sizeof(DataType_T) == TbfDefaultMemoryAlignement),
                  "Alignement is incorrect regarding DataType size");

    using DataType = DataType_T;
    using Width = TbfMemoryDim::Variable;
    using Height = TbfMemoryDim::Fixed<NbRows>;
    using Major = TbfMemoryDim::RowMajor;
    constexpr static long int BlockMemoryAlignementBytes = MemoryAlignementBytes;

    template <class FuncType>
    static void ApplyToAllElements(DataType* inPtrToData, const long int inNbItems, FuncType&& inFunc){
        const long int leadingDim = TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes);
        for(long int idxRow = 0 ; idxRow < NbRows ; ++idxRow){
            DataType* inPtrToDataRow = reinterpret_cast<DataType*>(reinterpret_cast<unsigned char*>(inPtrToData)+ idxRow*leadingDim);
            for(long int idx = 0 ; idx < inNbItems ; ++idx){
                inFunc(inPtrToDataRow[idx]);
            }
        }
    }

    template <class FuncType>
    static void ApplyToAllElementsConst(const DataType* inPtrToData, const long int inNbItems, FuncType&& inFunc){
        const long int leadingDim = TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes);
        for(long int idxRow = 0 ; idxRow < NbRows ; ++idxRow){
            const DataType* inPtrToDataRow = reinterpret_cast<const DataType*>(reinterpret_cast<const unsigned char*>(inPtrToData)+ idxRow*leadingDim);
            for(long int idx = 0 ; idx < inNbItems ; ++idx){
                inFunc(inPtrToDataRow[idx]);
            }
        }
    }
#ifdef __NVCC__
    __device__ __host__
#endif
    static long int GetMemorySizeFromNbItems(const long int inNbItems){
        const long int leadingDim = TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes);
        return NbRows*leadingDim;
    }

    class Viewer{
        DataType* ptrToData;
        const long int nbItems;
        const long int leadingDim;
    public:
        #ifdef __NVCC__
                __device__ __host__
        #endif
        explicit Viewer(DataType* inPtrToData, const long int inNbItems)
            : ptrToData(inPtrToData), nbItems(inNbItems),
              leadingDim(TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes)){}
        #ifdef __NVCC__
                __device__ __host__
        #endif
        DataType& getItem(const long int inIdx, const long int inIdxRow){
            DataType* ptrToDataRow = reinterpret_cast<DataType*>(reinterpret_cast<unsigned char*>(ptrToData)+ inIdxRow*leadingDim);
            return ptrToDataRow[inIdx];
        }
        #ifdef __NVCC__
                __device__ __host__
        #endif
        long int getNbItems() const{
            return nbItems;
        }
    };


    class ViewerConst{
        const DataType* ptrToData;
        const long int nbItems;
        const long int leadingDim;
    public:
        #ifdef __NVCC__
                __device__ __host__
        #endif
        explicit ViewerConst(const DataType* inPtrToData, const long int inNbItems)
            : ptrToData(inPtrToData), nbItems(inNbItems),
              leadingDim(TbfUtils::GetLeadingDim<DataType>(inNbItems, MemoryAlignementBytes)){}

        #ifdef __NVCC__
                __device__ __host__
        #endif
        const DataType& getItem(const long int inIdx, const long int inIdxRow){
            const DataType* ptrToDataRow = reinterpret_cast<const DataType*>(reinterpret_cast<const unsigned char*>(ptrToData)+ inIdxRow*leadingDim);
            return ptrToDataRow[inIdx];
        }

        #ifdef __NVCC__
                __device__ __host__
        #endif
        long int getNbItems() const{
            return nbItems;
        }
    };
};

#endif

