#ifndef TBFMEMORYBLOCK_HPP
#define TBFMEMORYBLOCK_HPP

#include "tbfglobal.hpp"

#include <tuple>
#include <array>
#include <memory>
#include <cassert>
#include <cstring>

template <class ... BlockDefinitions>
class TbfMemoryBlock{

    static_assert( TbfUtils::all_true<std::is_pod<BlockDefinitions>::value ... >::value, "All types must be POD");

    constexpr static long int NbBlocks = sizeof...(BlockDefinitions);
    static_assert(NbBlocks >= 1, "There must be at least one block");
    static_assert(NbBlocks == std::tuple_size<std::tuple<BlockDefinitions ...>>::value, "Must be the same or something is wrong...");

    using TupleOfBlockDefinitions = std::tuple<typename std::decay<BlockDefinitions>::type ...>;

    template <class SizeContainerClass>
    constexpr static std::array<std::pair<long int, long int>, NbBlocks + 1> GetSizeAndOffsetOfBlocks(const SizeContainerClass& inSizes){
        std::array<long int, NbBlocks> sizeOfBlocks = {};
        std::array<long int, NbBlocks> alignementOfBlocks = {};

        TbfUtils::for_each<TupleOfBlockDefinitions>([&](auto structWithBlockType, auto idxBlock){
            using BlockType = typename decltype(structWithBlockType)::BlockTypeT;
            sizeOfBlocks[idxBlock] = BlockType::GetMemorySizeFromNbItems(inSizes[idxBlock]);
            alignementOfBlocks[idxBlock] = BlockType::BlockMemoryAlignementBytes;
        });

        std::array<std::pair<long int, long int>, NbBlocks + 1> sizeAndOffset;

        sizeAndOffset[0].first = sizeOfBlocks[0];
        sizeAndOffset[0].second = 0;

        for(long int idxBlock = 1 ; idxBlock < static_cast<long int>(inSizes.size()) ; ++idxBlock){
            sizeAndOffset[idxBlock].first = sizeOfBlocks[idxBlock];
            sizeAndOffset[idxBlock].second = (sizeAndOffset[idxBlock-1].first + sizeAndOffset[idxBlock-1].second);
        }

        sizeAndOffset[NbBlocks].first = 0;
        sizeAndOffset[NbBlocks].second = (sizeAndOffset[NbBlocks-1].first + sizeAndOffset[NbBlocks-1].second);

        return sizeAndOffset;
    }

    long int allocatedMemorySizeInByte;
    std::unique_ptr<unsigned char[]> rawMemoryPtr;
    long int* nbItemsInBlocks;
    long int* offsetOfBlocksForPtrs;
    std::array<unsigned char*,NbBlocks> blockRawPtrs;
    bool objectOwnData;

    void constructAllItems(){
        applyToAllElements([](auto& inItem){
            static_assert (std::is_reference<decltype(inItem)>::value, "Should be a ref here");
            using ItemType = typename std::decay<decltype(inItem)>::type;
            new(&inItem) ItemType();
        });
    }

    void freeAllItems(){
        assert(objectOwnData);

        applyToAllElements([](auto& inItem){
            static_assert (std::is_reference<decltype(inItem)>::value, "Should be a ref here");
            using ItemType = typename std::decay<decltype(inItem)>::type;
            inItem.~ItemType();
        });
    }

public:
    TbfMemoryBlock()
        : allocatedMemorySizeInByte(0), rawMemoryPtr(nullptr), nbItemsInBlocks(nullptr),
        offsetOfBlocksForPtrs(nullptr), objectOwnData(false){
        for(auto& blockPtr : blockRawPtrs){
            blockPtr = nullptr;
        }
    }

    ~TbfMemoryBlock(){
        if(objectOwnData == false){
            rawMemoryPtr.release();
        }
        else{
            freeAllItems();
        }
    }

    TbfMemoryBlock(const TbfMemoryBlock&) = delete;
    TbfMemoryBlock& operator=(const TbfMemoryBlock&) = delete;

    TbfMemoryBlock(TbfMemoryBlock&& other) : TbfMemoryBlock(){
        (*this) = std::move(other);
    }

    TbfMemoryBlock& operator=(TbfMemoryBlock&& other){
        allocatedMemorySizeInByte = other.allocatedMemorySizeInByte;
        rawMemoryPtr = std::move(other.rawMemoryPtr);
        nbItemsInBlocks = other.nbItemsInBlocks;
        offsetOfBlocksForPtrs = other.offsetOfBlocksForPtrs;
        blockRawPtrs = other.blockRawPtrs;
        objectOwnData = other.objectOwnData;

        other.allocatedMemorySizeInByte = 0;
        other.nbItemsInBlocks = nullptr;
        other.offsetOfBlocksForPtrs = nullptr;
        for(auto& blockPtr : other.blockRawPtrs){
            blockPtr = nullptr;
        }
        assert(other.rawMemoryPtr.get() == nullptr);
        other.objectOwnData = false;

        return *this;
    }

    template <class SizeContainerClass>
    explicit TbfMemoryBlock(SizeContainerClass&& inNbItemsInBlocks) : TbfMemoryBlock(){
        resetBlocksFromSizes(std::forward<SizeContainerClass>(inNbItemsInBlocks));
    }

    explicit TbfMemoryBlock(unsigned char* inRawMemoryPtr, const long int inBlockSizeInByte)
        : allocatedMemorySizeInByte(inBlockSizeInByte), rawMemoryPtr(inRawMemoryPtr), nbItemsInBlocks(nullptr),
        offsetOfBlocksForPtrs(nullptr), objectOwnData(false){

        if(rawMemoryPtr == nullptr){
            assert(allocatedMemorySizeInByte == 0);
            for(auto& blockPtr : blockRawPtrs){
                blockPtr = nullptr;
            }
        }
        else{
            nbItemsInBlocks = reinterpret_cast<long int*>(&rawMemoryPtr[allocatedMemorySizeInByte] - (sizeof(long int) * NbBlocks));
            offsetOfBlocksForPtrs = reinterpret_cast<long int*>(&rawMemoryPtr[allocatedMemorySizeInByte] - (sizeof(long int) * NbBlocks)
                                                            - (sizeof(long int) * NbBlocks));

            for(long int idxPtr = 0 ; idxPtr < NbBlocks ; ++idxPtr){
                blockRawPtrs[idxPtr] = &rawMemoryPtr[offsetOfBlocksForPtrs[idxPtr]];
            }
        }
    }

    template <class SizeContainerClass>
    void resetBlocksFromSizes(SizeContainerClass&& inNbItemsInBlocks){
        if(objectOwnData){
            freeAllItems();
        }

        // We allocate, so we will have to deallocate too
        objectOwnData = true;

        const std::array<std::pair<long int, long int>, NbBlocks + 1> sizeAndOffsetOfBlocks = GetSizeAndOffsetOfBlocks(inNbItemsInBlocks);
        const long int totalMemoryToAlloc = (sizeAndOffsetOfBlocks[NbBlocks].second
                                            + sizeof(long int) * NbBlocks
                                            + sizeof(long int) * NbBlocks);

        if(allocatedMemorySizeInByte < totalMemoryToAlloc){
            rawMemoryPtr.reset(new unsigned char[totalMemoryToAlloc]);
            allocatedMemorySizeInByte = totalMemoryToAlloc;
        }
        memset(rawMemoryPtr.get(), 0, totalMemoryToAlloc);

        nbItemsInBlocks = reinterpret_cast<long int*>(&rawMemoryPtr[allocatedMemorySizeInByte] - (sizeof(long int) * NbBlocks));
        offsetOfBlocksForPtrs = reinterpret_cast<long int*>(&rawMemoryPtr[allocatedMemorySizeInByte] - (sizeof(long int) * NbBlocks)
                                                        - (sizeof(long int) * NbBlocks));

        for(long int idxBlock = 0 ; idxBlock < NbBlocks ; ++idxBlock){
            nbItemsInBlocks[idxBlock] = inNbItemsInBlocks[idxBlock];
            offsetOfBlocksForPtrs[idxBlock] = sizeAndOffsetOfBlocks[idxBlock].second;
            blockRawPtrs[idxBlock] = &rawMemoryPtr[offsetOfBlocksForPtrs[idxBlock]];
        }

        constructAllItems();
    }

    #ifdef __NVCC__
    __device__ __host__
    #endif
    bool isEmpty() const {
        return !nbItemsInBlocks;
    }

    unsigned char* getPtr(){
        return rawMemoryPtr.get();
    }

    const unsigned char* getPtr() const{
        return rawMemoryPtr.get();
    }

    //////////////////////////////////////////////////////////////////////

    template <class FuncType>
    void applyToAllElements(FuncType&& inFunc){
        assert(rawMemoryPtr != nullptr);
        if(nbItemsInBlocks){
            TbfUtils::for_each<TupleOfBlockDefinitions>([&](auto structWithBlockType, auto idxBlock){
                using BlockType = typename decltype(structWithBlockType)::BlockTypeT;
                BlockType::ApplyToAllElements(reinterpret_cast<typename BlockType::DataType*>(blockRawPtrs[idxBlock]),
                                              nbItemsInBlocks[idxBlock], inFunc);
            });
        }
    }

    template <long int IdxBlock, class FuncType>
    auto applyToBlock(FuncType&& inFunc){
        if(nbItemsInBlocks){
            using BlockType = typename std::tuple_element<IdxBlock, TupleOfBlockDefinitions>::type;
            BlockType::ApplyToAllElements(reinterpret_cast<typename BlockType::DataType>(blockRawPtrs[IdxBlock]),
                                          nbItemsInBlocks[IdxBlock], inFunc);
        }
    }

    template <long int IdxBlock>
    #ifdef __NVCC__
    __device__ __host__
    #endif
        auto getViewerForBlock(){
        static_assert(IdxBlock < NbBlocks, "Index of block out of range");

        using BlockType = typename std::tuple_element<IdxBlock, TupleOfBlockDefinitions>::type;
        if(nbItemsInBlocks){
            return typename BlockType::Viewer(reinterpret_cast<typename BlockType::DataType*>(blockRawPtrs[IdxBlock]),
                                                 nbItemsInBlocks[IdxBlock]);
        }
        else{
            return typename BlockType::Viewer(nullptr, 0);
        }
    }

    //////////////////////////////////////////////////////////////////////

    template <class FuncType>
    void applyToAllElementsConst(FuncType&& inFunc) const {
        if(nbItemsInBlocks){
            TbfUtils::for_each<TupleOfBlockDefinitions>([&](auto structWithBlockType, auto idxBlock){
                using BlockType = typename decltype(structWithBlockType)::BlockTypeT;
                BlockType::ApplyToAllElementsConst(reinterpret_cast<typename BlockType::DataType*>(blockRawPtrs[idxBlock]),
                                              nbItemsInBlocks[idxBlock], inFunc);
            });
        }
    }

    template <long int IdxBlock, class FuncType>
    auto applyToBlockConst(FuncType&& inFunc) const {
        if(nbItemsInBlocks){
            using BlockType = typename std::tuple_element<IdxBlock, TupleOfBlockDefinitions>::type;
            BlockType::ApplyToAllElementsConst(reinterpret_cast<typename BlockType::DataType>(blockRawPtrs[IdxBlock]),
                                          nbItemsInBlocks[IdxBlock], inFunc);
        }
    }

    template <long int IdxBlock>
    #ifdef __NVCC__
    __device__ __host__
    #endif
        auto getViewerForBlockConst() const {
        static_assert(IdxBlock < NbBlocks, "Index of block out of range");

        using BlockType = typename std::tuple_element<IdxBlock, TupleOfBlockDefinitions>::type;
        if(nbItemsInBlocks){
            return typename BlockType::ViewerConst(reinterpret_cast<typename BlockType::DataType*>(blockRawPtrs[IdxBlock]),
                                                 nbItemsInBlocks[IdxBlock]);
        }
        else{
            return typename BlockType::ViewerConst(nullptr, 0);
        }
    }

#ifdef __NVCC__
    __device__ __host__
#endif
    auto getAllocatedMemorySizeInByte() const{
        return allocatedMemorySizeInByte;
    }
};


#endif

