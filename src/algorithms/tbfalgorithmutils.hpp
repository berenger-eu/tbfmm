#ifndef TBFALGORITHMUTILS_HPP
#define TBFALGORITHMUTILS_HPP

#include "tbfglobal.hpp"

#include "containers/tbfvectorview.hpp"

#include <cassert>
#include <type_traits>

namespace TbfAlgorithmUtils{


template <class IndexContainerClass, class GroupContainerClassSource, class GroupContainerClassTarget, class FuncType>
inline void TbfMapIndexesAndBlocks(IndexContainerClass inIndexes, GroupContainerClassSource& inGroups, const std::ptrdiff_t idxWorkingGroup,
                                    GroupContainerClassTarget& inGroupsTarget, FuncType&& inFunc){
    if(std::size(inIndexes) == 0 || std::size(inGroups) == 0){
        return;
    }

    std::sort(std::begin(inIndexes), std::end(inIndexes), TbfXtoXInteraction<decltype (inIndexes[0].indexSrc)>::SrcFirst);

    std::ptrdiff_t idxCurrentIndex = 0;
    std::ptrdiff_t idxCurrentGroup = 0;// TODO (idxWorkingGroup == 0 ? 1 : 0);

    while(idxCurrentIndex != static_cast<long int>(std::size(inIndexes))
          && idxCurrentGroup != static_cast<long int>(std::size(inGroups))){
        const auto firstItem = std::lower_bound(inIndexes.begin() + idxCurrentIndex, inIndexes.end(),
                                                inGroups[idxCurrentGroup].getStartingSpacialIndex(),
                                                [](const auto& element, const auto& value){
            return element.indexSrc < value;
        });
        if(firstItem == inIndexes.end()){
            break;
        }
        if(inGroups[idxCurrentGroup].getEndingSpacialIndex() < (*firstItem).indexSrc){
            const auto firstGroup = std::lower_bound(inGroups.begin() + idxCurrentGroup, inGroups.end(),
                                                    (*firstItem).indexSrc,
                                                    [](const auto& element, const auto& value){
                return element.getEndingSpacialIndex() < value;
            });
            if(firstGroup == inGroups.end()){
                break;
            }
            idxCurrentGroup = std::distance(inGroups.begin(), firstGroup);
        }
        else{
            const auto lastItem = std::upper_bound(firstItem, inIndexes.end(),
                                                    inGroups[idxCurrentGroup].getEndingSpacialIndex(),
                                                    [](const auto& value, const auto& element){
                return !(element.indexSrc <= value);
            });

            assert(inGroups[idxCurrentGroup].getStartingSpacialIndex() <= (*firstItem).indexSrc
                   && (*firstItem).indexSrc <= inGroups[idxCurrentGroup].getEndingSpacialIndex());
            assert(inGroups[idxCurrentGroup].getStartingSpacialIndex() <= (*(lastItem-1)).indexSrc
                   && (*(lastItem-1)).indexSrc <= inGroups[idxCurrentGroup].getEndingSpacialIndex());

            inFunc(inGroupsTarget[idxWorkingGroup], inGroups[idxCurrentGroup],
                   TbfMakeVectorView(inIndexes,std::distance(inIndexes.begin(),firstItem),
                                  std::distance(inIndexes.begin(),lastItem)-std::distance(inIndexes.begin(),firstItem)));

            idxCurrentIndex = std::distance(inIndexes.begin(), lastItem);
        }
    }
}

template <class IndexContainerClass, class GroupContainerClass, class FuncType>
inline void TbfMapIndexesAndBlocks(IndexContainerClass&& inIndexes, GroupContainerClass& inGroups, const std::ptrdiff_t idxWorkingGroup,
                                  FuncType&& inFunc){
    TbfMapIndexesAndBlocks(std::forward<IndexContainerClass>(inIndexes), inGroups, idxWorkingGroup, inGroups,
                           std::forward<FuncType>(inFunc));
}

///////////////////////////////////////////////////////////////////////////////

template <class IndexContainerClass, class GroupContainerClassSource, class GroupContainerClassTarget, class FuncType>
inline void TbfMapIndexesAndBlocksIndexes(IndexContainerClass inIndexes, GroupContainerClassSource& inGroups, const long int idxWorkingGroup,
                                   [[maybe_unused]] GroupContainerClassTarget& inGroupsTarget, FuncType&& inFunc){
    if(std::size(inIndexes) == 0 || std::size(inGroups) == 0){
        return;
    }

    std::sort(std::begin(inIndexes), std::end(inIndexes), TbfXtoXInteraction<decltype (inIndexes[0].indexSrc)>::SrcFirst);

    long int idxCurrentIndex = 0;
    long int idxCurrentGroup = 0;// TODO (idxWorkingGroup == 0 ? 1 : 0);

    while(idxCurrentIndex != static_cast<long int>(std::size(inIndexes))
           && idxCurrentGroup != static_cast<long int>(std::size(inGroups))){
        const auto firstItem = std::lower_bound(inIndexes.begin() + idxCurrentIndex, inIndexes.end(),
                                                inGroups[idxCurrentGroup].getStartingSpacialIndex(),
                                                [](const auto& element, const auto& value){
                                                    return element.indexSrc < value;
                                                });
        if(firstItem == inIndexes.end()){
            break;
        }
        if(inGroups[idxCurrentGroup].getEndingSpacialIndex() < (*firstItem).indexSrc){
            const auto firstGroup = std::lower_bound(inGroups.begin() + idxCurrentGroup, inGroups.end(),
                                                     (*firstItem).indexSrc,
                                                     [](const auto& element, const auto& value){
                                                         return element.getEndingSpacialIndex() < value;
                                                     });
            if(firstGroup == inGroups.end()){
                break;
            }
            idxCurrentGroup = std::distance(inGroups.begin(), firstGroup);
        }
        else{
            const auto lastItem = std::upper_bound(firstItem, inIndexes.end(),
                                                   inGroups[idxCurrentGroup].getEndingSpacialIndex(),
                                                   [](const auto& value, const auto& element){
                                                       return !(element.indexSrc <= value);
                                                   });

            assert(inGroups[idxCurrentGroup].getStartingSpacialIndex() <= (*firstItem).indexSrc
                   && (*firstItem).indexSrc <= inGroups[idxCurrentGroup].getEndingSpacialIndex());
            assert(inGroups[idxCurrentGroup].getStartingSpacialIndex() <= (*(lastItem-1)).indexSrc
                   && (*(lastItem-1)).indexSrc <= inGroups[idxCurrentGroup].getEndingSpacialIndex());

            inFunc(idxWorkingGroup, idxCurrentGroup,
                   TbfMakeVectorView(inIndexes,std::distance(inIndexes.begin(),firstItem),
                                     std::distance(inIndexes.begin(),lastItem)-std::distance(inIndexes.begin(),firstItem)));

            idxCurrentIndex = std::distance(inIndexes.begin(), lastItem);
        }
    }
}

template <class IndexContainerClass, class GroupContainerClass, class FuncType>
inline void TbfMapIndexesAndBlocksIndexes(IndexContainerClass&& inIndexes, GroupContainerClass& inGroups, const long int idxWorkingGroup,
                                   FuncType&& inFunc){
    TbfMapIndexesAndBlocksIndexes(std::forward<IndexContainerClass>(inIndexes), inGroups, idxWorkingGroup, inGroups,
                           std::forward<FuncType>(inFunc));
}

///////////////////////////////////////////////////////////////////////////////

enum TbfOperations {
    TbfP2P  = (1 << 0),
    TbfP2M  = (1 << 1),
    TbfM2M  = (1 << 2),
    TbfM2L  = (1 << 3),
    TbfL2L  = (1 << 4),
    TbfL2P  = (1 << 5),

    TbfNearField = TbfP2P,
    TbfFarField  = (TbfP2M|TbfM2M|TbfM2L|TbfL2L|TbfL2P),

    TbfNearAndFarFields = (TbfNearField|TbfFarField),

    TbfBottomToTopStages = (TbfP2M|TbfM2M),
    TbfTopToBottomStages = (TbfL2L|TbfL2P),
    TbfTransferStages = (TbfM2L|TbfP2P)
};

class TbfOperationsPriorities {
    const int treeHeight;

    const int prioP2P;
    const int prioL2P;
    const int prioL2L;
    const int prioM2L;
    const int prioM2M;
    const int prioP2M;

public:
    TbfOperationsPriorities(const long int inTreeHeight)
        : treeHeight(int(inTreeHeight)),
          prioP2P(0),
          prioL2P(1),
          prioL2L(2),
          prioM2L(prioL2L+treeHeight),
          prioM2M(prioM2L+treeHeight),
          prioP2M(prioM2M+treeHeight){}

    int getP2PPriority() const{
        return prioP2P;
    }

    int getP2MPriority() const{
        return prioP2M;
    }

    int getM2MPriority(const long int inLevel) const{
        return prioM2M+treeHeight-int(inLevel)-1;
    }

    int getM2LPriority(const long int inLevel) const{
        return prioM2L+treeHeight-int(inLevel)-1;
    }

    int getL2LPriority(const long int inLevel) const{
        return prioL2L+treeHeight-int(inLevel)-1;
    }

    int getL2PPriority() const{
        return prioL2P;
    }
};
///////////////////////////////////////////////////////////////////////////////


template <const bool>
class BoolSelecter;

template <>
class BoolSelecter<true> : public std::true_type {};

template <>
class BoolSelecter<false> : public std::false_type {};

// See http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2015/n4502.pdf.
template <typename...>
using void_t = void;

#define CUDA_OP_DETECT(OP_NAME)\
template <typename, template <typename> class, typename = void_t<>>\
    struct detect_##OP_NAME : std::false_type {};\
        \
    template <typename T, template <typename> class Op>\
    struct detect_##OP_NAME<T, Op, void_t<Op<T>>> : BoolSelecter<T::OP_NAME> {};\
        \
    template <typename T>\
    using OP_NAME##_test = decltype(T::OP_NAME);\
        \
    template <typename T>\
    using class_has_##OP_NAME = detect_##OP_NAME<T, OP_NAME##_test>;

#define CUDA_METH_DETECT(OP_NAME)\
template <typename, template <typename> class, typename = void_t<>>\
    struct detect_##OP_NAME : std::false_type {};\
        \
    template <typename T, template <typename> class Op>\
    struct detect_##OP_NAME<T, Op, void_t<Op<T>>> : std::true_type {};\
        \
    template <typename T>\
    using OP_NAME##_test = decltype(&T::OP_NAME);\
        \
    template <typename T>\
    using class_has_##OP_NAME = detect_##OP_NAME<T, OP_NAME##_test>;

template <class KernelClass>
class KernelHardwareSupport{
    CUDA_OP_DETECT(CudaP2P)
    CUDA_OP_DETECT(CudaP2M)
    CUDA_OP_DETECT(CudaM2M)
    CUDA_OP_DETECT(CudaM2L)
    CUDA_OP_DETECT(CudaL2L)
    CUDA_OP_DETECT(CudaL2P)

    CUDA_OP_DETECT(CpuP2P)
    CUDA_OP_DETECT(CpuP2M)
    CUDA_OP_DETECT(CpuM2M)
    CUDA_OP_DETECT(CpuM2L)
    CUDA_OP_DETECT(CpuL2L)
    CUDA_OP_DETECT(CpuL2P)

    CUDA_METH_DETECT(initCudaKernelData)
    CUDA_METH_DETECT(releaseCudaKernelData)
public:
    // Check if cuda is enabled
    constexpr static bool CudaP2P = class_has_CudaP2P<KernelClass>::value;
    constexpr static bool CudaP2M = class_has_CudaP2M<KernelClass>::value;
    constexpr static bool CudaM2M = class_has_CudaM2M<KernelClass>::value;
    constexpr static bool CudaM2L = class_has_CudaM2L<KernelClass>::value;
    constexpr static bool CudaL2L = class_has_CudaL2L<KernelClass>::value;
    constexpr static bool CudaL2P = class_has_CudaL2P<KernelClass>::value;

    constexpr static bool CpuP2P = (!CudaP2P || class_has_CpuP2P<KernelClass>::value);
    constexpr static bool CpuP2M = (!CudaP2M || class_has_CpuP2M<KernelClass>::value);
    constexpr static bool CpuM2M = (!CudaM2M || class_has_CpuM2M<KernelClass>::value);
    constexpr static bool CpuM2L = (!CudaM2L || class_has_CpuM2L<KernelClass>::value);
    constexpr static bool CpuL2L = (!CudaL2L || class_has_CpuL2L<KernelClass>::value);
    constexpr static bool CpuL2P = (!CudaL2P || class_has_CpuL2P<KernelClass>::value);

    constexpr static bool HasCudaInit = class_has_initCudaKernelData<KernelClass>::value;
    constexpr static bool HasCudaRelease = class_has_releaseCudaKernelData<KernelClass>::value;
};

}

#endif

