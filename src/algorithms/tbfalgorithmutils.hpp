#ifndef TBFALGORITHMUTILS_HPP
#define TBFALGORITHMUTILS_HPP

#include "tbfglobal.hpp"

#include "containers/tbfvectorview.hpp"

#include <cassert>

namespace TbfAlgorithmUtils{


template <class IndexContainerClass, class GroupContainerClassSource, class GroupContainerClassTarget, class FuncType>
inline void TbfMapIndexesAndBlocks(IndexContainerClass inIndexes, GroupContainerClassSource& inGroups, const long int idxWorkingGroup,
                                    GroupContainerClassTarget& inGroupsTarget, FuncType&& inFunc){
    if(std::size(inIndexes) == 0 || std::size(inGroups) == 0){
        return;
    }

    std::sort(std::begin(inIndexes), std::end(inIndexes), TbfXtoXInteraction<decltype (inIndexes[0].indexSrc)>::SrcFirst);

    long int idxCurrentIndex = 0;
    long int idxCurrentGroup = (idxWorkingGroup == 0 ? 1 : 0);

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

            inFunc(inGroups[idxWorkingGroup], inGroups[idxCurrentGroup],
                   TbfMakeVectorView(inIndexes,std::distance(inIndexes.begin(),firstItem),
                                  std::distance(inIndexes.begin(),lastItem)-std::distance(inIndexes.begin(),firstItem)));

            idxCurrentIndex = std::distance(inIndexes.begin(), lastItem);
        }
    }
}

template <class IndexContainerClass, class GroupContainerClass, class FuncType>
inline void TbfMapIndexesAndBlocks(IndexContainerClass&& inIndexes, GroupContainerClass& inGroups, const long int idxWorkingGroup,
                                  FuncType&& inFunc){
    TbfMapIndexesAndBlocks(std::forward<IndexContainerClass>(inIndexes), inGroups, idxWorkingGroup, inGroups,
                           std::forward<FuncType>(inFunc));
}


enum LFmmOperations {
    LFmmP2P  = (1 << 0),
    LFmmP2M  = (1 << 1),
    LFmmM2M  = (1 << 2),
    LFmmM2L  = (1 << 3),
    LFmmL2L  = (1 << 4),
    LFmmL2P  = (1 << 5),

    LFmmNearField = LFmmP2P,
    LFmmFarField  = (LFmmP2M|LFmmM2M|LFmmM2L|LFmmL2L|LFmmL2P),

    LFmmNearAndFarFields = (LFmmNearField|LFmmFarField)
};

class LFmmOperationsPriorities {
    const int treeHeight;

    const int prioP2P;
    const int prioL2P;
    const int prioL2L;
    const int prioM2L;
    const int prioM2M;
    const int prioP2M;

public:
    LFmmOperationsPriorities(const long int inTreeHeight)
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

}

#endif

