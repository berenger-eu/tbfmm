#ifndef TBFMORTONSPACEINDEX_HPP
#define TBFMORTONSPACEINDEX_HPP

#include "tbfglobal.hpp"

#include "utils/tbfutils.hpp"
#include "core/tbfinteraction.hpp"

#include <vector>
#include <array>
#include <cassert>

template <long int Dim_T, class ConfigurationClass_T>
class TbfMortonSpaceIndex{
public:
    static_assert (Dim_T > 0, "Dimension must be greater than 0" );

    using IndexType = long int;
    using ConfigurationClass = ConfigurationClass_T;
    using RealType = typename ConfigurationClass::RealType;

    static constexpr long int Dim = Dim_T;

protected:
    const ConfigurationClass configuration;

    long int getTreeCoordinate(const RealType inRelativePosition, const long int inDim) const {
        assert(inRelativePosition >= 0 && inRelativePosition <= configuration.getBoxWidths()[inDim]);
        if(inRelativePosition == configuration.getBoxWidths()[inDim]){
            return (1 << (configuration.getTreeHeight()-1))-1;
        }
        const RealType indexFReal = inRelativePosition / configuration.getLeafWidths()[inDim];
        return static_cast<long int>(indexFReal);
    }

public:
    TbfMortonSpaceIndex(const ConfigurationClass& inConfiguration)
        : configuration(inConfiguration){
    }

    IndexType getUpperBound(const long int inLevel) const{
        return (IndexType(1) << (inLevel * Dim));
    }

    template <class PositionType>
    IndexType getIndexFromPosition(const PositionType& inPos) const {
        std::array<long int,Dim> host;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            host[idxDim] = getTreeCoordinate( inPos[idxDim] - configuration.getBoxCorner()[idxDim], idxDim);
        }

        return getIndexFromBoxPos(host);
    }

    std::array<RealType,Dim> getRealPosFromBoxPos(const std::array<long int,Dim>& inPos) const {
        const std::array<RealType,Dim> boxCorner(configuration.getBoxCenter(),-(configuration.getBoxWidths()/2));

        std::array<RealType,Dim> host;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            host[idxDim] = ( inPos[idxDim]*configuration.getLeafWidths()[idxDim] + boxCorner[idxDim] );
        }

        return host;
    }

    std::array<long int,Dim> getBoxPosFromIndex(IndexType inMindex) const{
        IndexType mask = 0x1LL;

        std::array<long int,Dim> boxPos;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            boxPos[idxDim] = 0;
        }

        while(inMindex >= mask) {
            for(long int idxDim = Dim-1 ; idxDim > 0 ; --idxDim){
                boxPos[idxDim] |= static_cast<long int>(inMindex & mask);
                inMindex >>= 1;
            }

            boxPos[0] |= static_cast<long int>(inMindex & mask);

            mask <<= 1;
        }

        return boxPos;
    }

    IndexType getParentIndex(IndexType inIndex) const{
        return inIndex >> Dim;
    }

    long int childPositionFromParent(const IndexType inIndexChild) const {
        return inIndexChild & static_cast<long int>(~(((~0UL)>>Dim)<<Dim));
    }

    IndexType getIndexFromBoxPos(const std::array<long int,Dim>& inBoxPos) const{
        IndexType index = 0x0LL;
        IndexType mask = 0x1LL;

        bool shouldContinue = false;

        std::array<IndexType,Dim> mcoord;
        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            mcoord[idxDim] = (inBoxPos[idxDim] << (Dim - idxDim - 1));
            shouldContinue |= ((mask << (Dim - idxDim - 1)) <= mcoord[idxDim]);
        }

        while(shouldContinue){
            shouldContinue = false;
            for(long int idxDim = Dim-1 ; idxDim >= 0 ; --idxDim){
                index |= (mcoord[idxDim] & mask);
                mask <<= 1;
                mcoord[idxDim] <<= (Dim-1);
                shouldContinue |= ((mask << (Dim - idxDim - 1)) <= mcoord[idxDim]);
            }
        }

        return index;
    }

    IndexType getChildIndexFromParent(const IndexType inParentIndex, const long int inChild) const{
        return (inParentIndex<<Dim) + inChild;
    }

    auto getInteractionListForIndex(const IndexType inMIndex, const long int inLevel) const{
        const long int boxLimite = (1 << (inLevel-1));

        std::vector<IndexType> indexes;

        const IndexType cellIndex = inMIndex;
        const auto cellPos = getBoxPosFromIndex(cellIndex);

        const IndexType parentCellIndex = getParentIndex(cellIndex);
        const auto parentCellPos = getBoxPosFromIndex(parentCellIndex);


        std::array<long int, Dim> minLimits;
        std::array<long int, Dim> maxLimits;
        std::array<long int, Dim> currentParentTest;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            if(parentCellPos[idxDim] == 0){
                minLimits[idxDim] = 0;
            }
            else{
                minLimits[idxDim] = -1;
            }
            if(parentCellPos[idxDim]+1 == boxLimite){
                maxLimits[idxDim] = 0;
            }
            else{
                maxLimits[idxDim] = 1;
            }
            currentParentTest[idxDim] = minLimits[idxDim];
        }


        while(true){
            {
                long int currentIdx = Dim-1;

                while(currentIdx >= 0 && currentParentTest[currentIdx] > maxLimits[currentIdx]){
                    currentParentTest[currentIdx] = minLimits[currentIdx];
                    currentIdx -= 1;
                    if(currentIdx >= 0){
                        currentParentTest[currentIdx] += 1;
                    }
                }
                if(currentIdx < 0){
                    break;
                }
            }

            auto otherParentPos = TbfUtils::AddVecToVec(parentCellPos, currentParentTest);
            const IndexType otherParentIndex = getIndexFromBoxPos(otherParentPos);

            for(long int idxChild = 0 ; idxChild < (1<<Dim) ; ++idxChild){
                const IndexType childIndex = getChildIndexFromParent(otherParentIndex, idxChild);
                auto childPos = getBoxPosFromIndex(childIndex);

                bool isTooClose = true;
                for(int idxDim = 0 ; isTooClose && idxDim < Dim ; ++idxDim){
                    if(std::abs(childPos[idxDim] - cellPos[idxDim]) > 1){
                        isTooClose = false;
                    }
                }

                if(isTooClose == false){
                    long int arrayPos = 0;
                    for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        arrayPos *= 7;
                        arrayPos += (childPos[idxDim] - cellPos[idxDim] + 3);
                    }
                    assert(arrayPos < TbfUtils::lipow(7,Dim));

                    indexes.push_back(childIndex);
                }
            }

            currentParentTest[Dim-1] += 1;
        }

        return indexes;
    }

    template <class GroupClass>
    auto getInteractionListForBlock(const GroupClass& inGroup, const long int inLevel) const{
        const long int boxLimite = (1 << (inLevel-1));

        std::vector<TbfXtoXInteraction<IndexType>> indexesInternal;
        indexesInternal.reserve(inGroup.getNbCells());

        std::vector<TbfXtoXInteraction<IndexType>> indexesExternal;
        indexesExternal.reserve(inGroup.getNbCells());

        for(long int idxCell = 0 ; idxCell < inGroup.getNbCells() ; ++idxCell){
            const IndexType cellIndex = inGroup.getCellSpacialIndex(idxCell);
            const auto cellPos = getBoxPosFromIndex(cellIndex);

            const IndexType parentCellIndex = getParentIndex(cellIndex);
            const auto parentCellPos = getBoxPosFromIndex(parentCellIndex);


            std::array<long int, Dim> minLimits;
            std::array<long int, Dim> maxLimits;
            std::array<long int, Dim> currentParentTest;

            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                if(parentCellPos[idxDim] == 0){
                    minLimits[idxDim] = 0;
                }
                else{
                    minLimits[idxDim] = -1;
                }
                if(parentCellPos[idxDim]+1 == boxLimite){
                    maxLimits[idxDim] = 0;
                }
                else{
                    maxLimits[idxDim] = 1;
                }
                currentParentTest[idxDim] = minLimits[idxDim];
            }


            while(true){
                {
                    long int currentIdx = Dim-1;

                    while(currentIdx >= 0 && currentParentTest[currentIdx] > maxLimits[currentIdx]){
                        currentParentTest[currentIdx] = minLimits[currentIdx];
                        currentIdx -= 1;
                        if(currentIdx >= 0){
                            currentParentTest[currentIdx] += 1;
                        }
                    }
                    if(currentIdx < 0){
                        break;
                    }
                }

                auto otherParentPos = TbfUtils::AddVecToVec(parentCellPos, currentParentTest);
                const IndexType otherParentIndex = getIndexFromBoxPos(otherParentPos);

                for(long int idxChild = 0 ; idxChild < (1<<Dim) ; ++idxChild){
                    const IndexType childIndex = getChildIndexFromParent(otherParentIndex, idxChild);
                    auto childPos = getBoxPosFromIndex(childIndex);

                    bool isTooClose = true;
                    for(int idxDim = 0 ; isTooClose && idxDim < Dim ; ++idxDim){
                        if(std::abs(childPos[idxDim] - cellPos[idxDim]) > 1){
                            isTooClose = false;
                        }
                    }

                    if(isTooClose == false){
                        long int arrayPos = 0;
                        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                            arrayPos *= 7;
                            arrayPos += (childPos[idxDim] - cellPos[idxDim] + 3);
                        }
                        assert(arrayPos < TbfUtils::lipow(7,Dim));

                        TbfXtoXInteraction<IndexType> interaction;
                        interaction.indexTarget = cellIndex;
                        interaction.indexSrc = childIndex;
                        interaction.globalTargetPos = idxCell;
                        interaction.arrayIndexSrc = arrayPos;

                        if(inGroup.getStartingSpacialIndex() <= interaction.indexSrc
                                && interaction.indexSrc <= inGroup.getEndingSpacialIndex()){
                            if(inGroup.getElementFromSpacialIndex(interaction.indexSrc)){
                                indexesInternal.push_back(interaction);
                            }
                        }
                        else{
                            indexesExternal.push_back(interaction);
                        }
                    }
                }

                currentParentTest[Dim-1] += 1;
            }
        }

        return std::make_pair(std::move(indexesInternal), std::move(indexesExternal));
    }


    auto getNeighborListForBlock(const IndexType cellIndex, const long int inLevel) const{
        const long int boxLimite = (1 << (inLevel));

        std::vector<IndexType> indexes;
        indexes.reserve(TbfUtils::lipow(3,Dim)/2);

        const auto cellPos = getBoxPosFromIndex(cellIndex);

        std::array<long int, Dim> minLimits;
        std::array<long int, Dim> maxLimits;
        std::array<long int, Dim> currentTest;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            if(cellPos[idxDim] == 0){
                minLimits[idxDim] = 0;
            }
            else{
                minLimits[idxDim] = -1;
            }
            if(cellPos[idxDim]+1 == boxLimite){
                maxLimits[idxDim] = 0;
            }
            else{
                maxLimits[idxDim] = 1;
            }
            currentTest[idxDim] = minLimits[idxDim];
        }

        while(true){
            {
                long int currentIdx = Dim-1;

                while(currentIdx >= 0 && currentTest[currentIdx] > maxLimits[currentIdx]){
                    currentTest[currentIdx] = minLimits[currentIdx];
                    currentIdx -= 1;
                    if(currentIdx >= 0){
                        currentTest[currentIdx] += 1;
                    }
                }
                if(currentIdx < 0){
                    break;
                }
            }

            auto otherPos = TbfUtils::AddVecToVec(cellPos, currentTest);

            bool isSelfCell = true;
            for(int idxDim = 0 ; isSelfCell && idxDim < Dim ; ++idxDim){
                if(std::abs(otherPos[idxDim] - cellPos[idxDim]) > 0){
                    isSelfCell = false;
                }
            }

            if(isSelfCell == false){
                long int arrayPos = 0;
                for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                    arrayPos *= 3;
                    arrayPos += (otherPos[idxDim] - cellPos[idxDim] + 1);
                }
                assert(arrayPos < TbfUtils::lipow(3, Dim));

                const IndexType otherIndex = getIndexFromBoxPos(otherPos);

                indexes.push_back(otherIndex);
            }

            currentTest[Dim-1] += 1;
        }

        return indexes;
    }

    template <class GroupClass>
    auto getNeighborListForBlock(const GroupClass& inGroup, const long int inLevel) const{
        const long int boxLimite = (1 << (inLevel));

        std::vector<TbfXtoXInteraction<IndexType>> indexesInternal;
        indexesInternal.reserve(inGroup.getNbLeaves());

        std::vector<TbfXtoXInteraction<IndexType>> indexesExternal;
        indexesExternal.reserve(inGroup.getNbLeaves());

        for(long int idxCell = 0 ; idxCell < inGroup.getNbLeaves() ; ++idxCell){
            const IndexType cellIndex = inGroup.getLeafSpacialIndex(idxCell);
            const auto cellPos = getBoxPosFromIndex(cellIndex);

            std::array<long int, Dim> minLimits;
            std::array<long int, Dim> maxLimits;
            std::array<long int, Dim> currentTest;

            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                if(cellPos[idxDim] == 0){
                    minLimits[idxDim] = 0;
                }
                else{
                    minLimits[idxDim] = -1;
                }
                if(cellPos[idxDim]+1 == boxLimite){
                    maxLimits[idxDim] = 0;
                }
                else{
                    maxLimits[idxDim] = 1;
                }
                currentTest[idxDim] = minLimits[idxDim];
            }


            while(true){
                {
                    long int currentIdx = Dim-1;

                    while(currentIdx >= 0 && currentTest[currentIdx] > maxLimits[currentIdx]){
                        currentTest[currentIdx] = minLimits[currentIdx];
                        currentIdx -= 1;
                        if(currentIdx >= 0){
                            currentTest[currentIdx] += 1;
                        }
                    }
                    if(currentIdx < 0){
                        break;
                    }
                }

                auto otherPos = TbfUtils::AddVecToVec(cellPos, currentTest);

                bool isSelfCell = true;
                for(int idxDim = 0 ; isSelfCell && idxDim < Dim ; ++idxDim){
                    if(std::abs(otherPos[idxDim] - cellPos[idxDim]) > 0){
                        isSelfCell = false;
                    }
                }

                if(isSelfCell == false){
                    long int arrayPos = 0;
                    for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        arrayPos *= 3;
                        arrayPos += (otherPos[idxDim] - cellPos[idxDim] + 1);
                    }
                    assert(arrayPos < TbfUtils::lipow(3, Dim));

                    const IndexType otherIndex = getIndexFromBoxPos(otherPos);

                    TbfXtoXInteraction<IndexType> interaction;
                    interaction.indexTarget = cellIndex;
                    interaction.indexSrc = otherIndex;
                    interaction.globalTargetPos = idxCell;
                    interaction.arrayIndexSrc = arrayPos;

                    if(inGroup.getStartingSpacialIndex() <= interaction.indexSrc
                            && interaction.indexSrc <= inGroup.getEndingSpacialIndex()){
                        if(inGroup.getElementFromSpacialIndex(interaction.indexSrc)){
                            indexesInternal.push_back(interaction);
                        }
                    }
                    else{
                        indexesExternal.push_back(interaction);
                    }
                }

                currentTest[Dim-1] += 1;
            }
        }

        return std::make_pair(std::move(indexesInternal), std::move(indexesExternal));
    }
};

#endif
