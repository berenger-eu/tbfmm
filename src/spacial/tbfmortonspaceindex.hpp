#ifndef TBFMORTONSPACEINDEX_HPP
#define TBFMORTONSPACEINDEX_HPP

#include "tbfglobal.hpp"

#include "utils/tbfutils.hpp"
#include "core/tbfinteraction.hpp"

#include <vector>
#include <array>
#include <cassert>

template <long int Dim_T, class ConfigurationClass_T, const bool IsPeriodic_v = false>
class TbfMortonSpaceIndex{
public:
    static_assert (Dim_T > 0, "Dimension must be greater than 0" );
    static_assert(Dim_T == ConfigurationClass_T::Dim, "Provided dimension must be equal to dimension of ConfigurationClass_T" );

    using IndexType = long int;
    using ConfigurationClass = ConfigurationClass_T;
    using RealType = typename ConfigurationClass::RealType;

    static constexpr long int Dim = Dim_T;
    static constexpr bool IsPeriodic = IsPeriodic_v;

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

    IndexType getUpperBoundAtLeafLevel() const{
        return getUpperBound(configuration.getTreeHeight()-1);
    }

    IndexType getBoxLimit(const long int inLevel) const{
        return (IndexType(1) << (inLevel));
    }

    IndexType getBoxLimitAtLeafLevel() const{
        return getBoxLimit(configuration.getTreeHeight()-1);
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
#ifdef __NVCC__
    __device__ __host__
#endif
    IndexType getParentIndex(IndexType inIndex) const{
        return inIndex >> Dim;
    }

    long int childPositionFromParent(const IndexType inIndexChild) const {
        return inIndexChild & static_cast<long int>(~(((~0UL)>>Dim)<<Dim));
    }

    const auto& getConfiguration() const{
        return configuration;
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
        std::vector<IndexType> indexes;

        if constexpr(IsPeriodic == false){
            if(inLevel < 2){
                return indexes;
            }
        }
        else{
            if(inLevel < 1){
                return indexes;
            }
        }

        const long int boxLimite = (1 << (inLevel));
        const long int boxLimiteParent = (1 << (inLevel-1));

        const IndexType cellIndex = inMIndex;
        const auto cellPos = getBoxPosFromIndex(cellIndex);

        const IndexType parentCellIndex = getParentIndex(cellIndex);
        const auto parentCellPos = getBoxPosFromIndex(parentCellIndex);


        std::array<long int, Dim> minLimits;
        std::array<long int, Dim> maxLimits;
        std::array<long int, Dim> currentParentTest;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            if constexpr(IsPeriodic == false){
                if(parentCellPos[idxDim] == 0){
                    minLimits[idxDim] = 0;
                }
                else{
                    minLimits[idxDim] = -1;
                }
                if(parentCellPos[idxDim]+1 == boxLimiteParent){
                    maxLimits[idxDim] = 0;
                }
                else{
                    maxLimits[idxDim] = 1;
                }
            }
            else{
                minLimits[idxDim] = -1;
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
            auto periodicShift = TbfUtils::make_array<long int, Dim>(0);

            if constexpr(IsPeriodic){
                for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                    if(otherParentPos[idxDim] < 0){
                        periodicShift[idxDim] = -boxLimite;
                        otherParentPos[idxDim] += boxLimiteParent;
                    }
                    else if(boxLimiteParent <= otherParentPos[idxDim]){
                        periodicShift[idxDim] = boxLimite;
                        otherParentPos[idxDim] -= boxLimiteParent;
                    }
                }
            }
            const IndexType otherParentIndex = getIndexFromBoxPos(otherParentPos);

            for(long int idxChild = 0 ; idxChild < (1<<Dim) ; ++idxChild){
                const IndexType childIndex = getChildIndexFromParent(otherParentIndex, idxChild);
                auto childPos = getBoxPosFromIndex(childIndex);

                bool isTooClose = true;
                for(int idxDim = 0 ; isTooClose && idxDim < Dim ; ++idxDim){
                    if(std::abs(childPos[idxDim] + periodicShift[idxDim] - cellPos[idxDim]) > 1){
                        isTooClose = false;
                    }
                }

                if(isTooClose == false){
                    long int arrayPos = 0;
                    for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        arrayPos *= 7;
                        arrayPos += (childPos[idxDim] + periodicShift[idxDim] - cellPos[idxDim] + 3);
                    }
                    assert(arrayPos < TbfUtils::lipow(7,Dim));

                    if constexpr(IsPeriodic){
                        [[maybe_unused]] auto generatedPos = getRelativePosFromInteractionIndex(arrayPos);
                        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                            assert((childPos[idxDim] + periodicShift[idxDim] - cellPos[idxDim]) == generatedPos[idxDim]);
                        }
                    }

                    indexes.push_back(childIndex);
                }
            }

            currentParentTest[Dim-1] += 1;
        }


        if constexpr(IsPeriodic){
            assert(std::size(indexes) == getNbInteractionsPerCell());
        }

        return indexes;
    }

    template <class GroupClass>
    auto getInteractionListForBlock(const GroupClass& inGroup, const long int inLevel, const bool testSelfInclusion = true) const{
        assert(inLevel >= 0);

        std::vector<TbfXtoXInteraction<IndexType>> indexesInternal;
        indexesInternal.reserve(inGroup.getNbCells());

        std::vector<TbfXtoXInteraction<IndexType>> indexesExternal;
        indexesExternal.reserve(inGroup.getNbCells());

        if constexpr(IsPeriodic == false){
            if(inLevel < 2){
                return std::make_pair(std::move(indexesInternal), std::move(indexesExternal));
            }
        }
        else{
            if(inLevel < 1){
                return std::make_pair(std::move(indexesInternal), std::move(indexesExternal));
            }
        }

        const long int boxLimite = (1 << (inLevel));
        const long int boxLimiteParent = (1 << (inLevel-1));

        for(long int idxCell = 0 ; idxCell < inGroup.getNbCells() ; ++idxCell){
            const IndexType cellIndex = inGroup.getCellSpacialIndex(idxCell);
            const auto cellPos = getBoxPosFromIndex(cellIndex);

            const IndexType parentCellIndex = getParentIndex(cellIndex);
            const auto parentCellPos = getBoxPosFromIndex(parentCellIndex);


            std::array<long int, Dim> minLimits;
            std::array<long int, Dim> maxLimits;
            std::array<long int, Dim> currentParentTest;

            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                if constexpr(IsPeriodic == false){
                    if(parentCellPos[idxDim] == 0){
                        minLimits[idxDim] = 0;
                    }
                    else{
                        minLimits[idxDim] = -1;
                    }
                    if(parentCellPos[idxDim]+1 == boxLimiteParent){
                        maxLimits[idxDim] = 0;
                    }
                    else{
                        maxLimits[idxDim] = 1;
                    }
                }
                else{
                    minLimits[idxDim] = -1;
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
                auto periodicShift = TbfUtils::make_array<long int, Dim>(0);

                if constexpr(IsPeriodic){
                    for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        if(otherParentPos[idxDim] < 0){
                            periodicShift[idxDim] = -boxLimite;
                            otherParentPos[idxDim] += boxLimiteParent;
                        }
                        else if(boxLimiteParent <= otherParentPos[idxDim]){
                            periodicShift[idxDim] = boxLimite;
                            otherParentPos[idxDim] -= boxLimiteParent;
                        }
                    }
                }
                const IndexType otherParentIndex = getIndexFromBoxPos(otherParentPos);


                for(long int idxChild = 0 ; idxChild < (1<<Dim) ; ++idxChild){
                    const IndexType childIndex = getChildIndexFromParent(otherParentIndex, idxChild);
                    auto childPos = getBoxPosFromIndex(childIndex);

                    bool isTooClose = true;
                    for(int idxDim = 0 ; isTooClose && idxDim < Dim ; ++idxDim){
                        if(std::abs(childPos[idxDim] + periodicShift[idxDim] - cellPos[idxDim]) > 1){
                            isTooClose = false;
                        }
                    }

                    if(isTooClose == false){
                        long int arrayPos = 0;
                        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                            arrayPos *= 7;
                            arrayPos += (childPos[idxDim] + periodicShift[idxDim] - cellPos[idxDim] + 3);
                        }
                        assert(arrayPos < TbfUtils::lipow(7,Dim));

                        TbfXtoXInteraction<IndexType> interaction;
                        interaction.indexTarget = cellIndex;
                        interaction.indexSrc = childIndex;
                        interaction.globalTargetPos = idxCell;
                        interaction.arrayIndexSrc = arrayPos;

                        if constexpr(IsPeriodic){
                            [[maybe_unused]] auto generatedPos = getRelativePosFromInteractionIndex(arrayPos);
                            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                                assert((childPos[idxDim] + periodicShift[idxDim] - cellPos[idxDim]) == generatedPos[idxDim]);
                            }
                        }

                        if(inGroup.getStartingSpacialIndex() <= interaction.indexSrc
                                && interaction.indexSrc <= inGroup.getEndingSpacialIndex()){
                            if(testSelfInclusion == false || inGroup.getElementFromSpacialIndex(interaction.indexSrc)){
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


    auto getNeighborListForIndex(const IndexType cellIndex, const long int inLevel, const bool upperExclusion = false) const{
        assert(inLevel >= 0);
        const long int boxLimite = (1 << (inLevel));

        std::vector<IndexType> indexes;
        indexes.reserve(TbfUtils::lipow(3,Dim)/2);

        const auto cellPos = getBoxPosFromIndex(cellIndex);

        std::array<long int, Dim> minLimits;
        std::array<long int, Dim> maxLimits;
        std::array<long int, Dim> currentTest;

        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            if constexpr(IsPeriodic == false){
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
            }
            else{
                minLimits[idxDim] = -1;
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

            bool isSelfCell = true;
            for(int idxDim = 0 ; isSelfCell && idxDim < Dim ; ++idxDim){
                if(currentTest[idxDim] != 0){
                    isSelfCell = false;
                }
            }

            if(isSelfCell == false){                
                auto otherPos = TbfUtils::AddVecToVec(cellPos, currentTest);

                long int arrayPos = 0;
                for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                    arrayPos *= 3;
                    arrayPos += (otherPos[idxDim] - cellPos[idxDim] + 1);
                }
                assert(arrayPos < TbfUtils::lipow(3, Dim));                

                if constexpr(IsPeriodic){
                    for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        otherPos[idxDim] = ((otherPos[idxDim]+boxLimite)%boxLimite);
                    }
                    [[maybe_unused]] const auto generatedPos = getRelativePosFromNeighborIndex(arrayPos);
                    for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        assert((currentTest[idxDim]) == generatedPos[idxDim]);
                    }
                }

                const IndexType otherIndex = getIndexFromBoxPos(otherPos);

                // We cannot compare with otherIndex < cellIndex due to periodicity
                if(upperExclusion == false || TbfUtils::lipow(3, Dim)/2 < arrayPos){
                    indexes.push_back(otherIndex);
                }
            }

            currentTest[Dim-1] += 1;
        }

        if constexpr(IsPeriodic){
            assert(std::size(indexes) == getNbNeighborsPerLeaf());
        }

        return indexes;
    }

    template <class GroupClass>
    auto getNeighborListForBlock(const GroupClass& inGroup, const long int inLevel, const bool upperExclusion = false, const bool testSelfInclusion = true) const{
        assert(inLevel >= 0);
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
                if constexpr(IsPeriodic == false){
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
                }
                else{
                    minLimits[idxDim] = -1;
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

                bool isSelfCell = true;
                for(int idxDim = 0 ; isSelfCell && idxDim < Dim ; ++idxDim){
                    if(currentTest[idxDim] != 0){
                        isSelfCell = false;
                    }
                }

                if(isSelfCell == false){
                    auto otherPos = TbfUtils::AddVecToVec(cellPos, currentTest);

                    long int arrayPos = 0;
                    for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                        arrayPos *= 3;
                        arrayPos += (otherPos[idxDim] - cellPos[idxDim] + 1);
                    }
                    assert(arrayPos < TbfUtils::lipow(3, Dim));

                    if constexpr(IsPeriodic){
                        for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                            otherPos[idxDim] = ((otherPos[idxDim]+boxLimite)%boxLimite);
                        }
                    }

                    const IndexType otherIndex = getIndexFromBoxPos(otherPos);

                    // We cannot compare with otherIndex < cellIndex due to periodicity
                    if(upperExclusion == false || TbfUtils::lipow(3, Dim)/2 < arrayPos){
                        TbfXtoXInteraction<IndexType> interaction;
                        interaction.indexTarget = cellIndex;
                        interaction.indexSrc = otherIndex;
                        interaction.globalTargetPos = idxCell;
                        interaction.arrayIndexSrc = arrayPos;

                        if constexpr(IsPeriodic){
                            [[maybe_unused]] const auto generatedPos = getRelativePosFromNeighborIndex(arrayPos);
                            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                                assert((currentTest[idxDim]) == generatedPos[idxDim]);
                            }
                        }

                        if(inGroup.getStartingSpacialIndex() <= interaction.indexSrc
                                && interaction.indexSrc <= inGroup.getEndingSpacialIndex()){
                            if(testSelfInclusion == false || inGroup.getElementFromSpacialIndex(interaction.indexSrc)){
                                indexesInternal.push_back(interaction);
                            }
                        }
                        else{
                            indexesExternal.push_back(interaction);
                        }
                    }
                }

                currentTest[Dim-1] += 1;
            }
        }

        return std::make_pair(std::move(indexesInternal), std::move(indexesExternal));
    }


    template <class GroupClass>
    auto getSelfListForBlock(const GroupClass& inGroup) const{
        std::vector<TbfXtoXInteraction<IndexType>> indexesSelf;
        indexesSelf.reserve(inGroup.getNbLeaves());

        long int arrayPos = 0;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            arrayPos *= 3;
            arrayPos += (1);
        }
        assert(arrayPos < TbfUtils::lipow(3, Dim));

        for(long int idxCell = 0 ; idxCell < inGroup.getNbLeaves() ; ++idxCell){
            const IndexType cellIndex = inGroup.getLeafSpacialIndex(idxCell);

            TbfXtoXInteraction<IndexType> interaction;
            interaction.indexTarget = cellIndex;
            interaction.indexSrc = cellIndex;
            interaction.globalTargetPos = idxCell;
            interaction.arrayIndexSrc = arrayPos;

            indexesSelf.emplace_back(interaction);
        }

        return indexesSelf;
    }


    static long int constexpr getNbChildrenPerCell() {
        return 1L << Dim;
    }

    static long int constexpr getNbInteractionsPerCell() {
        long int nbNeighbors = 1;
        long int nbNeighborsTooClose = 1;
        for(long int idxNeigh = 0 ; idxNeigh < Dim ; ++idxNeigh){
            nbNeighbors *= 6;
            nbNeighborsTooClose *= 3;
        }
        return nbNeighbors - nbNeighborsTooClose;
    }

    static long int constexpr getNbNeighborsPerLeaf() {
        long int nbNeighbors = 1;
        for(long int idxNeigh = 0 ; idxNeigh < Dim ; ++idxNeigh){
            nbNeighbors *= 3;
        }
        return nbNeighbors - 1;
    }

    static long int constexpr get3PowDim() {
        long int nbNeighbors = 1;
        for(long int idxNeigh = 0 ; idxNeigh < Dim ; ++idxNeigh){
            nbNeighbors *= 3;
        }
        return nbNeighbors;
    }

    static auto getRelativePosFromInteractionIndex(long int inArrayPos){
        std::array<long int, Dim> pos;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            pos[Dim-1-idxDim] = (inArrayPos%7) - 3;
            inArrayPos /= 7;
        }
        return pos;
    }

    static auto getRelativePosFromNeighborIndex(long int inArrayPos){
        std::array<long int, Dim> pos;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            pos[Dim-1-idxDim] = (inArrayPos%3) - 1;
            inArrayPos /= 3;
        }
        return pos;
    }

    static auto getInteractionIndexFromRelativePos(const std::array<long int, Dim>& pos){
        long int arrayPos = 0;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            arrayPos *= 7;
            assert(-3 <= pos[idxDim] && pos[idxDim] <= 3);
            arrayPos += (pos[idxDim] + 3);
        }
        return arrayPos;
    }

    static auto getNeighborIndexFromRelativePos(const std::array<long int, Dim>& pos){
        long int arrayPos = 0;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            arrayPos *= 3;
            assert(-1 <= pos[idxDim] && pos[idxDim] <= 1);
            arrayPos += (pos[idxDim] + 1);
        }
        return arrayPos;
    }

    auto getColorsIdxAtLeafLevel(const IndexType inLeafIndex) const{
        const auto leafPos = getBoxPosFromIndex(inLeafIndex);
        long int colorIdx = 0;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            colorIdx *= 3;
            colorIdx += (leafPos[idxDim]%3);
        }
        return colorIdx;
    }

    template <class StreamClass>
    friend  StreamClass& operator<<(StreamClass& inStream, const TbfMortonSpaceIndex& inSpaceSystem) {
        inStream << "TbfMortonSpaceIndex @ " << &inSpaceSystem << "\n";
        inStream << " - Configuration: " << "\n";
        inStream << inSpaceSystem.configuration << "\n";
        return inStream;
    }
};

#endif
