#include "UTester.hpp"

#include "utils/tbfutils.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <set>

class TestLipow : public UTester< TestLipow > {
    using Parent = UTester< TestLipow >;
    
    void TestBasic() {
        const int Dim = 3;
        const long int TreeHeight = 5;

        using RealType = double;

        const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
        const std::array<RealType, Dim> inBoxCenter{{0.5, 0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, inBoxCenter);
        const TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim> > morton(configuration);

        using IndexType = typename TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim> >::IndexType;

        UASSERTEEQUAL(morton.getUpperBound(0), IndexType(1));
        UASSERTEEQUAL(morton.getUpperBound(1), IndexType(8));
        UASSERTEEQUAL(morton.getUpperBound(2), IndexType(8*8));

        for(long int idxLevel = 0 ; idxLevel < TreeHeight-1 ; ++idxLevel){
            std::set<std::array<long int,Dim>> generatedPositions;
            std::set<IndexType> generatedChildIndexes;

            for(IndexType idx = 0 ; idx < morton.getUpperBound(idxLevel) ; ++idx){
                auto pos = morton.getBoxPosFromIndex(idx);
                UASSERTETRUE(generatedPositions.find(pos) == generatedPositions.end());
                generatedPositions.insert(pos);

                UASSERTETRUE(idx == morton.getIndexFromBoxPos(pos));

                for(long int idxChild = 0 ; idxChild < 8 ; ++idxChild){
                    auto childIndex = morton.getChildIndexFromParent(idx, idxChild);
                    UASSERTETRUE(idxChild == morton.childPositionFromParent(childIndex));
                    UASSERTETRUE(idx == morton.getParentIndex(childIndex));
                    UASSERTETRUE(childIndex < morton.getUpperBound(idxLevel+1));
                    UASSERTETRUE(generatedChildIndexes.find(childIndex) == generatedChildIndexes.end());
                    generatedChildIndexes.insert(childIndex);
                }

                {
                    const auto parent = morton.getParentIndex(idx);
                    const auto parentPos = morton.getBoxPosFromIndex(parent);

                    const auto interactionList = morton.getInteractionListForIndex(idx, idxLevel);
                    for(auto interaction : interactionList){
                        const auto posInteraction = morton.getBoxPosFromIndex(interaction);

                        {
                            bool atLeastOneMoreThanOne = false;

                            for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                                UASSERTETRUE(std::abs(pos[idxDim] - posInteraction[idxDim]) <= 3);
                                if(std::abs(pos[idxDim] - posInteraction[idxDim]) > 1){
                                    atLeastOneMoreThanOne = true;
                                }
                            }
                            UASSERTETRUE(atLeastOneMoreThanOne);
                        }
                        {
                            bool atLeastOneMoreThanZero = false;

                            const auto interactionParent = morton.getParentIndex(interaction);
                            const auto interactionParentPos = morton.getBoxPosFromIndex(interactionParent);
                            for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                                UASSERTETRUE(std::abs(parentPos[idxDim] - interactionParentPos[idxDim]) <= 1);
                                if(std::abs(pos[idxDim] - posInteraction[idxDim]) > 0){
                                    atLeastOneMoreThanZero = true;
                                }
                            }
                            UASSERTETRUE(atLeastOneMoreThanZero);
                        }
                    }
                }

                {
                    const auto interactionList = morton.getNeighborListForBlock(idx, idxLevel);
                    for(auto interaction : interactionList){
                        const auto posInteraction = morton.getBoxPosFromIndex(interaction);

                        {
                            bool atLeastOneMoreThanZero = false;

                            for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                                UASSERTETRUE(std::abs(pos[idxDim] - posInteraction[idxDim]) <= 1);
                                if(std::abs(pos[idxDim] - posInteraction[idxDim]) > 0){
                                    atLeastOneMoreThanZero = true;
                                }
                            }
                            UASSERTETRUE(atLeastOneMoreThanZero);
                        }
                    }
                }
            }
        }
    }
    
    void SetTests() {
        Parent::AddTest(&TestLipow::TestBasic, "Basic test for Morton");
    }
};

// You must do this
TestClass(TestLipow)


