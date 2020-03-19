#include "UTester.hpp"

#include "utils/tbfutils.hpp"
#include "spacial/tbfhilbertspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <set>

class TestHilbert : public UTester< TestHilbert > {
    using Parent = UTester< TestHilbert >;

    void TestBasic() {
        const int Dim = 3;
        const long int TreeHeight = 5;

        using RealType = double;

        const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
        const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
        const TbfHilbertSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim> > hilbert(configuration);

        using IndexType = typename TbfHilbertSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim> >::IndexType;

        UASSERTEEQUAL(hilbert.getUpperBound(0), IndexType(1));
        UASSERTEEQUAL(hilbert.getUpperBound(1), IndexType(hilbert.getNbChildrenPerCell()));
        UASSERTEEQUAL(hilbert.getUpperBound(2), IndexType(hilbert.getNbChildrenPerCell()*hilbert.getNbChildrenPerCell()));

        for(long int idxLevel = 0 ; idxLevel < TreeHeight-1 ; ++idxLevel){
            std::set<std::array<long int,Dim>> generatedPositions;
            std::set<IndexType> generatedChildIndexes;

            for(IndexType idx = 0 ; idx < hilbert.getUpperBound(idxLevel) ; ++idx){
                auto pos = hilbert.getBoxPosFromIndex(idx);
                UASSERTETRUE(generatedPositions.find(pos) == generatedPositions.end());
                generatedPositions.insert(pos);

                UASSERTETRUE(idx == hilbert.getIndexFromBoxPos(pos));

                for(long int idxChild = 0 ; idxChild < hilbert.getNbChildrenPerCell() ; ++idxChild){
                    auto childIndex = hilbert.getChildIndexFromParent(idx, idxChild);
                    UASSERTETRUE(idxChild == hilbert.childPositionFromParent(childIndex));
                    UASSERTETRUE(idx == hilbert.getParentIndex(childIndex));
                    UASSERTETRUE(childIndex < hilbert.getUpperBound(idxLevel+1));
                    UASSERTETRUE(generatedChildIndexes.find(childIndex) == generatedChildIndexes.end());
                    generatedChildIndexes.insert(childIndex);
                }

                {
                    const auto parent = hilbert.getParentIndex(idx);
                    const auto parentPos = hilbert.getBoxPosFromIndex(parent);

                    const auto interactionList = hilbert.getInteractionListForIndex(idx, idxLevel);
                    for(auto interaction : interactionList){
                        const auto posInteraction = hilbert.getBoxPosFromIndex(interaction);

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

                            const auto interactionParent = hilbert.getParentIndex(interaction);
                            const auto interactionParentPos = hilbert.getBoxPosFromIndex(interactionParent);
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
                    const auto interactionList = hilbert.getNeighborListForIndex(idx, idxLevel);
                    for(auto interaction : interactionList){
                        const auto posInteraction = hilbert.getBoxPosFromIndex(interaction);

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
        Parent::AddTest(&TestHilbert::TestBasic, "Basic test for Hilbert");
    }
};

// You must do this
TestClass(TestHilbert)


