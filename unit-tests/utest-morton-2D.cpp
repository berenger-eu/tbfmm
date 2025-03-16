#include "UTester.hpp"

#include "utils/tbfutils.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"

#include <set>
#include <math.h>

class TestMorton : public UTester<TestMorton>
{
    using Parent = UTester<TestMorton>;

    void TestBasic()
    {
        const int Dim = 2;
        const long int TreeHeight = 5;

        using RealType = double;

        const std::array<RealType, Dim> BoxWidths{{1, 1}};
        const std::array<RealType, Dim> BoxCenter{{0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
        const TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>> morton(configuration);

        using IndexType = typename TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>>::IndexType;

        for (long int idxLevel = 0; idxLevel < TreeHeight - 1; ++idxLevel)
        {
            UASSERTEEQUAL(morton.getUpperBound(idxLevel), IndexType(std::pow(morton.getNbChildrenPerCell(), idxLevel)));
        }

        for (long int idxLevel = 0; idxLevel < TreeHeight - 1; ++idxLevel)
        {
            std::set<std::array<long int, Dim>> generatedPositions;
            std::set<IndexType> generatedChildIndexes;

            for (IndexType idx = 0; idx < morton.getUpperBound(idxLevel); ++idx)
            {
                auto pos = morton.getBoxPosFromIndex(idx);
                UASSERTETRUE(generatedPositions.find(pos) == generatedPositions.end());
                generatedPositions.insert(pos);

                UASSERTETRUE(idx == morton.getIndexFromBoxPos(pos));

                for (long int idxChild = 0; idxChild < morton.getNbChildrenPerCell(); ++idxChild)
                {
                    auto childIndex = morton.getChildIndexFromParent(idx, idxChild);
                    UASSERTETRUE(idxChild == morton.childPositionFromParent(childIndex));
                    UASSERTETRUE(idx == morton.getParentIndex(childIndex));
                    UASSERTETRUE(childIndex < morton.getUpperBound(idxLevel + 1));
                    UASSERTETRUE(generatedChildIndexes.find(childIndex) == generatedChildIndexes.end());
                    generatedChildIndexes.insert(childIndex);
                }

                {
                    const auto parent = morton.getParentIndex(idx);
                    const auto parentPos = morton.getBoxPosFromIndex(parent);

                    const auto interactionList = morton.getInteractionListForIndex(idx, idxLevel);
                    for (auto interaction : interactionList)
                    {
                        const auto posInteraction = morton.getBoxPosFromIndex(interaction);

                        {
                            bool atLeastOneMoreThanOne = false;

                            for (int idxDim = 0; idxDim < Dim; ++idxDim)
                            {
                                UASSERTETRUE(std::abs(pos[idxDim] - posInteraction[idxDim]) <= 3);
                                if (std::abs(pos[idxDim] - posInteraction[idxDim]) > 1)
                                {
                                    atLeastOneMoreThanOne = true;
                                }
                            }
                            UASSERTETRUE(atLeastOneMoreThanOne);
                        }
                        {
                            bool atLeastOneMoreThanZero = false;

                            const auto interactionParent = morton.getParentIndex(interaction);
                            const auto interactionParentPos = morton.getBoxPosFromIndex(interactionParent);
                            for (int idxDim = 0; idxDim < Dim; ++idxDim)
                            {
                                UASSERTETRUE(std::abs(parentPos[idxDim] - interactionParentPos[idxDim]) <= 1);
                                if (std::abs(pos[idxDim] - posInteraction[idxDim]) > 0)
                                {
                                    atLeastOneMoreThanZero = true;
                                }
                            }
                            UASSERTETRUE(atLeastOneMoreThanZero);
                        }
                    }
                }

                {
                    const auto interactionList = morton.getNeighborListForIndex(idx, idxLevel);
                    for (auto interaction : interactionList)
                    {
                        const auto posInteraction = morton.getBoxPosFromIndex(interaction);

                        {
                            bool atLeastOneMoreThanZero = false;

                            for (int idxDim = 0; idxDim < Dim; ++idxDim)
                            {
                                UASSERTETRUE(std::abs(pos[idxDim] - posInteraction[idxDim]) <= 1);
                                if (std::abs(pos[idxDim] - posInteraction[idxDim]) > 0)
                                {
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

    void TestBasicPeriodic()
    {
        const int Dim = 2;
        const long int TreeHeight = 5;

        using RealType = double;

        const std::array<RealType, Dim> BoxWidths{{1, 1}};
        const std::array<RealType, Dim> BoxCenter{{0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
        const TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>, true> morton(configuration);

        using IndexType = typename TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>, true>::IndexType;

        for (long int idxLevel = 0; idxLevel < TreeHeight - 1; ++idxLevel)
            UASSERTEEQUAL(morton.getUpperBound(idxLevel), IndexType(std::pow(morton.getNbChildrenPerCell(), idxLevel)));

        for (long int idxLevel = 3; idxLevel < TreeHeight - 1; ++idxLevel)
        {
            std::set<std::array<long int, Dim>> generatedPositions;
            std::set<IndexType> generatedChildIndexes;

            for (IndexType idx = 0; idx < morton.getUpperBound(idxLevel); ++idx)
            {
                auto pos = morton.getBoxPosFromIndex(idx);
                UASSERTETRUE(generatedPositions.find(pos) == generatedPositions.end());
                generatedPositions.insert(pos);

                UASSERTETRUE(idx == morton.getIndexFromBoxPos(pos));

                for (long int idxChild = 0; idxChild < morton.getNbChildrenPerCell(); ++idxChild)
                {
                    auto childIndex = morton.getChildIndexFromParent(idx, idxChild);
                    UASSERTETRUE(idxChild == morton.childPositionFromParent(childIndex));
                    UASSERTETRUE(idx == morton.getParentIndex(childIndex));
                    UASSERTETRUE(childIndex < morton.getUpperBound(idxLevel + 1));
                    UASSERTETRUE(generatedChildIndexes.find(childIndex) == generatedChildIndexes.end());
                    generatedChildIndexes.insert(childIndex);
                }

                {
                    const auto interactionList = morton.getInteractionListForIndex(idx, idxLevel);
                    UASSERTEEQUAL(static_cast<long int>(std::size(interactionList)), morton.getNbInteractionsPerCell());
                }

                {
                    const auto interactionList = morton.getNeighborListForIndex(idx, idxLevel);
                    UASSERTEEQUAL(static_cast<long int>(std::size(interactionList)), morton.getNbNeighborsPerLeaf());
                }
            }
        }
    }

    void SetTests()
    {
        Parent::AddTest(&TestMorton::TestBasic, "Basic test for Morton");
        Parent::AddTest(&TestMorton::TestBasicPeriodic, "Basic test for Morton (periodic)");
    }
};

// You must do this
TestClass(TestMorton)
