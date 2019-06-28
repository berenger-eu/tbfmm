#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "kernels/testkernel/tbftestkernel.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "utils/tbftimer.hpp"

/// - add uniform kernels
/// - add GPU kernels
/// - do unit tests
/// - periodicity
/// - ensure code quality with visilibity of Template

#include <iostream>


int main(){
    using RealType = double;
    const int Dim = 3;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
    const long int TreeHeight = 8;
    const std::array<RealType, Dim> inBoxCenter{{0.5, 0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, inBoxCenter);

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = 1000;

    TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

    std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);

    for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        particlePositions[idxPart] = randomGenerator.getNewItem();
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    constexpr long int NbDataValuesPerParticle = Dim;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;

    TbfDefaultSpaceIndexType<RealType> spacialSystem(configuration);

    {
        TbfParticlesContainer<RealType, RealType, NbDataValuesPerParticle, long int, NbRhsValuesPerParticle> particles(spacialSystem, particlePositions);

        std::vector<typename TbfDefaultSpaceIndexType<RealType>::IndexType> leafIndexes(particles.getNbLeaves());

        for(long int idxLeaf = 0 ; idxLeaf < particles.getNbLeaves() ; ++idxLeaf){
            leafIndexes[idxLeaf] = particles.getLeafSpacialIndex(idxLeaf);
        }

        TbfCellsContainer<RealType, MultipoleClass, LocalClass> cells(leafIndexes);
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int inNbElementsPerBlock = 50;
    const bool inOneGroupPerParent = false;

    if(TreeHeight > 2){
        TbfTimer timerBuildTree;

        TbfTree<RealType, RealType, NbDataValuesPerParticle, long int, NbRhsValuesPerParticle, MultipoleClass, LocalClass> tree(configuration, inNbElementsPerBlock,
                                                                                    particlePositions, inOneGroupPerParent);

        timerBuildTree.stop();
        std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

        TbfAlgorithm<RealType, TbfTestKernel<RealType>> algorithm(configuration);
        algorithm.execute(tree, TbfAlgorithmUtils::LFmmP2M | TbfAlgorithmUtils::LFmmM2M | TbfAlgorithmUtils::LFmmM2L);

        tree.applyToAllLeaves([&tree, NbParticles](auto&& leafHeader, const long int* /*particleIndexes*/,
                              const std::array<RealType*, Dim> /*particleDataPtr*/, const std::array<long int*, 1> /*particleRhsPtr*/){
            auto groupForCell = tree.findGroupWithCell(TreeHeight-1, leafHeader.spaceIndex);

            if(!groupForCell){
                throw std::runtime_error("Test 1 -- P2M There must be a cell for each leaf at index " + std::to_string(leafHeader.spaceIndex));
            }

            auto multipoleData = (*groupForCell).first.get().getCellMultipole((*groupForCell).second);

            assert(leafHeader.nbParticles);

            if(multipoleData[0] != leafHeader.nbParticles){
                throw std::runtime_error("Test 1 -- P2M Invalide number of particles, should be " + std::to_string(leafHeader.nbParticles)
                        + " is " + std::to_string(multipoleData[0]));
            }
        });

        tree.applyToAllCells([&spacialSystem,&tree](const long int inLevel, auto&& cellHeader,
                             const std::optional<std::reference_wrapper<MultipoleClass>> cellMultipole,
                             const std::optional<std::reference_wrapper<LocalClass>> /*cellLocalr*/){
            if(1 < inLevel && inLevel < TreeHeight-1){
                const int NbChild = (1 << Dim);
                long int totalSum = 0;
                for(long int idxChild = 0 ; idxChild < NbChild ; ++idxChild){
                    auto indexChild = spacialSystem.getChildIndexFromParent(cellHeader.spaceIndex, idxChild);
                    auto groupForCell = tree.findGroupWithCell(inLevel+1, indexChild);
                    if(groupForCell){
                        auto multipoleData = (*groupForCell).first.get().getCellMultipole((*groupForCell).second);
                        totalSum += multipoleData[0];
                    }
                }

                assert(totalSum);

                if(cellMultipole && (*cellMultipole).get()[0] != totalSum){
                    throw std::runtime_error("Test 1 -- M2M Invalide number of particles, should be " + std::to_string(totalSum)
                            + " is " + std::to_string((*cellMultipole).get()[0]) + " for index " + std::to_string(cellHeader.spaceIndex)
                            + " at level " + std::to_string(inLevel));
                }
            }
        });

        tree.applyToAllCells([&spacialSystem,&tree](const long int inLevel, auto&& cellHeader,
                             const std::optional<std::reference_wrapper<MultipoleClass>> /*cellMultipole*/,
                             const std::optional<std::reference_wrapper<LocalClass>> cellLocalr){
            auto indexes = spacialSystem.getInteractionListForIndex(cellHeader.spaceIndex, inLevel);
            long int totalSum = 0;
            for(auto index : indexes){
                auto groupForCell = tree.findGroupWithCell(inLevel, index);
                if(groupForCell){
                    auto multipoleData = (*groupForCell).first.get().getCellMultipole((*groupForCell).second);
                    totalSum += multipoleData[0];
                }
            }
            if(cellLocalr && (*cellLocalr).get()[0] != totalSum){
                throw std::runtime_error("Test 1 -- M2L Invalide number of particles, should be " + std::to_string(totalSum)
                        + " is " + std::to_string((*cellLocalr).get()[0]));
            }
        });
    }

    {
        TbfTree<RealType, RealType, NbDataValuesPerParticle, long int, NbRhsValuesPerParticle, MultipoleClass, LocalClass> tree(configuration, inNbElementsPerBlock,
                                                                                    particlePositions, inOneGroupPerParent);

        TbfAlgorithm<RealType, TbfTestKernel<RealType>> algorithm(configuration);
        algorithm.execute(tree, TbfAlgorithmUtils::LFmmP2P);

        tree.applyToAllLeaves([&tree, &spacialSystem](auto&& leafHeader, const long int* /*particleIndexes*/,
                              const std::array<RealType*, Dim> /*particleDataPtr*/, const std::array<long int*, 1> particleRhsPtr){
            auto indexes = spacialSystem.getNeighborListForBlock(leafHeader.spaceIndex, TreeHeight-1);
            long int totalSum = 0;
            for(auto index : indexes){
                auto groupForLeaf = tree.findGroupWithLeaf(index);
                if(groupForLeaf){
                    auto nbParticles = (*groupForLeaf).first.get().getNbParticlesInLeaf((*groupForLeaf).second);
                    totalSum += nbParticles;
                }
            }

            totalSum += leafHeader.nbParticles - 1;

            for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                if(particleRhsPtr[0][idxPart] != totalSum){
                    throw std::runtime_error("Test 2 -- P2P Final Invalide number of particles, should be " + std::to_string(totalSum)
                            + " is " + std::to_string(particleRhsPtr[0][idxPart]));
                }
            }
        });
    }

    {
        TbfTree<RealType, RealType, NbDataValuesPerParticle, long int, NbRhsValuesPerParticle, MultipoleClass, LocalClass> tree(configuration, inNbElementsPerBlock,
                                                                                    particlePositions, inOneGroupPerParent);

        TbfAlgorithm<RealType, TbfTestKernel<RealType>> algorithm(configuration);
        algorithm.execute(tree);

        tree.applyToAllCells([&spacialSystem,&tree](const long int inLevel, auto&& cellHeader,
                             const std::optional<std::reference_wrapper<MultipoleClass>> /*cellMultipole*/,
                             const std::optional<std::reference_wrapper<LocalClass>> cellLocalr){
            auto indexes = spacialSystem.getInteractionListForIndex(cellHeader.spaceIndex, inLevel);
            long int totalSum = 0;
            for(auto index : indexes){
                auto groupForCell = tree.findGroupWithCell(inLevel, index);
                if(groupForCell){
                    auto multipoleData = (*groupForCell).first.get().getCellMultipole((*groupForCell).second);
                    totalSum += multipoleData[0];
                }
            }
            if(2 < inLevel){
                auto parentIndex = spacialSystem.getParentIndex(cellHeader.spaceIndex);
                auto groupForCell = tree.findGroupWithCell(inLevel-1, parentIndex);
                if(!groupForCell){
                    throw std::runtime_error("Test 3 -- L2L+M2L Each cell must have a parent");
                }

                auto parentLocalData = (*groupForCell).first.get().getCellLocal((*groupForCell).second);
                totalSum += parentLocalData[0];
            }
            if(cellLocalr && (*cellLocalr).get()[0] != totalSum){
                throw std::runtime_error("Test 3 -- L2L+M2L Invalide number of particles, should be " + std::to_string(totalSum)
                        + " is " + std::to_string((*cellLocalr).get()[0]));
            }
        });


        tree.applyToAllLeaves([NbParticles](auto&& leafHeader, const long int* /*particleIndexes*/,
                              const std::array<RealType*, Dim> /*particleDataPtr*/, const std::array<long int*, 1> particleRhsPtr){
            for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                if(particleRhsPtr[0][idxPart] != NbParticles-1){
                    throw std::runtime_error("Test 3 -- Final Invalide number of particles, should be " + std::to_string(NbParticles-1)
                            + " is " + std::to_string(particleRhsPtr[0][idxPart]));
                }
            }
        });
    }

    return 0;
}

