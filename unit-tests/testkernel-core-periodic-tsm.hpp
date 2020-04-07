#ifndef TESTKERNELPERIODICTSM_CORE_HPP
#define TESTKERNELPERIODICTSM_CORE_HPP

#include "UTester.hpp"

#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftreetsm.hpp"
#include "kernels/testkernel/tbftestkernel.hpp"
#include "algorithms/tbfalgorithmutils.hpp"
#include "algorithms/periodic/tbfalgorithmperiodictoptreetsm.hpp"
#include "utils/tbftimer.hpp"


template <class AlgorithmClassTsm>
class TestTestKernelPeriodicTsm : public UTester< TestTestKernelPeriodicTsm<AlgorithmClassTsm> > {
    using Parent = UTester< TestTestKernelPeriodicTsm<AlgorithmClassTsm> >;
    using RealType = typename AlgorithmClassTsm::RealType;

    void CorePart(const long int NbParticles, const long int NbElementsPerBlock,
                  const bool OneGroupPerParent, const long int TreeHeight){
        const int Dim = 3;

        /////////////////////////////////////////////////////////////////////////////////////////

        const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
        const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

        /////////////////////////////////////////////////////////////////////////////////////////

        TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

        std::vector<std::array<RealType, Dim>> particlePositionsSource(NbParticles);
        std::vector<std::array<RealType, Dim>> particlePositionsTarget(NbParticles);

        for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
            particlePositionsSource[idxPart] = randomGenerator.getNewItem();
            particlePositionsTarget[idxPart] = randomGenerator.getNewItem();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        constexpr long int NbDataValuesPerParticle = Dim;
        constexpr long int NbRhsValuesPerParticle = 1;
        using MultipoleClass = std::array<long int,1>;
        using LocalClass = std::array<long int,1>;
        using SpacialSystemPeriodic = TbfDefaultSpaceIndexTypePeriodic<RealType>;
        const long int LastWorkingLevel = TbfDefaultLastLevelPeriodic;
        using TreeClassTsm = TbfTreeTsm<RealType,
                                  RealType,
                                  NbDataValuesPerParticle,
                                  long int,
                                  NbRhsValuesPerParticle,
                                  MultipoleClass,
                                  LocalClass,
                                  SpacialSystemPeriodic>;
        using TopPeriodicAlgorithmClassTsm = TbfAlgorithmPeriodicTopTreeTsm<RealType,
                                                                      typename AlgorithmClassTsm::KernelClass,
                                                                      MultipoleClass,
                                                                      LocalClass,
                                                                      SpacialSystemPeriodic>;


        /////////////////////////////////////////////////////////////////////////////////////////

        for(long int idxExtraLevel = -1 ; idxExtraLevel < 5 ; ++idxExtraLevel){
            TreeClassTsm tree(configuration, particlePositionsSource, particlePositionsTarget, NbElementsPerBlock, OneGroupPerParent);

            AlgorithmClassTsm algorithm(configuration, LastWorkingLevel);
            TopPeriodicAlgorithmClassTsm topAlgorithm(configuration, idxExtraLevel);

            TbfTimer timerExecute;

            // Bottom to top
            algorithm.execute(tree, TbfAlgorithmUtils::TbfBottomToTopStages);
            // Periodic at the top (could be done in parallel with TbfTransferStages)
            topAlgorithm.execute(tree);
            // Transfer (could be done in parallel with topAlgorithm.execute)
            algorithm.execute(tree, TbfAlgorithmUtils::TbfTransferStages);
            // Top to bottom
            algorithm.execute(tree, TbfAlgorithmUtils::TbfTopToBottomStages);

            timerExecute.stop();
            std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

            const long int nbRepeatitionInTotal = topAlgorithm.getNbTotalRepetitions();

            tree.applyToAllLeavesTarget([this, nbRepeatitionInTotal, NbParticles](auto&& leafHeader, const long int* /*particleIndexes*/,
                                  const std::array<RealType*, Dim> /*particleDataPtr*/, const std::array<long int*, 1> particleRhsPtr){
                for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                    UASSERTEEQUAL(particleRhsPtr[0][idxPart], (nbRepeatitionInTotal*NbParticles));
                }
            });

            std::cout << "Perform the periodic FMM with parameters:" << std::endl;
            std::cout << " - idxExtraLevel: " << idxExtraLevel << std::endl;
            std::cout << " - Number of repeat per dim: " << topAlgorithm.getNbRepetitionsPerDim() << std::endl;
            std::cout << " - Number of times the real box is duplicated: " << topAlgorithm.getNbTotalRepetitions() << std::endl;
            std::cout << " - Repeatition interves: " << TbfUtils::ArrayPrinter(topAlgorithm.getRepetitionsIntervals().first)
                      << " " << TbfUtils::ArrayPrinter(topAlgorithm.getRepetitionsIntervals().second) << std::endl;
            std::cout << " - original configuration: " << configuration << std::endl;
            std::cout << " - top tree configuration: " << TopPeriodicAlgorithmClassTsm::GenerateAboveTreeConfiguration(configuration,idxExtraLevel) << std::endl;
        }

        /////////////////////////////////////////////////////////////////////////////////////////

    }

    void TestBasic() {
        for(long int idxNbParticles = 1 ; idxNbParticles <= 10000 ; idxNbParticles *= 10){
            for(const long int idxNbElementsPerBlock : std::vector<long int>{{100, 10000000}}){
                for(const bool idxOneGroupPerParent : std::vector<bool>{{true, false}}){
                    for(long int idxTreeHeight = 2 ; idxTreeHeight < 5 ; ++idxTreeHeight){
                        CorePart(idxNbParticles, idxNbElementsPerBlock, idxOneGroupPerParent, idxTreeHeight);
                    }
                }
            }
        }


        for(long int idxNbParticles = 1 ; idxNbParticles <= 1000 ; idxNbParticles *= 10){
            for(const long int idxNbElementsPerBlock : std::vector<long int>{{1, 100, 10000000}}){
                for(const bool idxOneGroupPerParent : std::vector<bool>{{true, false}}){
                    const long int idxTreeHeight = 3;
                    CorePart(idxNbParticles, idxNbElementsPerBlock, idxOneGroupPerParent, idxTreeHeight);
                }
            }
        }
    }

    void SetTests() {
        Parent::AddTest(&TestTestKernelPeriodicTsm<AlgorithmClassTsm>::TestBasic, "Basic test based on the test kernel with periodicity and tsm");
    }
};

#endif
