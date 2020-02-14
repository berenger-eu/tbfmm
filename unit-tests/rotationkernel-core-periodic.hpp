#ifndef TESTKERNEL_CORE_HPP
#define TESTKERNEL_CORE_HPP

#include "UTester.hpp"

#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "kernels/rotationkernel/FRotationKernel.hpp"
#include "algorithms/tbfalgorithmutils.hpp"
#include "utils/tbftimer.hpp"
#include "utils/tbfaccuracychecker.hpp"
#include "algorithms/tbfalgorithmperiodictoptree.hpp"
#include "kernels/counterkernels/tbfinteractionprinter.hpp"


template <class RealType, template <typename T1, typename T2, typename T3> class TestAlgorithmClass>
class TestRotationKernel : public UTester< TestRotationKernel<RealType, TestAlgorithmClass> > {
    using Parent = UTester< TestRotationKernel<RealType, TestAlgorithmClass> >;

    void CorePart(const long int NbParticles, const long int inNbElementsPerBlock,
                  const bool inOneGroupPerParent, const long int TreeHeight){
        const int Dim = 3;

        /////////////////////////////////////////////////////////////////////////////////////////

        const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
        const std::array<RealType, Dim> inBoxCenter{{0.5, 0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, inBoxCenter);

        /////////////////////////////////////////////////////////////////////////////////////////

        TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

        std::vector<std::array<RealType, Dim+1>> particlePositions(NbParticles);

        for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
            auto pos = randomGenerator.getNewItem();
            particlePositions[idxPart][0] = pos[0];
            particlePositions[idxPart][1] = pos[1];
            particlePositions[idxPart][2] = pos[2];
            particlePositions[idxPart][3] = RealType(0.01);
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        const unsigned int P = 12;
        constexpr long int NbDataValuesPerParticle = Dim+1;
        constexpr long int NbRhsValuesPerParticle = 4;

        constexpr long int VectorSize = ((P+2)*(P+1))/2;

        using MultipoleClass = std::array<std::complex<RealType>, VectorSize>;
        using LocalClass = std::array<std::complex<RealType>, VectorSize>;

        using SpacialSystemPeriodic = TbfDefaultSpaceIndexTypePeriodic<RealType>;
        using KernelClass = TbfInteractionPrinter<FRotationKernel<RealType, P, SpacialSystemPeriodic>>;

        const long int LastWorkingLevel = TbfDefaultLastLevelPeriodic;
        using AlgorithmClass = TestAlgorithmClass<RealType, KernelClass, SpacialSystemPeriodic>;
        using TreeClass = TbfTree<RealType,
                                  RealType,
                                  NbDataValuesPerParticle,
                                  RealType,
                                  NbRhsValuesPerParticle,
                                  MultipoleClass,
                                  LocalClass,
                                  SpacialSystemPeriodic>;
        using TopPeriodicAlgorithmClass = TbfAlgorithmPeriodicTopTree<RealType,
                                                                      typename AlgorithmClass::KernelClass,
                                                                      MultipoleClass,
                                                                      LocalClass,
                                                                      SpacialSystemPeriodic>;

        /////////////////////////////////////////////////////////////////////////////////////////

        /*for(long int idxExtraLevel = -1 ; idxExtraLevel < 5 ; ++idxExtraLevel)*/{
            const long int idxExtraLevel = -1; // TODO
            TbfTimer timerBuildTree;

            TreeClass tree(configuration, inNbElementsPerBlock, TbfUtils::make_const(particlePositions), inOneGroupPerParent);

            timerBuildTree.stop();
            std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

            std::unique_ptr<AlgorithmClass> algorithm(new AlgorithmClass(configuration, LastWorkingLevel));
            std::unique_ptr<TopPeriodicAlgorithmClass> topAlgorithm(new TopPeriodicAlgorithmClass(configuration, idxExtraLevel));

            TbfTimer timerExecute;

            // Bottom to top
            algorithm->execute(tree, TbfAlgorithmUtils::TbfBottomToTopStages);
            // Periodic at the top (could be done in parallel with TbfTransferStages)
            topAlgorithm->execute(tree);
            // Transfer (could be done in parallel with topAlgorithm.execute)
            algorithm->execute(tree, TbfAlgorithmUtils::TbfTransferStages);
            // Top to bottom
            algorithm->execute(tree, TbfAlgorithmUtils::TbfTopToBottomStages);

            timerExecute.stop();
            std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

            /////////////////////////////////////////////////////////////////////////////////////////

            {
                std::array<RealType*, 4> particles;
                for(auto& vec : particles){
                    vec = new RealType[NbParticles]();
                }
                std::array<RealType*, NbRhsValuesPerParticle> particlesRhs;
                for(auto& vec : particlesRhs){
                    vec = new RealType[NbParticles]();
                }

                for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                    particles[0][idxPart] = particlePositions[idxPart][0];
                    particles[1][idxPart] = particlePositions[idxPart][1];
                    particles[2][idxPart] = particlePositions[idxPart][2];
                    particles[3][idxPart] = particlePositions[idxPart][3];
                }

                std::array<RealType*, 4> particlesRepeat;
                for(auto& vec : particlesRepeat){
                    vec = new RealType[NbParticles]();
                }

                TbfTimer timerDirect;

                const auto startRepeatInterval = topAlgorithm->getRepetitionsIntervals().first;
                const auto endRepeatInterval = topAlgorithm->getRepetitionsIntervals().second;

                for(long int idxX = startRepeatInterval[0] ; idxX <= endRepeatInterval[0] ; ++idxX){
                    for(long int idxY = startRepeatInterval[1] ; idxY <= endRepeatInterval[1] ; ++idxY){
                        for(long int idxZ = startRepeatInterval[2] ; idxZ <= endRepeatInterval[2] ; ++idxZ){
                            const bool shouldBeDone = ((idxX != 0) || (idxY != 0) || (idxZ != 0));

                            if(shouldBeDone){
                                for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                                    particlesRepeat[0][idxPart] = particles[0][idxPart] + RealType(idxX) * BoxWidths[0];
                                    particlesRepeat[1][idxPart] = particles[1][idxPart] + RealType(idxY) * BoxWidths[1];
                                    particlesRepeat[2][idxPart] = particles[2][idxPart] + RealType(idxZ) * BoxWidths[2];
                                    particlesRepeat[3][idxPart] = particles[3][idxPart];
                                }

                                FP2PR::template GenericFullRemote<RealType>(TbfUtils::make_const(particlesRepeat), NbParticles,
                                                                            particles, particlesRhs, NbParticles);
                            }
                        }
                    }
                }

                FP2PR::template GenericInner<RealType>( particles, particlesRhs, NbParticles);

                timerDirect.stop();

                std::cout << "Direct execute in " << timerDirect.getElapsed() << "s" << std::endl;

                //////////////////////////////////////////////////////////////////////

                std::array<TbfAccuracyChecker<RealType>, 4> partcilesAccuracy;
                std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

                tree.applyToAllLeaves([&particles,&partcilesAccuracy,&particlesRhs,&partcilesRhsAccuracy]
                                      (auto&& leafHeader, const long int* particleIndexes,
                                      const std::array<RealType*, 4> particleDataPtr,
                                      const std::array<RealType*, NbRhsValuesPerParticle> particleRhsPtr){
                    for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                        for(int idxValue = 0 ; idxValue < 4 ; ++idxValue){
                           partcilesAccuracy[idxValue].addValues(particles[idxValue][particleIndexes[idxPart]],
                                                                particleDataPtr[idxValue][idxPart]);
                        }
                        for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                           partcilesRhsAccuracy[idxValue].addValues(particlesRhs[idxValue][particleIndexes[idxPart]],
                                                                particleRhsPtr[idxValue][idxPart]);
                        }
                    }
                });

                std::cout << "Perform the periodic FMM with parameters:" << std::endl;
                std::cout << " - idxExtraLevel: " << idxExtraLevel << std::endl;
                std::cout << " - Nb repeat per dim: " << topAlgorithm->getNbRepetitionsPerDim() << std::endl;
                std::cout << " - Number of times the real box is duplicated: " << topAlgorithm->getNbTotalRepetitions() << std::endl;
                std::cout << " - Repeatition interves: " << TbfUtils::ArrayPrinter(topAlgorithm->getRepetitionsIntervals().first)
                          << " " << TbfUtils::ArrayPrinter(topAlgorithm->getRepetitionsIntervals().second) << std::endl;
                std::cout << " - original configuration: " << configuration << std::endl;
                std::cout << " - top tree configuration: " << TopPeriodicAlgorithmClass::GenerateAboveTreeConfiguration(configuration,idxExtraLevel) << std::endl;

                std::cout << "Relative differences:" << std::endl;
                for(int idxValue = 0 ; idxValue < 4 ; ++idxValue){
                   std::cout << " - Data " << idxValue << " = " << partcilesAccuracy[idxValue] << std::endl;
                   UASSERTETRUE(partcilesAccuracy[idxValue].getRelativeL2Norm() < 1e-16);
                }
                for(int idxValue = 0 ; idxValue < 4 ; ++idxValue){
                   std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
                   if constexpr (std::is_same<float, RealType>::value){
                       UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
                   }
                   else{
                       UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
                   }
                }

                //////////////////////////////////////////////////////////////////////

                for(auto& vec : particles){
                    delete[] vec;
                }
                for(auto& vec : particlesRepeat){
                    delete[] vec;
                }
                for(auto& vec : particlesRhs){
                    delete[] vec;
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////

            {
                std::vector<std::array<RealType, Dim+1>> extendedParticlePositions(NbParticles * topAlgorithm->getNbTotalRepetitions());

                const auto startRepeatInterval = topAlgorithm->getRepetitionsIntervals().first;
                const auto endRepeatInterval = topAlgorithm->getRepetitionsIntervals().second;

                long int idxPartDest = 0;

                for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                    assert(idxPartDest < NbParticles * topAlgorithm->getNbTotalRepetitions());
                    extendedParticlePositions[idxPartDest] = particlePositions[idxPart];
                    idxPartDest += 1;
                }

                for(long int idxX = startRepeatInterval[0] ; idxX <= endRepeatInterval[0] ; ++idxX){
                    for(long int idxY = startRepeatInterval[1] ; idxY <= endRepeatInterval[1] ; ++idxY){
                        for(long int idxZ = startRepeatInterval[2] ; idxZ <= endRepeatInterval[2] ; ++idxZ){
                            if(idxX != 0 || idxY != 0 || idxZ != 0){
                                for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                                    assert(idxPartDest < NbParticles * topAlgorithm->getNbTotalRepetitions());
                                    extendedParticlePositions[idxPartDest][0] = particlePositions[idxPart][0] + RealType(idxX) * BoxWidths[0];
                                    extendedParticlePositions[idxPartDest][1] = particlePositions[idxPart][1] + RealType(idxY) * BoxWidths[1];
                                    extendedParticlePositions[idxPartDest][2] = particlePositions[idxPart][2] + RealType(idxZ) * BoxWidths[2];
                                    extendedParticlePositions[idxPartDest][3] = particlePositions[idxPart][3];
                                    idxPartDest += 1;
                                }
                            }
                        }
                    }
                }

                const auto upperConfiguration = TopPeriodicAlgorithmClass::GenerateAboveTreeConfiguration(configuration,idxExtraLevel);
                const std::array<RealType, Dim> extendedBoxWidths = upperConfiguration.getBoxWidths();
                const std::array<RealType, Dim> extendedBoxCenter = upperConfiguration.getBoxCenter();
                const long int extendedTreeHeight = topAlgorithm->getExtendedLevel(configuration.getTreeHeight());

                const TbfSpacialConfiguration<RealType, Dim> extendedConfiguration(extendedTreeHeight, extendedBoxWidths, extendedBoxCenter);

                std::cout << "Extended configuration:" << std::endl;
                std::cout << extendedConfiguration << std::endl;

                using SpacialSystemNonPeriodic = TbfDefaultSpaceIndexType<RealType>;
                using AlgorithmClassNonPeriodic = TestAlgorithmClass<RealType, KernelClass, SpacialSystemNonPeriodic>;
                using TreeClassNonPeriodic = TbfTree<RealType,
                                          RealType,
                                          NbDataValuesPerParticle,
                                          RealType,
                                          NbRhsValuesPerParticle,
                                          MultipoleClass,
                                          LocalClass,
                                          SpacialSystemNonPeriodic>;

                TreeClassNonPeriodic extendedTree(extendedConfiguration, inNbElementsPerBlock, TbfUtils::make_const(extendedParticlePositions), inOneGroupPerParent);

                std::unique_ptr<AlgorithmClassNonPeriodic> extentedAlgorithm(new AlgorithmClassNonPeriodic(extendedConfiguration, 3));

                extentedAlgorithm->execute(extendedTree);

                //////////////////////////////////////////////////////////////////////

                std::array<TbfAccuracyChecker<RealType>, 4> partcilesAccuracy;
                std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

                tree.applyToAllLeaves([&extendedTree, &configuration, &topAlgorithm, &partcilesAccuracy,&partcilesRhsAccuracy]
                                      (auto&& leafHeader, const long int* particleIndexes,
                                      const std::array<RealType*, 4> particleDataPtr,
                                      const std::array<RealType*, NbRhsValuesPerParticle> particleRhsPtr){
                    long int sameIndex = topAlgorithm->getExtendedIndex(leafHeader.spaceIndex, configuration.getTreeHeight() - 1);
                    auto foundLeaf = extendedTree.findGroupWithLeaf(sameIndex);
                    assert(foundLeaf);

                    auto particlesGoupr = foundLeaf->first;
                    const long int leafIndex = foundLeaf->second;

                    auto leafData = particlesGoupr.get().getParticleData(leafIndex);
                    auto leafRhs = particlesGoupr.get().getParticleRhs(leafIndex);
                    const long int* leafIndexes = particlesGoupr.get().getParticleIndexes(leafIndex);

                    for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
                        assert(leafIndexes[idxPart] == particleIndexes[idxPart]);
                        for(int idxValue = 0 ; idxValue < 4 ; ++idxValue){
                           partcilesAccuracy[idxValue].addValues(leafData[idxValue][idxPart],
                                                                particleDataPtr[idxValue][idxPart]);
                        }
                        for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                           partcilesRhsAccuracy[idxValue].addValues(leafRhs[idxValue][idxPart],
                                                                particleRhsPtr[idxValue][idxPart]);
                        }
                    }
                });

                std::cout << "Relative differences:" << std::endl;
                for(int idxValue = 0 ; idxValue < 4 ; ++idxValue){
                   std::cout << " - Data " << idxValue << " = " << partcilesAccuracy[idxValue] << std::endl;
                   UASSERTETRUE(partcilesAccuracy[idxValue].getRelativeL2Norm() < 1e-16);
                }
                for(int idxValue = 0 ; idxValue < 4 ; ++idxValue){
                   std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
                   if constexpr (std::is_same<float, RealType>::value){
                       UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-5);
                   }
                   else{
                       UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-14);
                   }
                }

                //////////////////////////////////////////////////////////////////////

                TbfAccuracyChecker<RealType> multipoleAccuracy;
                TbfAccuracyChecker<RealType> localAccuracy;

                tree.applyToAllCells([&extendedTree, &configuration, &topAlgorithm, &multipoleAccuracy,&localAccuracy]
                                      (const long int idxLevel, auto&& cellHeader,
                                      const auto& multipoleRef,
                                      const auto& localRef){
                    assert(multipoleRef);
                    assert(localRef);

                    const MultipoleClass& multipole = (*multipoleRef);
                    const LocalClass& local = (*localRef);

                    long int sameIndex = topAlgorithm->getExtendedIndex(cellHeader.spaceIndex, idxLevel); // TODO
                    auto foundCell = extendedTree.findGroupWithCell(topAlgorithm->getExtendedLevel(idxLevel), sameIndex);
                    assert(foundCell);

                    auto cellGoupr = foundCell->first;
                    const long int cellIndex = foundCell->second;

                    auto multipoleData = cellGoupr.get().getCellMultipole(cellIndex);
                    auto localData = cellGoupr.get().getCellLocal(cellIndex);

                    for(int idxMultipole = 0 ; idxMultipole < static_cast<long int>(std::size(multipole)) ; ++idxMultipole){
                        multipoleAccuracy.addValues(multipole[idxMultipole].real(), multipoleData[idxMultipole].real());
                        multipoleAccuracy.addValues(multipole[idxMultipole].imag(), multipoleData[idxMultipole].imag());
                    }

                    for(int idxLocal = 0 ; idxLocal < static_cast<long int>(std::size(local)) ; ++idxLocal){
                        localAccuracy.addValues(local[idxLocal].real(), localData[idxLocal].real());
                        localAccuracy.addValues(local[idxLocal].imag(), localData[idxLocal].imag());
                    }
                });

                std::cout << " - Multipole " << multipoleAccuracy << std::endl;
                if constexpr (std::is_same<float, RealType>::value){
                    UASSERTETRUE(multipoleAccuracy.getRelativeL2Norm() < 9e-5);
                }
                else{
                    UASSERTETRUE(multipoleAccuracy.getRelativeL2Norm() < 9e-14);
                }

                std::cout << " - Local " << localAccuracy << std::endl;
                if constexpr (std::is_same<float, RealType>::value){
                    UASSERTETRUE(localAccuracy.getRelativeL2Norm() < 9e-5);
                }
                else{
                    UASSERTETRUE(localAccuracy.getRelativeL2Norm() < 9e-14);
                }
            }
        }
    }

    void TestBasic() {
//        for(long int idxNbParticles = 1 ; idxNbParticles <= 1000 ; idxNbParticles *= 10){
//            for(const long int idxNbElementsPerBlock : std::vector<long int>{{1, 100, 10000000}}){
//                for(const bool idxOneGroupPerParent : std::vector<bool>{{true, false}}){
//                    for(long int idxTreeHeight = 2 ; idxTreeHeight < 4 ; ++idxTreeHeight){
//                        CorePart(idxNbParticles, idxNbElementsPerBlock, idxOneGroupPerParent, idxTreeHeight);
//                    }
//                }
//            }
//        }
        // TODO
        CorePart(1, 100, true, 2);
    }

    void SetTests() {
        Parent::AddTest(&TestRotationKernel<RealType, TestAlgorithmClass>::TestBasic, "Basic test based on the rotation kernel");
    }
};

#endif
