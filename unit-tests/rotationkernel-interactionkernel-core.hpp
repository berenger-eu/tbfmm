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
#include "kernels/counterkernels/tbfinteractioncounter.hpp"
#include "kernels/counterkernels/tbfinteractiontimer.hpp"


template <class RealType, template <typename T1, typename T2, typename T3> class TestAlgorithmClass>
class TestRotationKernelInteraction : public UTester< TestRotationKernelInteraction<RealType, TestAlgorithmClass> > {
    using Parent = UTester< TestRotationKernelInteraction<RealType, TestAlgorithmClass> >;

    template <template <typename T3> class KernelInteractionCounter>
    void CorePart(const long int NbParticles, const long int NbElementsPerBlock,
                  const bool OneGroupPerParent, const long int TreeHeight){
        const int Dim = 3;

        /////////////////////////////////////////////////////////////////////////////////////////

        const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
        const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

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

        using KernelClass = KernelInteractionCounter<FRotationKernel<RealType, P>>;

        using AlgorithmClass = TestAlgorithmClass<RealType, KernelClass, TbfDefaultSpaceIndexType<RealType> >;

        using TreeClass = TbfTree<RealType,
                                  RealType,
                                  NbDataValuesPerParticle,
                                  RealType,
                                  NbRhsValuesPerParticle,
                                  MultipoleClass,
                                  LocalClass>;

        /////////////////////////////////////////////////////////////////////////////////////////

        TbfTimer timerBuildTree;

        TreeClass tree(configuration, TbfUtils::make_const(particlePositions), NbElementsPerBlock, OneGroupPerParent);

        timerBuildTree.stop();
        std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

        std::unique_ptr<AlgorithmClass> algorithm(new AlgorithmClass(configuration));

        TbfTimer timerExecute;

        algorithm->execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////

        auto counters = typename KernelClass::ReduceType();

        algorithm->applyToAllKernels([&](const auto& inKernel){
            counters = KernelClass::ReduceType::Reduce(counters, inKernel.getReduceData());
        });

        std::cout << counters << std::endl;

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

            TbfTimer timerDirect;

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
            for(auto& vec : particlesRhs){
                delete[] vec;
            }
        }

    }

    void TestBasic() {
        for(long int idxTreeHeight = 1 ; idxTreeHeight < 4 ; ++idxTreeHeight){
            const long int nbParticles = 1000;
            const long int nbElementsPerBlock = 100;
            const bool oneGroupPerParent = false;
            CorePart<TbfInteractionCounter>(nbParticles, nbElementsPerBlock, oneGroupPerParent, idxTreeHeight);
            CorePart<TbfInteractionTimer>(nbParticles, nbElementsPerBlock, oneGroupPerParent, idxTreeHeight);
        }
    }

    void SetTests() {
        Parent::AddTest(&TestRotationKernelInteraction<RealType, TestAlgorithmClass>::TestBasic, "Basic test based on the rotation kernel");
    }
};

#endif
