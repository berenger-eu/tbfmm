#include "UTester.hpp"

#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "kernels/P2P/FP2PR.hpp"
#include "algorithms/tbfalgorithmutils.hpp"
#include "utils/tbfaccuracychecker.hpp"


class TestP2P : public UTester< TestP2P > {
    using Parent = UTester< TestP2P >;

    template <class RealType, const long int NbParticles>
    void TestBasic() {
        {
            const int Dim = 3;
            const long int NbRhsValuesPerParticle = 4;

            /////////////////////////////////////////////////////////////////////////////////////////

            const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};

            /////////////////////////////////////////////////////////////////////////////////////////

            TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

            std::vector<std::array<RealType, Dim+1>> particlePositions(NbParticles);

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                auto pos = randomGenerator.getNewItem();
                particlePositions[idxPart][0] = pos[0];
                particlePositions[idxPart][1] = pos[1];
                particlePositions[idxPart][2] = pos[2];
                particlePositions[idxPart][3] = RealType(0.01);
            }

            /////////////////////////////////////////////////////////////////////////////////////////

            std::array<RealType*, 4> particles;
            for(auto& vec : particles){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, NbRhsValuesPerParticle> particlesRhs;
            for(auto& vec : particlesRhs){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar;
            for(auto& vec : particlesRhsScalar){
                vec = new RealType[NbParticles]();
            }

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                particles[0][idxPart] = particlePositions[idxPart][0];
                particles[1][idxPart] = particlePositions[idxPart][1];
                particles[2][idxPart] = particlePositions[idxPart][2];
                particles[3][idxPart] = particlePositions[idxPart][3];
            }

            FP2PR::template GenericInner<RealType>( particles, particlesRhs, NbParticles);
            FP2PR::template GenericInnerScalar<RealType>( particles, particlesRhsScalar, NbParticles);

            /////////////////////////////////////////////////////////////////////////////////////////

            std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                   partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar[idxValue][idxPart],
                                                            particlesRhs[idxValue][idxPart]);
                }
            }

            for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
               std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
               if constexpr (std::is_same<float, RealType>::value){
                   UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
               }
               else{
                   UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
               }
            }
        }
        {
            const int Dim = 3;
            const long int NbRhsValuesPerParticle = 4;

            /////////////////////////////////////////////////////////////////////////////////////////

            const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};

            /////////////////////////////////////////////////////////////////////////////////////////

            TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

            std::vector<std::array<RealType, Dim+1>> particlePositions1(NbParticles);
            std::vector<std::array<RealType, Dim+1>> particlePositions2(NbParticles);

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                {
                    auto pos = randomGenerator.getNewItem();
                    particlePositions1[idxPart][0] = pos[0];
                    particlePositions1[idxPart][1] = pos[1];
                    particlePositions1[idxPart][2] = pos[2];
                    particlePositions1[idxPart][3] = RealType(0.01);
                }
                {
                    auto pos = randomGenerator.getNewItem();
                    particlePositions2[idxPart][0] = pos[0];
                    particlePositions2[idxPart][1] = pos[1];
                    particlePositions2[idxPart][2] = pos[2];
                    particlePositions2[idxPart][3] = RealType(0.02);
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////

            std::array<RealType*, 4> particles1;
            for(auto& vec : particles1){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, 4> particles2;
            for(auto& vec : particles2){
                vec = new RealType[NbParticles]();
            }

            std::array<RealType*, NbRhsValuesPerParticle> particlesRhs1;
            for(auto& vec : particlesRhs1){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar1;
            for(auto& vec : particlesRhsScalar1){
                vec = new RealType[NbParticles]();
            }

            std::array<RealType*, NbRhsValuesPerParticle> particlesRhs2;
            for(auto& vec : particlesRhs2){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar2;
            for(auto& vec : particlesRhsScalar2){
                vec = new RealType[NbParticles]();
            }

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                particles1[0][idxPart] = particlePositions1[idxPart][0];
                particles1[1][idxPart] = particlePositions1[idxPart][1];
                particles1[2][idxPart] = particlePositions1[idxPart][2];
                particles1[3][idxPart] = particlePositions1[idxPart][3];
                particles2[0][idxPart] = particlePositions2[idxPart][0];
                particles2[1][idxPart] = particlePositions2[idxPart][1];
                particles2[2][idxPart] = particlePositions2[idxPart][2];
                particles2[3][idxPart] = particlePositions2[idxPart][3];
            }

            FP2PR::template FullMutual<RealType>( particles1, particlesRhs1, NbParticles,  particles2, particlesRhs2, NbParticles);
            FP2PR::template FullMutualScalar<RealType>( particles1, particlesRhsScalar1, NbParticles,  particles2, particlesRhsScalar2, NbParticles);

            /////////////////////////////////////////////////////////////////////////////////////////

            std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                   partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar1[idxValue][idxPart],
                                                            particlesRhs1[idxValue][idxPart]);
                   partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar2[idxValue][idxPart],
                                                            particlesRhs2[idxValue][idxPart]);
                }
            }

            for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
               std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
               if constexpr (std::is_same<float, RealType>::value){
                   UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
               }
               else{
                   UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
               }
            }
        }
        {
            const int Dim = 3;
            const long int NbRhsValuesPerParticle = 4;

            /////////////////////////////////////////////////////////////////////////////////////////

            const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};

            /////////////////////////////////////////////////////////////////////////////////////////

            TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

            std::vector<std::array<RealType, Dim+1>> particlePositions1(NbParticles);
            std::vector<std::array<RealType, Dim+1>> particlePositions2(NbParticles);

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                {
                    auto pos = randomGenerator.getNewItem();
                    particlePositions1[idxPart][0] = pos[0];
                    particlePositions1[idxPart][1] = pos[1];
                    particlePositions1[idxPart][2] = pos[2];
                    particlePositions1[idxPart][3] = RealType(0.01);
                }
                {
                    auto pos = randomGenerator.getNewItem();
                    particlePositions2[idxPart][0] = pos[0];
                    particlePositions2[idxPart][1] = pos[1];
                    particlePositions2[idxPart][2] = pos[2];
                    particlePositions2[idxPart][3] = RealType(0.02);
                }
            }

            /////////////////////////////////////////////////////////////////////////////////////////

            std::array<RealType*, 4> particles1;
            for(auto& vec : particles1){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, 4> particles2;
            for(auto& vec : particles2){
                vec = new RealType[NbParticles]();
            }

            std::array<RealType*, NbRhsValuesPerParticle> particlesRhs2;
            for(auto& vec : particlesRhs2){
                vec = new RealType[NbParticles]();
            }
            std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar2;
            for(auto& vec : particlesRhsScalar2){
                vec = new RealType[NbParticles]();
            }

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                particles1[0][idxPart] = particlePositions1[idxPart][0];
                particles1[1][idxPart] = particlePositions1[idxPart][1];
                particles1[2][idxPart] = particlePositions1[idxPart][2];
                particles1[3][idxPart] = particlePositions1[idxPart][3];
                particles2[0][idxPart] = particlePositions2[idxPart][0];
                particles2[1][idxPart] = particlePositions2[idxPart][1];
                particles2[2][idxPart] = particlePositions2[idxPart][2];
                particles2[3][idxPart] = particlePositions2[idxPart][3];
            }

            FP2PR::template GenericFullRemote<RealType>( particles1, NbParticles,  particles2, particlesRhs2, NbParticles);
            FP2PR::template GenericFullRemoteScalar<RealType>( particles1, NbParticles,  particles2, particlesRhsScalar2, NbParticles);

            /////////////////////////////////////////////////////////////////////////////////////////

            std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

            for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
                for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
                   partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar2[idxValue][idxPart],
                                                            particlesRhs2[idxValue][idxPart]);
                }
            }

            for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
               std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
               if constexpr (std::is_same<float, RealType>::value){
                   UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
               }
               else{
                   UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
               }
            }
        }
    }

    void SetTests() {
        Parent::AddTest(&TestP2P::TestBasic<float,7>, "Basic test for P2P float");
        Parent::AddTest(&TestP2P::TestBasic<double,3>, "Basic test for P2P double");
        Parent::AddTest(&TestP2P::TestBasic<float,1000>, "Basic test for P2P float");
        Parent::AddTest(&TestP2P::TestBasic<double,1000>, "Basic test for P2P double");
    }
};

// You must do this
TestClass(TestP2P)


