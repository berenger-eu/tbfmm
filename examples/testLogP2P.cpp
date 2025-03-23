
#include <iostream>
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"

#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"

#include "kernels/logkernel/tbflogkernel.hpp"
#include "kernels/P2P/FP2PLog.hpp"
#include "algorithms/tbfalgorithmutils.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"

#include "utils/tbfaccuracychecker.hpp"
#include "utils/tbftimer.hpp"
#include "utils/tbfparams.hpp"

class TestP2P
{
public:
    template <class RealType, const long int NbParticles>
    static void TestBasic()
    {
        // {
        //     if (TbfParams::ExistParameter(argc, argv, {"-h", "--help"}))
        //     {
        //         std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        //         std::cout << "[HELP] where params are:" << std::endl;
        //         std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        //         std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        //         std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        //         return 1;
        //     }

        //     const int Dim = 2;
        //     const long int NbRhsValuesPerParticle = 1;

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     const std::array<RealType, Dim> BoxWidths{{1, 1}};
        //     const long int TreeHeight = TbfParams::GetValue<long int>(argc, argv, {"-th", "--tree-height"}, 8);
        //     const std::array<RealType, Dim> BoxCenter{{0.5, 0.5}};
        //     const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     std::cout << "Particles info" << std::endl;
        //     std::cout << " - Tree height = " << TreeHeight << std::endl;
        //     std::cout << " - Number of particles = " << NbParticles << std::endl;

        //     TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

        //     std::vector<std::array<RealType, Dim + 1>> particlePositions(NbParticles);

        //     for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
        //     {
        //         auto pos = randomGenerator.getNewItem();
        //         particlePositions[idxPart][0] = pos[0];
        //         particlePositions[idxPart][1] = pos[1];
        //         particlePositions[idxPart][2] = RealType(0.01);
        //     }

        //     /////////////////////////////////////////////////////////////////////////////////////////
        //     using ParticleDataType = RealType;
        //     constexpr long int NbDataValuesPerParticle = Dim;
        //     using ParticleRhsType = long int;
        //     constexpr long int NbRhsValuesPerParticle = 1;
        //     using MultipoleClass = std::array<long int, 1>;
        //     using LocalClass = std::array<long int, 1>;
        //     using SpaceIndexType = TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>, false>;
        //     using TreeClass = TbfTree<RealType,
        //                               ParticleDataType,
        //                               NbDataValuesPerParticle,
        //                               ParticleRhsType,
        //                               NbRhsValuesPerParticle,
        //                               MultipoleClass,
        //                               LocalClass,
        //                               SpaceIndexType>;

        //     /////////////////////////////////////////////////////////////////////////////////////////
        //     std::array<RealType *, 3> particles;
        //     for (auto &vec : particles)
        //     {
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType *, NbRhsValuesPerParticle> particlesRhs;
        //     for (auto &vec : particlesRhs)
        //     {
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType *, NbRhsValuesPerParticle> particlesRhsScalar;
        //     for (auto &vec : particlesRhsScalar)
        //     {
        //         vec = new RealType[NbParticles]();
        //     }

        //     for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
        //     {
        //         particles[0][idxPart] = particlePositions[idxPart][0];
        //         particles[1][idxPart] = particlePositions[idxPart][1];
        //         particles[2][idxPart] = particlePositions[idxPart][2];
        //     }

        //     FP2PLog::template GenericInner<RealType>(particles, particlesRhs, NbParticles);
        //     FP2PLog::template GenericInnerScalar<RealType>(particles, particlesRhsScalar, NbParticles);

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

        //     for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
        //     {
        //         for (int idxValue = 0; idxValue < NbRhsValuesPerParticle; ++idxValue)
        //         {
        //             partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar[idxValue][idxPart],
        //                                                      particlesRhs[idxValue][idxPart]);
        //         }
        //     }

        //     for (int idxValue = 0; idxValue < NbRhsValuesPerParticle; ++idxValue)
        //     {
        //         std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;

        //         if constexpr (std::is_same<float, RealType>::value)
        //         {
        //             std::cout << "This is " << (partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2) << std::endl;
        //         }
        //         else
        //         {
        //             std::cout << "This is " << (partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3) << std::endl;
        //         }

        //         //        UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
        //         //    }
        //         //    else{
        //         //        UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
        //         //    }
        //     }

        //     TbfTimer timerBuildTree;

        //     TreeClass tree(configuration, particlePositions);

        //     timerBuildTree.stop();
        //     std::cout << "Build the tree in " << timerBuildTree.getElapsed() << "s" << std::endl;

        //     {
        //         using KernelClass = TbfLogKernel<RealType, SpaceIndexType>;
        //         using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;
        //         AlgorithmClass algorithm(configuration);
        //         std::cout << "algorithm -- " << algorithm << std::endl;

        //         TbfTimer timerExecute;

        //         algorithm.execute(tree, TbfAlgorithmUtils::TbfP2P);

        //         timerExecute.stop();
        //         std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;
        //     }

        // }
        // {
        //     const int Dim = 3;
        //     const long int NbRhsValuesPerParticle = 4;

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

        //     std::vector<std::array<RealType, Dim+1>> particlePositions1(NbParticles);
        //     std::vector<std::array<RealType, Dim+1>> particlePositions2(NbParticles);

        //     for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        //         {
        //             auto pos = randomGenerator.getNewItem();
        //             particlePositions1[idxPart][0] = pos[0];
        //             particlePositions1[idxPart][1] = pos[1];
        //             particlePositions1[idxPart][2] = pos[2];
        //             particlePositions1[idxPart][3] = RealType(0.01);
        //         }
        //         {
        //             auto pos = randomGenerator.getNewItem();
        //             particlePositions2[idxPart][0] = pos[0];
        //             particlePositions2[idxPart][1] = pos[1];
        //             particlePositions2[idxPart][2] = pos[2];
        //             particlePositions2[idxPart][3] = RealType(0.02);
        //         }
        //     }

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     std::array<RealType*, 4> particles1;
        //     for(auto& vec : particles1){
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType*, 4> particles2;
        //     for(auto& vec : particles2){
        //         vec = new RealType[NbParticles]();
        //     }

        //     std::array<RealType*, NbRhsValuesPerParticle> particlesRhs1;
        //     for(auto& vec : particlesRhs1){
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar1;
        //     for(auto& vec : particlesRhsScalar1){
        //         vec = new RealType[NbParticles]();
        //     }

        //     std::array<RealType*, NbRhsValuesPerParticle> particlesRhs2;
        //     for(auto& vec : particlesRhs2){
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar2;
        //     for(auto& vec : particlesRhsScalar2){
        //         vec = new RealType[NbParticles]();
        //     }

        //     for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        //         particles1[0][idxPart] = particlePositions1[idxPart][0];
        //         particles1[1][idxPart] = particlePositions1[idxPart][1];
        //         particles1[2][idxPart] = particlePositions1[idxPart][2];
        //         particles1[3][idxPart] = particlePositions1[idxPart][3];
        //         particles2[0][idxPart] = particlePositions2[idxPart][0];
        //         particles2[1][idxPart] = particlePositions2[idxPart][1];
        //         particles2[2][idxPart] = particlePositions2[idxPart][2];
        //         particles2[3][idxPart] = particlePositions2[idxPart][3];
        //     }

        //     FP2PR::template FullMutual<RealType>( particles1, particlesRhs1, NbParticles,  particles2, particlesRhs2, NbParticles);
        //     FP2PR::template FullMutualScalar<RealType>( particles1, particlesRhsScalar1, NbParticles,  particles2, particlesRhsScalar2, NbParticles);

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

        //     for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        //         for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
        //            partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar1[idxValue][idxPart],
        //                                                     particlesRhs1[idxValue][idxPart]);
        //            partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar2[idxValue][idxPart],
        //                                                     particlesRhs2[idxValue][idxPart]);
        //         }
        //     }

        //     for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
        //        std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
        //        if constexpr (std::is_same<float, RealType>::value){
        //            UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
        //        }
        //        else{
        //            UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
        //        }
        //     }
        // }
        // {
        //     const int Dim = 3;
        //     const long int NbRhsValuesPerParticle = 4;

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

        //     std::vector<std::array<RealType, Dim+1>> particlePositions1(NbParticles);
        //     std::vector<std::array<RealType, Dim+1>> particlePositions2(NbParticles);

        //     for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        //         {
        //             auto pos = randomGenerator.getNewItem();
        //             particlePositions1[idxPart][0] = pos[0];
        //             particlePositions1[idxPart][1] = pos[1];
        //             particlePositions1[idxPart][2] = pos[2];
        //             particlePositions1[idxPart][3] = RealType(0.01);
        //         }
        //         {
        //             auto pos = randomGenerator.getNewItem();
        //             particlePositions2[idxPart][0] = pos[0];
        //             particlePositions2[idxPart][1] = pos[1];
        //             particlePositions2[idxPart][2] = pos[2];
        //             particlePositions2[idxPart][3] = RealType(0.02);
        //         }
        //     }

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     std::array<RealType*, 4> particles1;
        //     for(auto& vec : particles1){
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType*, 4> particles2;
        //     for(auto& vec : particles2){
        //         vec = new RealType[NbParticles]();
        //     }

        //     std::array<RealType*, NbRhsValuesPerParticle> particlesRhs2;
        //     for(auto& vec : particlesRhs2){
        //         vec = new RealType[NbParticles]();
        //     }
        //     std::array<RealType*, NbRhsValuesPerParticle> particlesRhsScalar2;
        //     for(auto& vec : particlesRhsScalar2){
        //         vec = new RealType[NbParticles]();
        //     }

        //     for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        //         particles1[0][idxPart] = particlePositions1[idxPart][0];
        //         particles1[1][idxPart] = particlePositions1[idxPart][1];
        //         particles1[2][idxPart] = particlePositions1[idxPart][2];
        //         particles1[3][idxPart] = particlePositions1[idxPart][3];
        //         particles2[0][idxPart] = particlePositions2[idxPart][0];
        //         particles2[1][idxPart] = particlePositions2[idxPart][1];
        //         particles2[2][idxPart] = particlePositions2[idxPart][2];
        //         particles2[3][idxPart] = particlePositions2[idxPart][3];
        //     }

        //     FP2PR::template GenericFullRemote<RealType>( particles1, NbParticles,  particles2, particlesRhs2, NbParticles);
        //     FP2PR::template GenericFullRemoteScalar<RealType>( particles1, NbParticles,  particles2, particlesRhsScalar2, NbParticles);

        //     /////////////////////////////////////////////////////////////////////////////////////////

        //     std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

        //     for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        //         for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
        //            partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar2[idxValue][idxPart],
        //                                                     particlesRhs2[idxValue][idxPart]);
        //         }
        //     }

        //     for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
        //        std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;
        //        if constexpr (std::is_same<float, RealType>::value){
        //            UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
        //        }
        //        else{
        //            UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
        //        }
        //     }
        // }
    }

    // void SetTests() {
    //     Parent::AddTest(&TestP2P::TestBasic<float,7>, "Basic test for P2P float");
    //     Parent::AddTest(&TestP2P::TestBasic<double,3>, "Basic test for P2P double");
    //     Parent::AddTest(&TestP2P::TestBasic<float,1000>, "Basic test for P2P float");
    //     Parent::AddTest(&TestP2P::TestBasic<double,1000>, "Basic test for P2P double");
    // }
};

// You must do this
// TestClass(TestP2P)

int main(int argc, char **argv)
{
    if (TbfParams::ExistParameter(argc, argv, {"-h", "--help"}))
    {
        std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        std::cout << "[HELP] where params are:" << std::endl;
        std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        return 1;
    }

    using RealType = double;
    const int Dim = 2;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1}};
    const long int TreeHeight = TbfParams::GetValue<long int>(argc, argv, {"-th", "--tree-height"}, 8);
    const std::array<RealType, Dim> BoxCenter{{0.5, 0.5}};
    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = TbfParams::GetValue<long int>(argc, argv, {"-nb", "--nb-particles"}, 1000);
    std::cout << "Particles info" << std::endl;
    std::cout << " - Tree height = " << TreeHeight << std::endl;
    std::cout << " - Number of particles = " << NbParticles << std::endl;

    TbfRandom<RealType, Dim> randomGenerator(BoxWidths);

    std::vector<std::array<RealType, Dim + 1>> particlePositions(NbParticles);

    for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
    {
        auto pos = randomGenerator.getNewItem();
        particlePositions[idxPart][0] = pos[0];
        particlePositions[idxPart][1] = pos[1];
        particlePositions[idxPart][2] = RealType(0.01);
    }

    /////////////////////////////////////////////////////////////////////////////////////////
    using ParticleDataType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim;
    using ParticleRhsType = long int;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int, 1>;
    using LocalClass = std::array<long int, 1>;
    using SpaceIndexType = TbfMortonSpaceIndex<Dim, TbfSpacialConfiguration<RealType, Dim>, false>;
    using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass,
                              SpaceIndexType>;

    /////////////////////////////////////////////////////////////////////////////////////////
    std::array<RealType *, 3> particles;
    for (auto &vec : particles)
    {
        vec = new RealType[NbParticles]();
    }
    std::array<RealType *, NbRhsValuesPerParticle> particlesRhs;
    for (auto &vec : particlesRhs)
    {
        vec = new RealType[NbParticles]();
    }
    std::array<RealType *, NbRhsValuesPerParticle> particlesRhsScalar;
    for (auto &vec : particlesRhsScalar)
    {
        vec = new RealType[NbParticles]();
    }

    for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
    {
        particles[0][idxPart] = particlePositions[idxPart][0];
        particles[1][idxPart] = particlePositions[idxPart][1];
        particles[2][idxPart] = particlePositions[idxPart][2];
    }

    FP2PLog::template GenericInner<RealType>(particles, particlesRhs, NbParticles);
    FP2PLog::template GenericInnerScalar<RealType>(particles, particlesRhsScalar, NbParticles);

    /////////////////////////////////////////////////////////////////////////////////////////

    std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

    for (long int idxPart = 0; idxPart < NbParticles; ++idxPart)
    {
        for (int idxValue = 0; idxValue < NbRhsValuesPerParticle; ++idxValue)
        {
            partcilesRhsAccuracy[idxValue].addValues(particlesRhsScalar[idxValue][idxPart],
                                                     particlesRhs[idxValue][idxPart]);
        }
    }

    for (int idxValue = 0; idxValue < NbRhsValuesPerParticle; ++idxValue)
    {
        std::cout << " - Rhs " << idxValue << " = " << partcilesRhsAccuracy[idxValue] << std::endl;

        if constexpr (std::is_same<float, RealType>::value)
        {
            std::cout << "This is " << (partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2) << std::endl;
        }
        else
        {
            std::cout << "This is " << (partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3) << std::endl;
        }

        //        UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-2);
        //    }
        //    else{
        //        UASSERTETRUE(partcilesRhsAccuracy[idxValue].getRelativeL2Norm() < 9e-3);
        //    }
    }

    TbfTimer timerBuildTree;

    TreeClass tree(configuration, particlePositions);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << "s" << std::endl;

    {
        using KernelClass = TbfLogKernel<RealType, SpaceIndexType>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass, SpaceIndexType>;
        AlgorithmClass algorithm(configuration);
        std::cout << "algorithm -- " << algorithm << std::endl;

        TbfTimer timerExecute;

        algorithm.execute(tree, TbfAlgorithmUtils::TbfP2P);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;
    }
    return 0;
}
