#include "UTester.hpp"


#include "tbfglobal.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "algorithms/tbfalgorithmselecter.hpp"
#include "utils/tbftimer.hpp"

#include <iostream>
#include <type_traits>


template <class RealType_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class KernelCheckConstness{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;

    template< class T >
    struct remove_cvref {
        typedef std::remove_cv_t<std::remove_reference_t<T>> type;
    };

    template <class ObjType>
    static void should_be_const_ref(){
        static_assert(std::is_lvalue_reference<ObjType>::value, "const check");
        static_assert(std::is_const<typename std::remove_reference<ObjType>::type>::value, "const check");
    }

    template <class ObjType>
    static void should_be_non_const_ref(){
        static_assert(std::is_lvalue_reference<ObjType>::value, "const check");
        static_assert(!std::is_const<typename std::remove_reference<ObjType>::type>::value, "const check");
    }

    template <class ObjType1, class ObjType2>
    static void should_be_same_base_type(){
        static_assert(std::is_same<typename remove_cvref<ObjType1>::type,
                                   typename remove_cvref<ObjType2>::type>::value, "same check");
    }

public:
    explicit KernelCheckConstness(const SpacialConfiguration& /*inConfiguration*/){}
    explicit KernelCheckConstness(const KernelCheckConstness&){}

    template <class CellSymbolicData, class ParticlesClass, class LeafClass>
    void P2M(CellSymbolicData&& /*inLeafIndex*/,  const long int [],
             ParticlesClass&& /*inParticles*/, const long int /*inNbParticles*/, LeafClass&& /*inOutLeaf*/) const {
        should_be_const_ref<CellSymbolicData>();
        should_be_const_ref<ParticlesClass>();
        should_be_non_const_ref<LeafClass>();
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2M(CellSymbolicData&& /*inCellIndex*/,
             const long int /*inLevel*/, CellClassContainer&& inLowerCell, CellClass&& /*inOutUpperCell*/,
             const long int /*childrenPos*/[], const long int /*inNbChildren*/) const {
        should_be_const_ref<CellSymbolicData>();
        should_be_const_ref<CellClassContainer>();
        should_be_non_const_ref<CellClass>();
        should_be_same_base_type<decltype(inLowerCell[0].get()), CellClass>();
    }

    template <class CellSymbolicData, class CellClassContainer, class CellClass>
    void M2L(CellSymbolicData&& /*inCellIndex*/,
             const long int /*inLevel*/, CellClassContainer&& /*inInteractingCells*/,
             const long int /*neighPos*/[], const long int /*inNbNeighbors*/,
             CellClass&& /*inOutCell*/) const {
        should_be_const_ref<CellSymbolicData>();
        should_be_const_ref<CellClassContainer>();
        should_be_non_const_ref<CellClass>();
    }

    template <class CellSymbolicData, class CellClass, class CellClassContainer>
    void L2L(CellSymbolicData&& /*inCellIndex*/,
             const long int /*inLevel*/, CellClass&& /*inUpperCell*/, CellClassContainer&& inOutLowerCells,
             const long int /*childrednPos*/[], const long int /*inNbChildren*/) const {
        should_be_const_ref<CellSymbolicData>();
        should_be_const_ref<CellClass>();
        should_be_non_const_ref<CellClassContainer>();
        should_be_same_base_type<decltype(inOutLowerCells[0].get()), CellClass>();
    }

    template <class CellSymbolicData, class LeafClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(CellSymbolicData&& /*inCellIndex*/,
             LeafClass&& /*inLeaf*/,  const long int [],
             ParticlesClassValues&& /*inOutParticles*/, ParticlesClassRhs&& /*inOutParticlesRhs*/,
             long int /*inNbParticles*/) const {
        should_be_const_ref<CellSymbolicData>();
        should_be_const_ref<LeafClass>();
        should_be_const_ref<ParticlesClassValues>();
        should_be_non_const_ref<ParticlesClassRhs>();
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(LeafSymbolicData&& /*inNeighborIndex*/,  const long int /*neighborIndexes*/[],
             ParticlesClassValues&& /*inParticlesNeighbors*/, ParticlesClassRhs&& /*inParticlesNeighborsRhs*/,
             const long int /*inNbParticlesNeighbors*/,
             LeafSymbolicData&& /*inTargetIndex*/,   const long int /*targetIndexes*/[], ParticlesClassValues&& /*inOutParticles*/,
             ParticlesClassRhs&& /*inOutParticlesRhs*/, const long int /*inNbOutParticles*/,
             const long /*arrayIndexSrc*/) const {
        should_be_const_ref<LeafSymbolicData>();
        should_be_const_ref<ParticlesClassValues>();
        should_be_non_const_ref<ParticlesClassRhs>();
    }


    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& /*inNeighborIndex*/, const long int /*neighborsIndexes*/[],
             const ParticlesClassValuesSource& /*inParticlesNeighbors*/,
             const long int /*inNbParticlesNeighbors*/,
             const LeafSymbolicDataTarget& /*inParticlesIndex*/, const long int /*targetIndexes*/[],
             const ParticlesClassValuesTarget& /*inOutParticles*/,
             ParticlesClassRhs& /*inOutParticlesRhs*/, const long int /*inNbOutParticles*/,
             const long /*arrayIndexSrc*/) const {
        should_be_const_ref<LeafSymbolicDataSource>();
        should_be_const_ref<ParticlesClassValuesSource>();
        should_be_const_ref<LeafSymbolicDataTarget>();
        should_be_const_ref<ParticlesClassValuesTarget>();
        should_be_non_const_ref<ParticlesClassRhs>();
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(LeafSymbolicData&& /*inSpacialIndex*/,   const long int /*targetIndexes*/[],
                  ParticlesClassValues&& /*inOutParticles*/,
                  ParticlesClassRhs&& /*inOutParticlesRhs*/, const long int /*inNbOutParticles*/) const {
        should_be_const_ref<LeafSymbolicData>();
        should_be_const_ref<ParticlesClassValues>();
        should_be_non_const_ref<ParticlesClassRhs>();
    }
};


class TestKernelConstness : public UTester< TestKernelConstness > {
    using Parent = UTester< TestKernelConstness >;

    void TestBasic() {
        using RealType = double;
        const int Dim = 3;

        /////////////////////////////////////////////////////////////////////////////////////////

        const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
        const long int TreeHeight = 8;
        const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

        const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

        /////////////////////////////////////////////////////////////////////////////////////////

        const long int NbParticles = 1000;

        TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

        std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);

        for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
            particlePositions[idxPart] = randomGenerator.getNewItem();
        }

        /////////////////////////////////////////////////////////////////////////////////////////

        constexpr long int NbDataValuesNeeded = 0;
        constexpr long int NbDataValuesPerParticle = Dim + NbDataValuesNeeded;
        constexpr long int NbRhsValuesPerParticle = 1;
        using MultipoleClass = std::array<RealType,1>;
        using LocalClass = std::array<RealType,1>;
        using KernelClass = KernelCheckConstness<RealType>;
        using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;
        using TreeClass = TbfTree<RealType,
                                  RealType,
                                  NbDataValuesPerParticle,
                                  RealType,
                                  NbRhsValuesPerParticle,
                                  MultipoleClass,
                                  LocalClass>;

        /////////////////////////////////////////////////////////////////////////////////////////

        TbfTimer timerBuildTree;

        TreeClass tree(configuration, particlePositions);

        timerBuildTree.stop();
        std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;

        AlgorithmClass algorithm(configuration);

        TbfTimer timerExecute;

        algorithm.execute(tree);

        timerExecute.stop();
        std::cout << "Execute in " << timerExecute.getElapsed() << std::endl;

        /////////////////////////////////////////////////////////////////////////////////////////

        tree.applyToAllLeaves([]
                              (auto& /*leafHeader*/, const long int* /*particleIndexes*/,
                              const std::array<RealType*, NbDataValuesPerParticle> /*particleDataPtr*/,
                              const std::array<RealType*, NbRhsValuesPerParticle> /*particleRhsPtr*/){
        });
    }

    void SetTests() {
        // Bug with GCC
#ifdef __GNUC__
        volatile bool willBeFalse = false;
        if(willBeFalse)
#endif
        Parent::AddTest(&TestKernelConstness::TestBasic, "Basic test for vector");
    }
};

// You must do this
TestClass(TestKernelConstness)


