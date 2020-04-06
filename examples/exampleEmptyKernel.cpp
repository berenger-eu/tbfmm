#include "tbfglobal.hpp"
#include "spacial/tbfmortonspaceindex.hpp"
#include "spacial/tbfspacialconfiguration.hpp"
#include "utils/tbfrandom.hpp"
#include "core/tbfcellscontainer.hpp"
#include "core/tbfparticlescontainer.hpp"
#include "core/tbfparticlesorter.hpp"
#include "core/tbftree.hpp"
#include "algorithms/sequential/tbfalgorithm.hpp"
#include "utils/tbftimer.hpp"

#include "utils/tbfparams.hpp"

// The following headers have to be used only if
// you want to manage the algorithm type explicitely
#ifdef TBF_USE_SPETABARU
#include "algorithms/smspetabaru/tbfsmspetabarualgorithm.hpp"
#endif
#ifdef TBF_USE_OPENMP
#include "algorithms/openmp/tbfopenmpalgorithm.hpp"
#endif


#include <iostream>


template <class RealType_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class KernelExample{
public:
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, SpaceIndexType::Dim>;
public:
    /// This is called from the main to create the first kernel
    /// this can be adapt to pass what you need and fill the class's attributes
    explicit KernelExample(const SpacialConfiguration& /*inConfiguration*/){}

    /// This is is called from the FMM algorithm to have one kernel per thread
    /// the access will be thread safe (only one kernel is copied at a time)
    /// So if the kernel needs a huge constant matrix, one could use a mutable
    /// shared pointer and safely update it in this copy function
    explicit KernelExample(const KernelExample&){}

    template <class CellSymbolicData, class ParticlesClass, class MultipoleClass>
    void P2M(const CellSymbolicData& /*inLeafIndex*/,
             const long int /*particlesIndexes*/[],
             const ParticlesClass& /*inParticles*/,
             const long int /*inNbParticles*/,
             MultipoleClass& /*inOutLeaf*/) const {
        /// inLeafIndex:
        ///  - .spacialIndex is the spacial index (Morton index for example) of the current leaf
        ///  - .boxCoord is the coordinate in the spacial box of the current leaf
        /// inParticles: is an array of pointers on the particles' data (not on the RHS)
        ///  - For i from 0 to Dim-1, inParticles[i][idxPart] gives the position of particle idxPart on axis i
        ///  - Then, for Dim <= i < NbData inParticles[i][idxPart] gives the data value for particle idxPart
        /// inNbParticles: is the number of particles
        /// inOutLeaf: is the multipole part (here MultipoleClass defined in the main)
    }

    template <class CellSymbolicData, class MultipoleClassContainer, class MultipoleClass>
    void M2M(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/,
             const MultipoleClassContainer& /*inLowerCell*/,
             MultipoleClass& /*inOutUpperCell*/,
             const long int /*childrenPos*/[],
             const long int /*inNbChildren*/) const {
        /// inCellIndex: is the spacial index (Morton index for example) of the parent
        /// inLevel: the level in the tree of the parent
        /// inLowerCell: a container over the multipole parts of the children (~ an array of MultipoleClass
        ///              defined in the main)
        ///              const MultipoleClass& child = inLowerCell[idxChild]
        /// inOutUpperCell: the parent multipole data.
        /// childrenPos: the position of each child. In 3D each index will be between 0 and 7 (as there
        ///              are 2^3 children).
        ///              This could be use to quickly retreive an transfer matrix, for example, one could
        ///              have an array "mat[8]" and always use "mat[idxChild]" for the children at the same
        ///              relative position.
        /// inNbChildren: the number of children.
        ///
        /// Please, note that the M2M might be called multiple time on the same parent. For instance,
        /// if the children are stored on different blocks, there will be one call per block.
    }

    template <class CellSymbolicData, class MultipoleClassContainer, class LocalClass>
    void M2L(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/,
             const MultipoleClassContainer& /*inInteractingCells*/,
             const long int /*neighPos*/[],
             const long int /*inNbNeighbors*/,
             LocalClass& /*inOutCell*/) const {
        /// inCellIndex: is the spacial index (Morton index for example) of the target cell.
        ///              The indexes of the neighbors could be found from inCellIndex and neighPos.
        /// inLevel: the level in the tree of the parent
        /// inInteractingCells: a container over the multipole parts of the children (~ an array of MultipoleClass
        ///                     defined in the main)
        ///                     const LocalClass& neigh = inInteractingCells[idxNeigh]
        /// neighPos: the position of each neigh. In 3D each index will be between 0 and 189.
        ///              This could be use to quickly retreive an transfer matrix, for example, one could
        ///              have an array "mat[189]" and always use "mat[idxNeigh]" for the children at the same
        ///              relative position.
        /// inNbNeighbors: the number of neighbors.
        /// inOutCell: the parent multipole data.
        ///
        /// Please, note that the M2L might be called multiple time on the target cell. For instance,
        /// if the children are stored on different blocks, there will be one call per block.
    }

    template <class CellSymbolicData, class LocalClass, class LocalClassContainer>
    void L2L(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/,
             const LocalClass& /*inUpperCell*/,
             LocalClassContainer& /*inOutLowerCells*/,
             const long int /*childrednPos*/[],
             const long int /*inNbChildren*/) const {
        /// inCellIndex: is the spacial index (Morton index for example) of the parent
        /// inLevel: the level in the tree of the parent
        /// inUpperCell: the parent local data.
        /// inOutLowerCells: a container over the local parts of the children (~ an array of LocalClass
        ///              defined in the main)
        ///              const LocalClass& child = inOutLowerCells[idxChild]
        /// childrenPos: the position of each child. In 3D each index will be between 0 and 7 (as there
        ///              are 2^3 children).
        ///              This could be use to quickly retreive an transfer matrix, for example, one could
        ///              have an array "mat[8]" and always use "mat[idxChild]" for the children at the same
        ///              relative position.
        /// inNbChildren: the number of children.
        ///
        /// Please, note that the L2L might be called multiple time on the same parent. For instance,
        /// if the children are stored on different blocks, there will be one call per block.
    }

    template <class CellSymbolicData, class LocalClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& /*inCellIndex*/,
             const LocalClass& /*inLeaf*/,
             const long int /*particlesIndexes*/[],
             const ParticlesClassValues& /*inOutParticles*/,
             ParticlesClassRhs& /*inOutParticlesRhs*/,
             const long int /*inNbParticles*/) const {
        /// inCellIndex: is the spacial index (Morton index for example) of the current cell
        /// inLeaf: is the local part (here LocalClass defined in the main)
        /// inLeafIndex: is the spacial index (Morton index for example) of the current leaf
        /// inOutParticles: is an array of pointers on the particles' data (not on the RHS)
        ///  - For i from 0 to Dim-1, inOutParticles[i][idxPart] gives the position of particle idxPart on axis i
        ///  - Then, for Dim <= i < NbData inOutParticles[i][idxPart] gives the data value for particle idxPart
        /// inOutParticlesRhs: is an array of pointers on the particles' rhs
        ///  - Then, for 0 <= i < NbRhs inOutParticlesRhs[i][idxPart] gives the data value for particle idxPart
        /// inNbParticles: is the number of particles
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2P(const LeafSymbolicData& /*inNeighborIndex*/,
             const long int /*neighborsIndexes*/[],
             const ParticlesClassValues& /*inParticlesNeighbors*/,
             ParticlesClassRhs& /*inParticlesNeighborsRhs*/,
             const long int /*inNbParticlesNeighbors*/,
             const LeafSymbolicData& /*inTargetIndex*/,
             const long int /*targetIndexes*/[],
             const ParticlesClassValues& /*inOutParticles*/,
             ParticlesClassRhs& /*inOutParticlesRhs*/,
             const long int /*inNbOutParticles*/,
             const long /*arrayIndexSrc*/) const {
        /// To compute the interations between a leaf and a neighbor (should be done in both way).
        /// inNeighborIndex: is the spacial index (Morton index for example) of the neighbor
        /// inParticlesNeighbors: is an array of pointers on the particles' data (not on the RHS)
        ///  - For i from 0 to Dim-1, inOutParticles[i][idxPart] gives the position of particle idxPart on axis i
        ///  - Then, for Dim <= i < NbData inOutParticles[i][idxPart] gives the data value for particle idxPart
        /// inParticlesNeighborsRhs: is an array of pointers on the particles' rhs
        ///  - Then, for 0 <= i < NbRhs inOutParticlesRhs[i][idxPart] gives the data value for particle idxPart
        /// inNeighborPos: is the number of particles in the neighbor container
        /// inOutParticles: is an array of pointers on the particles' data (not on the RHS)
        ///  - For i from 0 to Dim-1, inOutParticles[i][idxPart] gives the position of particle idxPart on axis i
        ///  - Then, for Dim <= i < NbData inOutParticles[i][idxPart] gives the data value for particle idxPart
        /// inOutParticlesRhs: is an array of pointers on the particles' rhs
        ///  - Then, for 0 <= i < NbRhs inOutParticlesRhs[i][idxPart] gives the data value for particle idxPart
        /// inNbOutParticles: is the number of particles in the target container
    }

    template <class LeafSymbolicDataSource, class ParticlesClassValuesSource, class LeafSymbolicDataTarget, class ParticlesClassValuesTarget, class ParticlesClassRhs>
    void P2PTsm(const LeafSymbolicDataSource& /*inNeighborIndex*/,
                const long int /*neighborsIndexes*/[],
                const ParticlesClassValuesSource& /*inParticlesNeighbors*/,
                const long int /*inNbParticlesNeighbors*/,
                const long int /*targetIndexes*/[],
                const LeafSymbolicDataTarget& /*inParticlesIndex*/,
                const ParticlesClassValuesTarget& /*inOutParticles*/,
                ParticlesClassRhs& /*inOutParticlesRhs*/,
                const long int /*inNbOutParticles*/,
                const long /*arrayIndexSrc*/) const {
        // TODO
    }

    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& /*inSpacialIndex*/,
                  const long int /*targetIndexes*/[],
                  const ParticlesClassValues& /*inOutParticles*/,
                  ParticlesClassRhs& /*inOutParticlesRhs*/,
                  const long int /*inNbOutParticles*/) const {
        /// To compute the interations inside a leaf.
        /// inSpacialIndex: is the spacial index (Morton index for example) of the neighbor
        /// inParticles: is an array of pointers on the particles' data (not on the RHS)
        ///  - For i from 0 to Dim-1, inParticles[i][idxPart] gives the position of particle idxPart on axis i
        ///  - Then, for Dim <= i < NbData inParticles[i][idxPart] gives the data value for particle idxPart
        /// inOutParticlesRhs: is an array of pointers on the particles' rhs
        ///  - Then, for 0 <= i < NbRhs inOutParticlesRhs[i][idxPart] gives the data value for particle idxPart
        /// inNbOutParticles: is the number of particles
    }
};


int main(int argc, char** argv){
    if(TbfParams::ExistParameter(argc, argv, {"-h", "--help"})){
        std::cout << "[HELP] Command " << argv[0] << " [params]" << std::endl;
        std::cout << "[HELP] where params are:" << std::endl;
        std::cout << "[HELP]   -h, --help: to get the current text" << std::endl;
        std::cout << "[HELP]   -th, --tree-height: the height of the tree" << std::endl;
        std::cout << "[HELP]   -nb, --nb-particles: specify the number of particles (when no file are given)" << std::endl;
        return 1;
    }

    using RealType = double;
    const int Dim = 3;

    /////////////////////////////////////////////////////////////////////////////////////////

    const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
    const long int TreeHeight = TbfParams::GetValue<long int>(argc, argv, {"-th", "--tree-height"}, 8);
    const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

    const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

    /////////////////////////////////////////////////////////////////////////////////////////

    const long int NbParticles = TbfParams::GetValue<long int>(argc, argv, {"-nb", "--nb-particles"}, 1000);

    std::cout << "Particles info" << std::endl;
    std::cout << " - Tree height = " << TreeHeight << std::endl;
    std::cout << " - Number of particles = " << NbParticles << std::endl;

    TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());

    std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);

    for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
        particlePositions[idxPart] = randomGenerator.getNewItem();
    }

    /////////////////////////////////////////////////////////////////////////////////////////

    using ParticleDataType = RealType; // TODO the type of the constant values of the particles (can be modified outside the kernel)
    constexpr long int NbDataValuesNeeded = 0; // TODO how many real values you need in the data part (in addition to the positions)
    constexpr long int NbDataValuesPerParticle = Dim + NbDataValuesNeeded;
    using ParticleRhsType = double; // TODO what type are the particle RHS
    constexpr long int NbRhsValuesPerParticle = 1; // TODO how many real values you need in the rhs part
    using MultipoleClass = std::array<RealType,1>; // TODO what is a multipole part, could be a class, but must be POD
    using LocalClass = std::array<RealType,1>; // TODO what is a local part, could be a class, but must be POD
    using KernelClass = KernelExample<RealType>;
    using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass>;
#ifdef TBF_USE_SPETABARU
    using AlgorithmClass = TbfSmSpetabaruAlgorithm<RealType, KernelClass>;
#elif defined(TBF_USE_OPENMP)
    using AlgorithmClass = TbfOpenmpAlgorithm<RealType, KernelClass>;
#else
    using AlgorithmClass = TbfAlgorithm<RealType, KernelClass>;
#endif
    // Or use:
    // TbfAlgorithmSelecter::type<RealType, KernelClass> algorithm(configuration);

    /////////////////////////////////////////////////////////////////////////////////////////

    TbfTimer timerBuildTree;

    TreeClass tree(configuration, particlePositions);

    timerBuildTree.stop();
    std::cout << "Build the tree in " << timerBuildTree.getElapsed() << "s" << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    AlgorithmClass algorithm(configuration);

    TbfTimer timerExecute;

    algorithm.execute(tree);

    timerExecute.stop();
    std::cout << "Execute in " << timerExecute.getElapsed() << "s" << std::endl;

    /////////////////////////////////////////////////////////////////////////////////////////

    tree.applyToAllLeaves([]
                          (auto& /*leafHeader*/, const long int* /*particleIndexes*/,
                          const std::array<ParticleDataType*, NbDataValuesPerParticle> /*particleDataPtr*/,
                          const std::array<ParticleRhsType*, NbRhsValuesPerParticle> /*particleRhsPtr*/){
        /// leafHeader.nbParticles: spacial index of the current cell
        /// particleIndexes: indexes of the particles, this correspond to the original order when
        ///                  creating the tree.
        /// particleDataPtr: array of pointers of the particles' data
        /// particleRhsPtr: array of pointers of the particles' rhs
    });

    return 0;
}

