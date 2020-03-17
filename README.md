[![pipeline status](https://gitlab.inria.fr/bramas/tbfmm/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/tbfmm/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)

TBFMM is a Fast Multipole Method (FMM) library parallelized with the task-based method.
It is designed to be easy to customize by creating new FMM kernels or new parallelization strategies.
It uses the block-tree (also known as the group-tree), which is well designed for the task-based parallelization, and can be easily extended to heterogeneous architectures (not yet supported but WIP).

The current document is at the same time a classic README but also the main documentation of TBFMM.
We try to answer all the questions that users could have regarding implementation and the use of the library.
Of course, we invite users to post an issues if they find a bug or have any question about TBFMM.

TBFMM uses C++ templates heavily.
It is not required to master templates in order to use TBFMM but it helps to better understand how things work internally.

# Compilation

TBFMM is based on standard C++17, hence it needs a "modern" C++ compiler.
So far, TBFMM has been tested on the following compilers:
- GNU g++ (7 and 8) https://gcc.gnu.org/
- Clang/LLVM (8 and 10) https://llvm.org/

TBFMM should work on Linux and Mac OS.

## How to compile

TBFMM uses CMake as build system https://cmake.org/

The build process consists in the following steps: adding git submodules (optional), creating a build directory, running cmake, configuring cmake, running make and that's all.
The submodules are Inastemp (for vectorization) and SPETABARU (a task-based runtime system).
Both are optional, and to be activated their repository must be cloned/updated before running cmake.

```bash
# To enable SPETABARU and Inastemp
git submodule init && git submodule update
# To enable only SPETABARU (run from the main directory)
git submodule init deps/spetabaru && git submodule update
# To enable only Inastemp (run from the main directory)
git submodule init deps/inastemp && git submodule update

# Creating a build directory (multiple build directory could be used independently)
mkdir build
# Go to the build dir
cd build

# Run cmake with default options
cmake ..
# To enable testing
cmake cmake -DUSE_TESTING=ON -DUSE_SIMU_TESTING=ON ..
# To set FFTW directory from cmake config (or set one of the environement variables FFTW_DIR or FFTWDIR)
cmake -DFFTW_ROOT=path-to-fftw ..
# To build in debug
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
# Or to build in release
cmake -DCMAKE_BUILD_TYPE=RELEASE ..

# Update an existing configuration by calling again cmake -D[an option]=[a value] ..
# or using ccmake ..

# Build everything
make
```

## Running the tests

If the test are enabled in the cmake configuration, one could run the tests with:
```bash
make test
```

To know more about possible failures, use:
```
CTEST_OUTPUT_ON_FAILURE=TRUE make test
```

## Coverage result

Can be found here: https://bramas.gitlabpages.inria.fr/tbfmm/

## OpenMP

CMake will try to check if OpenMP is supported by the system.
If it is the case, all the OpenMP-based code will be enabled, otherwise it will be removed from the compilation process ensuring that the library can compile (but will run in sequential or with SPETABARU).

## Inastemp

Inatemp is a vectorization library that makes it possible to implement a single kernel with an abstract vector data type, which is then compiled for most vectorization instruction sets.
It supports SSE, AVX(2), AVX512, ARM SVE, etc.
To know more, you can have a look at https://gitlab.inria.fr/bramas/inastemp

In TBFMM, only the P2P kernel of the so called rotation and uniform kernels are vectorized with Inastemp (`src/kernels/unifkernel/FP2PR.hpp`).
Therefore, if you are performing simulations with one of these kernels, it is recommended to enable Inastemp, since the performance difference can be impressive.

To avoid having to manage external dependencies, Inastemp is shipped as a git submodule, and thus it will be managed by our cmake files.
But, you must explicitly pull the submobule to enable it.
```bash
# To enable only Inastemp (runned from the main directory)
git submodule init deps/inastemp && git submodule update
```

## SPETABARU

SPETABARU is a C++ task-based runtime system that has speculative execution capability.
Currently, speculation is not used in TBFMM because it requires tasks with a specific data access pattern.
To know more, you can have a look at https://gitlab.inria.fr/bramas/spetabaru

SPETABARU is pure standard C++, and so it does not need any dependencies (apart from the C++ libs/compiler).
It could be a nice alternative to OpenMP when this appears complicated to have an OpenMP lib (as it is sometime the case one some Mac).

To avoid having to manage external dependencies, SPETABARU is shipped as a git submodule, and thus it will be managed by our cmake files.
But, you must explicitly pull the submobule to enable it.
```bash
# To enable only SPETABARU (runned from the main directory)
git submodule init deps/spetabaru && git submodule update
```

# TBFMM design

## Dimension (DIM)

TBFMM can be run simulations of arbitrary dimension.
The dimension must be known at compile time and is used as a template in most TBFMM classes.
Usually, it is declared at the beginning of a main file:

```cpp
const int Dim = 3;
```

## Real/floating type (RealType)

The data type used for the spacial configuration and the position should be specified at compile time to many TBFMM classes.
Usually, it is declared at the beginning of a main file:
```cpp
using RealType = double;
```

## Spacial configuration (TbfSpacialConfiguration)

The description of the spacial environment is saved into the TbfSpacialConfiguration class.
This class stored the desired height of the tree, the simulation box's width and center.
```cpp
// Consider Dim = 3
const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
const long int TreeHeight = 8;
const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
```

## Block/group tree

TBFMM uses the block/group tree, which is an octree where several cells of the same level are allocated and managed together as a group of cells (or as a memory block).
This data structure is well described here http://berenger.eu/blog/wp-content/uploads/2016/07/BerengerBramas-thesis.pdf

The advantage of this structure is that the memory blocks can be moved/copied with a `memcpy`, which is really convenient in HPC when we want to use GPUs or MPI.
The positive side effects are a great data locality and forcing a low level abstraction on the data in the group, which mitigate the overhead.

On the other hand, the constraint (drawback?) is that particles and cells must be POD (or at least to be raw copiable).
For instance, an `std::vector` is not, because it has a pointer internally such that making a raw copy of an object will lead to an undefined/invalid state (it could work in a shared memory system, but clearly not with different memory nodes).

In the context of the FMM, this approach cut the particles and cells in multiple parts depending on what will be read/write by the kernels.
For example, a cell is not a class, but described as three independent parts: a symbolic part, a multipole part and a local part.
Therefore, TBFMM will then allocate multiple parts of the same kind together and maintain the coherency.

## Cell

A cell is composed of:
- a symbolic part
This includes the spacial index and spacial coordinate.
This cannot be configured by the users (without deep modification of the source code).
This is expected to remain constant for a complete FMM algorithm execution.
- a multipole part (`MultipoleClass`)
This is a structure/class that represents the target of a P2M/M2M, or the source of a M2L.
This really depends on the type of kernel (and maybe even the degree of a kernel).
Here a possible examples of what could be a multipole part:
```cpp
// The multipole is simply a long int
using MultipoleClass = long int;

// The multipole is an array of a size known at compile time
using MultipoleClass = std::array<double, 3>;

// The multipole is an array of a size known at compile time
// but based on a degree of the accuracy (P is a constant)
const long int MultipoleArraySize = P*P; 
using MultipoleClass = std::array<double, MultipoleArraySize>;

// The multipole is a POD struct/class
struct MyMultipole{
    long int anInteger;
    double aDouble;
    std::array<double, 128> someValues;
};
using MultipoleClass = MyMultipole;
```
- a local part (`LocalClass`)
This is a structure/class that represent the target of a M2L/L2L, or the source of a L2L/L2P.
This really depends on the type of kernel (and maybe even the degree of a kernel).
Here a possible examples of what could be a local part:
```cpp
// The local is simply a long int
using LocalClass = long int;

// The local is an array of a size known at compile time
using LocalClass = std::array<double, 3>;

// The local is an array of a size known at compile time
// but based on a degree of the accuracy (P is a constant)
const long int LocalArraySize = P*P; 
using LocalClass = std::array<double, LocalArraySize>;

// The local is a POD struct/class
struct MyLocal{
    long int anInteger;
    double aDouble;
    std::array<double, 128> someValues;
};
using LocalClass = MyLocal;
```

Therefore, inside each `main` file of TBFMM, you will see template lines that define what is a multipole part and a local part.
Also, if you create your own kernel, it is clear that you will need to update these lines.

For example, in `tests/testRandomParticles.cpp` you will see:
```cpp
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;
```

## Particles

In TBFMM, a particle cannot be defined as a single struct/class, but as multiple data types that will be allocated and managed by TBFMM.
A particle is defined by three elements:
- a tuple of values that should remained unchanged during a FMM step, that we call "symbolic" data.
  By default, it includes the positions of the particles and it could also be any type of physical values, such as weights.
  These are the sources values of P2M and P2P, but they are also used in the L2P.
  The data type of these values can be chosen by the users.
  We usually call it `DataParticleType` and make it equal to the `RealType`, which is `float` or `double` in most cases.
  The users can only specify how many of symbolic values per particle, but the type will `RealType`.
  The first values are used to store the particles' positions, and the extra values can be used by the users as they want.

  If the type of `DataParticleType` is not set to `RealType` then conversion will happen and re-building the tree will be based on `DataParticleType`.

```cpp
// The type of values
using DataParticleType = RealType;

// If no values are needed (apart from the position):
constexpr long int NbDataValuesPerParticle = Dim;

// Or, if two values are needed in addition to the values for the position:
constexpr long int NbDataValuesPerParticle = Dim + 2;
```

- a set of values that will be updated during the FMM step.
These are the target values of the L2P and P2P, and they cannot be accessed during the P2M.

```cpp
// The type of value that will be changed by the kernel
using RhsParticleType = long int;
// The number of values
constexpr long int NbRhsValuesPerParticle = 1;
```

- an index value par particle, which cannot be controlled by the users, but is passed to the kernel.


## Tree (TbfTree)

In TBFMM, the tree will transform a set of particles into a block-tree and supports all the access methods that the kernels need to perform the FMM algorithm, but also different method to iterate on the particles/cells, or to find a given cell.
The tree class must be defined with all the template parameters related to the cells and particles.

```cpp
    using DataParticleType = RealType;
    constexpr long int NbDataValuesPerParticle = Dim;
    using RhsParticleType = long int;
    constexpr long int NbRhsValuesPerParticle = 1;
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;
    using TreeClass = TbfTree<RealType,
                              DataParticleType,
                              NbDataValuesPerParticle,
                              RhsParticleType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass>;
    // From here, TreeClass can be used, but it needs parameters
```

The tree needs four parameters to be instanciated:
- the spacial configuration (of type `TbfSpacialConfiguration`)
- the size of the blocks (`NbElementsPerBlock`)
- the positions of the particles, which must be an container that supports `std::size` and which has two dimensions. The first one is the index of the particles, and the second one the index of the positions. For example, we classically use `std::vector<std::array<RealType, Dim>>`.
- a Boolean to choose the parent/children blocking strategies (`OneGroupPerParent`).
When this value is set to `true` the blocking strategy will try to set one parent group per child group.
There will be potentially 2 parent groups because the first cells of the child group may have the same parent as the last cell of the previous group.
If set to `false`, the cells are simply grouped by chunk of size `NbElementsPerBlock`.

In order to know how to iterate on the tree's elements or how to find a cell/leaf, we refer to the corresponding section of the current document.

## Kernel

In TBFMM, the kernels must have a specific interface with different methods where the type of the parameters is strict.
However, we recommanded to use template to facilitate the implementation of a kernel.
More precisely, the data types given to the tree (`TbfTree`) could be used directly in the prototype of the kernel, but we advise to use template instead and to create generic kernels.
For instance, if one set the multipole part of the cell as being of type `X`, it is clear that `X` will be passed to the P2M/M2M/M2L when the FMM algorithm will be executed.
But it is better to use a template to accept `X` as parameter, such that future modifications will not impact the methods' prototypes.

A kernel should have the following methods:
- a copy constructor.
This is mandatory only for parallel FMM algorithms, because these algorithms will copy the kernel to have only one kernel object per thread.
This guarantees that each kernel is called by only one thread, thus if the kernel modifies some of its attributes in its methods, this will still be valid (and will not require any exclusion mechanism).
On the other hand, the users should consider optimizing in the creation and copy of their kernels.
For example, if the kernel need a matrix that takes time to be initialized, instead of recomputing it, it could faster to copy it in the copy constructor.
One could even use a shared pointer to have all the kernels using the same matrix (if the matrix is used in read only inside the methods).
In such case, it is not needed to protect the smart pointer when it is duplicated because the kernels are created one after the other by the parallel algorithms.

- a P2M, which takes particles (one leaf) as input and one cell as output.
The prototype is as follows:
```cpp
    template <class CellSymbolicData, class ParticlesClass, class MultipoleClass>
    void P2M(const CellSymbolicData& /*inLeafIndex*/,
             const long int /*particlesIndexes*/[],
             const ParticlesClass& /*inParticles*/,
             const long int /*inNbParticles*/,
             MultipoleClass& /*inOutLeaf*/) const {
```
Here, the `ParticlesClass` can be seen as an array of `ParticlesClass` pointers.
Therefore, `inParticles[idxData][idxParticle]` can be used to access the `idxData` attribute of the particle `idxParticle`.
The first values of `idxData` are the positions (x, y, z, ...) and then the attributes that the user should need.

Said differently,  `inParticles` is of type `DataParticleType* [NbDataValuesPerParticle]`.

- a M2M, which takes one or several child cells as input and one cell as output.
The prototype is as follows:
```cpp
    template <class CellSymbolicData, class MultipoleClassContainer, class MultipoleClass>
    void M2M(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/,
             const MultipoleClassContainer& /*inLowerCell*/,
             MultipoleClass& /*inOutUpperCell*/,
             const long int /*childrenPos*/[],
             const long int /*inNbChildren*/) const {
```
Here, the `MultipoleClassContainer` can be seen as an array of `MultipoleClass` references.
Therefore, `inLowerCell[idxCell].get()` returns a cell of type `MultipoleClass`.
The `childrenPos` array should be use to know the position of the children relatively to the parent `inOutUpperCell`.
This is given by the `childPositionFromParent` method of the spacial ordering system (space filling curve), which could currently be Morton (`TbfMortonSpaceIndex`) or Hilbert (`TbfHilbertSpaceIndex`) ordering.

For example, if the Morton indexing is used, `childrenPos` will contain values between 0 and (2^D)-1.
For a dimension equals to 3, the values will be between 0 and 8-1, and will be equal to the Morton index of a cube of size 2 in each dimension.
To know the relative box coordinate, one could call `getBoxPosFromIndex` of the space system.

`inNbChildren` is the number of children, it will be at most (2^D)-1.
A M2M could be called on a given parent more than once.
In fact, if the children of a cell are split over multiple groups, there will certainly be one call per group.
Therefore, one should carefully update the target cell (without zeroing).

- a M2L, which takes one or several cells as input and one cell as output.
The prototype is as follows:
```cpp
    template <class CellSymbolicData, class MultipoleClassContainer, class LocalClass>
    void M2L(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/,
             const MultipoleClassContainer& /*inInteractingCells*/,
             const long int /*neighPos*/[],
             const long int /*inNbNeighbors*/,
             LocalClass& /*inOutCell*/) const {
```
Here the `inOutCell` parameter is the local part of the target cell.
Its symbolic data are given in `inCellIndex`.

The `inInteractingCells` is a container with all the multipole parts of the cells that interact with the target cell (it could be seen as an array of `MultipoleClass` references).

As in the M2M/L2L, the `neighPos` array indexes that allow to know the relative position of each interacting cell.
To know the relative box coordinate, the users can call `getRelativePosFromInteractionIndex` and pass the index to obtain an array that stores the coordinates, from the spacial index system Morton (`TbfMortonSpaceIndex`) or Hilbert (`TbfHilbertSpaceIndex`) ordering.

- a L2L, which takes one cell as input and one or several cells as output.
The prototype is as follows:
```cpp
    template <class CellSymbolicData, class LocalClass, class LocalClassContainer>
    void L2L(const CellSymbolicData& /*inCellIndex*/,
             const long int /*inLevel*/,
             const LocalClass& /*inUpperCell*/,
             LocalClassContainer& /*inOutLowerCells*/,
             const long int /*childrednPos*/[],
             const long int /*inNbChildren*/) const {
```
It is simply the opposite of the M2M, where multipole (`MultipoleClass`) is replaced by local (`LocalClass`), and the direction of interaction from children to parent is replaced by parent to children.

- a L2P, which takes one cell as input and particles (one leaf) as output.
The prototype is as follows:
```cpp
    template <class CellSymbolicData, class LocalClass, class ParticlesClassValues, class ParticlesClassRhs>
    void L2P(const CellSymbolicData& /*inCellIndex*/,
             const LocalClass& /*inLeaf*/,
             const long int /*particlesIndexes*/[],
             const ParticlesClassValues& /*inOutParticles*/,
             ParticlesClassRhs& /*inOutParticlesRhs*/,
             const long int /*inNbParticles*/) const {
```
This can be seen as the opposite as the P2M, where the multipole part is replaced by a local part, and where the transfer from particles to cell is replaced by cell to particles.

Therefore, the parameter `inOutParticlesRhs` of type `ParticlesClassRhs` is an array of pointers to the rhs of the particles, can be seen as being of type `RhsParticleType* [NbRhsValuesPerParticle]`.

- a P2P, which takes particles (two leaves) as input and output.
The prototype is as follows:
```cpp
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
```
- a P2PTsm, which takes particles (one leaf) as input and particles (one leaf) as output.
The prototype is as follows:
```cpp
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
```
- a P2PInner, which takes particles (one leaf) as input and output.
The prototype is as follows:
```cpp
    template <class LeafSymbolicData, class ParticlesClassValues, class ParticlesClassRhs>
    void P2PInner(const LeafSymbolicData& /*inSpacialIndex*/,
                  const long int /*targetIndexes*/[],
                  const ParticlesClassValues& /*inOutParticles*/,
                  ParticlesClassRhs& /*inOutParticlesRhs*/,
                  const long int /*inNbOutParticles*/) const {
```


An example of empty kernel is given in `tests/exampleEmptyKernel.cpp`.

## Algorithms

In TBFMM, an algorithm takes a kernel and a tree and computes the FMM.

There are several different kernels:

- sequential or parallel
- all particles are sources and targets, or target/source model (Tsm)

In addition, an extra algorithm can be used to apply periodicity above the level 1 (to simulate a repetition of the simulation box).



Here is an example of asking TBFMM to provide the best algorithm class (sequential < OpenMP < SPETABARU)

```cpp
// Let TBFMM select the right algorithm class (for kernel = KernelClass)
using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass<RealType>>;
// Could be specific (but needs to be sure algorithms are supported)
#ifdef TBF_USE_SPETABARU
    using AlgorithmClass = TbfSmSpetabaruAlgorithm<RealType, KernelClass>;
#elif defined(TBF_USE_OPENMP)
    using AlgorithmClass = TbfOpenmpAlgorithm<RealType, KernelClass>;
#else
    using AlgorithmClass = TbfAlgorithm<RealType, KernelClass>;
#endif

// Create an algorithm where the kernel will create using the default constructor
AlgorithmClass algorithm(configuration);
// Equivalent to
AlgorithmClass algorithm(configuration, KernelClass<RealType>());
// Sometime the kernel constructor needs parameters
// In this case, we create the kernel and pass it to the algorithm
KernelClass<RealType> myKernel(some parameters);
AlgorithmClass algorithm(configuration, myKernel);

// Execute the complete FMM
algorithm.execute(tree);

// Execute only a P2P
algorithm.execute(tree, TbfAlgorithmUtils::TbfP2P);
// Could be one of the following (or a combination with binary OR of the following)
// TbfP2P TbfP2M TbfM2M TbfM2L TbfL2L TbfL2P
// TbfNearField = TbfP2P,
// TbfFarField  = (TbfP2M|TbfM2M|TbfM2L|TbfL2L|TbfL2P),
// TbfNearAndFarFields = (TbfNearField|TbfFarField),
// TbfBottomToTopStages = (TbfP2M|TbfM2M),
// TbfTopToBottomStages = (TbfL2L|TbfL2P),
// TbfTransferStages
```



# How-to

## Basic example

## Changing cells

## Counting the number of interactions

## Timing the different operations

## Iterating on the tree

## Have source and target particles

## Use periodicity

## Vectorization of kernels

## Using mesh element as particles

## Rebuilding the tree

## Computing an accuracy

## Select the height of the tree (treeheight)

## Select the block size (blocksize)

## Creating a new kernel

## Find a cell or leaf in the tree (if it exists)

## Spacial ordering/indexing (Morton, Hilbert, space filling curve)

## Execute only part of the FMM



## Macros





# Issues

## Uniform kernel cannot be used

## It is slower with the FMM (compared to direct interactions)

## Make command builds nothing

