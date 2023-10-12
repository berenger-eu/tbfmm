[![pipeline status](https://gitlab.inria.fr/bramas/tbfmm/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/tbfmm/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02444/status.svg)](https://doi.org/10.21105/joss.02444)

# Introduction 

TBFMM is a Fast Multipole Method (FMM) library (header-only) parallelized with the task-based method. It is designed to be easy to customize by creating new FMM kernels or new parallelization strategies. It uses the block-tree hierarchical data structure (also known as the group-tree), which is well designed for the task-based parallelization, and can be easily extended to heterogeneous architectures (not yet supported but WIP).

Users can implement new FMM kernels, new types of interacting elements or even new parallelization strategies.
As such, it can be used as a simulation toolbox for **scientists in physics or applied mathematics**.
It enables users to perform simulations while delegating the data structure, the algorithm and the parallelization to the library.
Besides, `TBFMM` can also provide an interesting use case for the **HPC research community** regarding parallelization, optimization and scheduling of applications handling irregular data structures.

The current document is at the same time a classic README but also the main documentation of TBFMM (a copy is available in the Wiki https://gitlab.inria.fr/bramas/tbfmm/-/wikis/home). We try to answer all the questions that users could have regarding implementation and the use of the library. Of course, we invite users to post an issue if they find a bug or have any question about TBFMM. Opening an issue can be done on the gitlab repository (https://gitlab.inria.fr/bramas/tbfmm/) or on the github repository (https://github.com/berenger-eu/tbfmm/).

TBFMM uses C++ templates heavily. It is not required to master templates in order to use TBFMM but it helps to better understand how things work internally.

TBFMM is under MIT license, however, some code related to the rotation and the uniform kernels (src/kernels/[rotation|uniform]), are under CeCILL-C license. Using TBFMM without one of these two kernels is under MIT, as zero part of these kernels will be involved.

A paper to cite TBFMM is available here: https://joss.theoj.org/papers/10.21105/joss.02444

# Table of contents

[[_TOC_]]

# Installation instruction

TBFMM is based on standard C++17, hence it needs a "modern" C++ compiler. TBFMM has been tested with the following compilers:
- GNU g++ (7 and 8) https://gcc.gnu.org/
- Clang/LLVM (8 and 10) https://llvm.org/

TBFMM should work on Linux and Mac OS, but has not been tested on Windows.
SPECX needs C++ compiler (for example GNU g++ version 8 or above).
Intel compiler (icpc) can be used to compile the code, however, its OpenMP library is currently not compatible with TBFMM (tested on version `19.0.4.243`).
StarPU needs a C compiler

## Dependency list

All the dependencies are optional:
- OpenMP (for parallelization)
- Inastemp (for P2P vectorization)
- Specx (for parallelization)
- FFTW (for the uniform kernel)
- StarPU (for task-based parallelization, similar to Specx)

## How to compile

TBFMM uses CMake as build system https://cmake.org/

The build process consists in the following steps: cloning the repository and moving to the corresponding folder, adding git submodules (optional), creating a build directory, running cmake, configuring cmake, running make and that's all. The submodules are Inastemp (for vectorization) and SPECX (a task-based runtime system). Both are optional, and to activate them, it is needed to clone their repository before running cmake.

```bash
# Cloning the repository
git clone https://gitlab.inria.fr/bramas/tbfmm.git
# Moving to the newly created "tbfmm" directory
cd tbfmm

# To enable SPECX and Inastemp
git submodule init && git submodule update
# To enable only SPECX (run from the main directory)
git submodule init deps/specx && git submodule update
# To enable only Inastemp (run from the main directory)
git submodule init deps/inastemp && git submodule update

# Creating a build directory (multiple build directory could be used independently)
mkdir build
# Go to the build dir
cd build

# Run cmake with default options
cmake ..
# To enable testing
cmake -DBUILD_TESTS=ON ..
# To set FFTW directory from cmake config (or set one of the environement variables FFTW_DIR or FFTWDIR)
cmake -DFFTW_ROOT=path-to-fftw ..
# To build in debug
cmake -DCMAKE_BUILD_TYPE=DEBUG ..
# Or to build in release
cmake -DCMAKE_BUILD_TYPE=RELEASE ..

# If a package has been found but should be disabled, this can be done
# with -DTBFMM_ENABLE_[PACKAGE]=OFF, where PACKAGE can be:
# SPECX, INASTEMP, FFTW, OPENMP

# To use StarPU, one has to set the env variable STARPU_DIR that contains the install dir of StarPU
# It is also advised to disable Specx, to compile faster
# The cmake variable TBFMM_STARPU_VERSION can be used to set starpu version (1.4 by default)
export STARPU_DIR=/my_computer/StarPU/install/
cmake -DTBFMM_ENABLE_SPECX=OFF
# Or
cmake -DTBFMM_USE_STARPU=/my_computer/StarPU/install/  -DTBFMM_ENABLE_SPECX=OFF
# even if the env variable is set, it is still possible to disable it with -DTBFMM_ENABLE_STARPU=OFF

# Update an existing configuration by calling again cmake -D[an option]=[a value] ..
# or using ccmake ..

# Build everything
make
```

## How to Compile on Windows
First, you need to install a MSYS2 MinGW build envirnment following extensive instructions provided in the links below

```bash
# Follow the article to install MinGW-x64
# https://code.visualstudio.com/docs/languages/cpp

# Particularly follow the article
# https://www.msys2.org/
# and update guide
# https://www.msys2.org/docs/updating/

# Make sure the following packages are installed
pacman -S mingw64/mingw-w64-x86_64-gcc
pacman -S mingw64/mingw-w64-x86_64-make
pacman -S mingw64/mingw-w64-x86_64-cmake
```

# Build in VS Code using gcc
You need to open the ```settings.json``` by ```Ctrl + Shift + P``` and search-select ```Preference: Open User settings (JSON)```,
and then add the lines below:
```bash
  "cmake.cmakePath": "C:\\msys64\\mingw64\\bin\\cmake.exe",
    "cmake.mingwSearchDirs": [
      "C:\\msys64\\mingw64\\bin"
   ],
   "cmake.generator": "MinGW Makefiles"
```
and then open the ```CMake: Edit User-Local CMake Kits``` which opens the ```cmake-tools-kits.json``` file where you need to have:
```bash
  {
    "name": "GCC 12.2.0 x86_64-w64-mingw32",
    "compilers": {
      "C": "C:\\msys64\\mingw64\\bin\\gcc.exe",
      "CXX": "C:\\msys64\\mingw64\\bin\\g++.exe"
    },
    "preferredGenerator": {
      "name": "MinGW Makefiles",
      "platform": "x64"
    },
    "environmentVariables": {
      "PATH": "C:/msys64/mingw64/bin/"
    }
  },
```
Be sure, to set a correct ```PATH``` value.


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

## OpenMP

CMake will try to check if OpenMP is supported by the system. If it is the case, all the OpenMP-based code will be enabled, otherwise it will be removed from the compilation process ensuring that the library can compile (but will run in sequential or with SPECX).

## Inastemp

Inatemp is a vectorization library that makes it possible to implement a single kernel with an abstract vector data type, which is then compiled for most vectorization instruction sets. It supports SSE, AVX(2), AVX512, ARM SVE, etc. To know more, we refer to https://gitlab.inria.fr/bramas/inastemp

In TBFMM, only the P2P kernel of the so called rotation and uniform kernels are vectorized with Inastemp (`src/kernels/unifkernel/FP2PR.hpp`). Therefore, if users are performing simulations with one of these kernels, it is recommended to enable Inastemp, since the performance difference can be impressive.

To avoid having to manage external dependencies, Inastemp is shipped as a git submodule, and thus it will be managed by our cmake files. But, you must explicitly pull the submobule to enable it.
```bash
# To enable only Inastemp (runned from the main directory)
git submodule init deps/inastemp && git submodule update
```

## SPECX

SPECX is a C++ task-based runtime system that has speculative execution capability. Currently, speculation is not used in TBFMM because it requires tasks with a specific data access pattern. To know more, one can have a look at https://gitlab.inria.fr/bramas/specx

SPECX is pure standard C++, and so it does not need any dependencies (apart from the C++ libs/compiler). It could be a nice alternative to OpenMP when this appears complicated to have an OpenMP lib (as it is sometime the case one some Mac).

To avoid having to manage external dependencies, SPECX is shipped as a git submodule, and thus it will be managed by our cmake files. But, the users must explicitly pull the submobule to enable it.
```bash
# To enable only SPECX (runned from the main directory)
git submodule init deps/specx && git submodule update
```

## Code organization

- CMakeLists.txt: the build configuration
- deps: the dependencies (inastemp/specx)
- src: the library
  - algorithms: the algorithms (sequential/openmp/specx)
  - containers: the low-level containers for pure POD approach
  - core: the trees, cells, particles related classes
  - load: basic loader to get particles from FMA files
  - spacial: all classes related to spacial aspects (configuration/morton/hilbert)
  - utils: several helpful classes
- examples: some examples to use TBFMM
- unit-tests: tests and stuff for the continuous integration

# TBFMM design and data structures

## Dimension (DIM)

TBFMM can run simulations of arbitrary dimension. The dimension must be known at compile time and is used as a template in most TBFMM classes. Usually, it is declared at the beginning of a main file:

```cpp
const int Dim = 3;
```

## Real/floating type (RealType)

The data type used for the spacial configuration and the position should be specified at compile time to many TBFMM classes. Usually, it is declared at the beginning of a main file:
```cpp
using RealType = double;
```

## Spacial configuration (TbfSpacialConfiguration)

The description of the spacial environment is saved into the `TbfSpacialConfiguration` class. This class stored the desired height of the tree, the simulation box's width and center.
```cpp
// Consider Dim = 3
const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
const long int TreeHeight = 8;
const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
```

## Block/group tree organization

TBFMM uses the block/group tree, which is an octree where several cells of the same level are allocated and managed together as a group of cells (or as a memory block). This data structure is described here http://berenger.eu/blog/wp-content/uploads/2016/07/BerengerBramas-thesis.pdf

The advantage of this structure is that the memory blocks can be moved/copied with a `memcpy`, which is really convenient in HPC when we want to use GPUs or MPI. The positive side effects are a great data locality and forcing a low level abstraction on the data in the group, which mitigates the overhead.

On the other hand, the constraint (drawback?) is that particles and cells must be POD (or at least to be raw copiable). For instance, an `std::vector` is not, because it has a pointer internally such that making a raw copy of an object will lead to an undefined/invalid state (it could work in a shared memory system, but clearly not with different memory nodes).

In the context of the FMM, this approach cuts the particles and cells in multiple parts depending on what will be read/write by the kernels. For example, a cell is not a class, but described as three independent parts: a symbolic part, a multipole part and a local part. Therefore, TBFMM will then allocate multiple parts of the same kind together and maintain the coherency.

## Cell

A cell is composed of:
- a symbolic part
This includes the spacial index and spacial coordinate.
This cannot be configured by the users (without deep modification of the source code).
This is expected to remain constant for a complete FMM algorithm execution.
- a multipole part (`MultipoleClass`)
This is a structure/class that represents the target of a P2M/M2M, or the source of a M2M/M2L.
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

Therefore, inside each `main` file of TBFMM, you will see template lines that define what is a multipole part and a local part. Also, if you create your own kernel, it is clear that you will need to update these lines.

For example, in `examples/testRandomParticles.cpp` you will see:
```cpp
    using MultipoleClass = std::array<long int,1>;
    using LocalClass = std::array<long int,1>;
```

## Particles

In TBFMM, a particle cannot be defined as a single struct/class, but as several data types that will be allocated and managed by TBFMM.
A particle is defined by three elements:

- a tuple of values that should remained unchanged during a FMM step, that we call "symbolic" data.
  By default, it includes the positions of the particles and it could also be any type of physical values, such as weights.
  These are the sources values of P2M and P2P, but they are also used in the L2P.
  The data type of these values can be chosen by the users.
  We usually call it `DataParticleType` and make it equal to the `RealType`, which is `float` or `double` in most cases.
  The users can only specify how many of symbolic values per particle there are.
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
  It corresponds to the indexes of the particles when they are inserted in the tree.
  So inside the kernels, it is possible to know the original index of each particle, which can be used as a particle id.


## Tree (TbfTree)

In TBFMM, the tree will transform a set of particles into a block-tree and supports all the access methods that the kernels need to perform the FMM algorithm, but also additional method to iterate on the particles/cells, or to find a given cell. The tree class must be defined with all the template parameters related to the cells and particles.

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
- the positions of the particles, which must be a container that supports `std::size` and which has two dimensions. The first one is the index of the particles, and the second one the index of the positions. For example, we classically use `std::vector<std::array<RealType, Dim>>`. The order of the particles in the array is used as an index that is given every time the particles are used. The first particle has index 0, etc.
- the size of the blocks (`NbElementsPerBlock`) [optional]
  If no values is passed to the constructor, then the block size is selected based on the particles positions and number of CPU cores (`TbfBlockSizeFinder`).
  The default value can be override by the `TBFMM_BLOCK_SIZE` environment variable.
- a Boolean to choose the parent/children blocking strategies (`OneGroupPerParent`). [optional]
  When this value is set to `true` the blocking strategy will try to set one parent group per child group.
  There will be potentially 2 parent groups because the first cells of the child group may have the same parent as the last cell of the previous group.
  If set to `false`, the cells are simply grouped by chunk of size `NbElementsPerBlock`.
  It is usually recommended to set it to `false` except when the "cost" of the interactions grows at each level, or when the amount of work is significant and the tree full/dense and ``NbElementsPerBlock` set to a power of 2.

In order to know how to iterate on the tree's elements or how to find a cell/leaf, we refer to the corresponding section of the current document.

## Kernel

In TBFMM, the kernels must have a specific interface with different methods where the type of the parameters is strict. However, we recommended to use template to facilitate the implementation of a kernel. More precisely, the data types given to the tree (`TbfTree`) could be used directly in the prototype of the kernel, but we advise to use template instead and to create generic kernels. For instance, if one set the multipole part of the cell as being of type `X`, it is clear that `X` will be passed to the P2M/M2M/M2L when the FMM algorithm will be executed. But it is better to use a template to accept `X` as parameter, such that future modifications will not impact the methods' prototypes.

The kernel never access the tree directly, instead the kernel is called by the algorithm with parts of the tree (cells or leaves). Consequently, the kernel has nothing, in its prototypes or attributes, that is related to the tree.

A kernel should have the following methods:
- a copy constructor.
This is mandatory only for parallel FMM algorithms, because these algorithms will copy the kernel to have one kernel object per thread.
This guarantees that each kernel is called by only one thread, thus if the kernel modifies some of its attributes in its methods, this will still be valid (and will not require any exclusion mechanism).
On the other hand, the users should consider optimizing in the creation and copy of their kernels.
For example, if the kernel need a matrix that takes time to be initialized, instead of recomputing it, it could be faster to copy it in the copy constructor.
One could even use a shared pointer to have all the kernels using the same matrix (if the matrix is used in read only inside the methods).
In such case, it is not needed to protect the smart pointer with a mutex when it is duplicated because the kernels are created one after the other sequentially by the parallel algorithms.

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

`particlesIndexes` corresponds to the indexes of the particles in this leaf, which is the positions of the particles in the array that was used to create the tree.

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
The P2P compute the interaction between one leaf and its neighbors.
The interactions are supposed to be two ways, so each leaf is used as a source and as a target.
As a consequence, not all neighbors are passed to the function, but only the lower half.
By doing so, we can apply symmetric interactions directly between all the given leaves. 

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
P2PTsm should compute the interaction with one target and multiple source neighbors.
This function is usually called in a Tsm FMM (when sources != targets).
The source leaves should not be modified, also they do not have rhs.

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

P2PInner is used to compute the interaction of a leaf on itself.

An example of empty kernel is given in `examples/exampleEmptyKernel.cpp`.

## Algorithms

In TBFMM, an algorithm takes a kernel and a tree and computes the FMM.

There are several different kernels:

- sequential or parallel
- all particles are sources and targets, or target/source model (Tsm)

In addition, an extra algorithm can be used to apply periodicity above the level 1 (to simulate a repetition of the simulation box).

Here is an example of asking TBFMM to provide the best algorithm class (sequential < OpenMP < SPECX < StarPU)

```cpp
// Let TBFMM select the right algorithm class (for kernel = KernelClass)
using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;
// Could be specific (but needs to be sure the algorithm is supported)
#ifdef TBF_USE_STARPU
    using AlgorithmClass = TbfSmStarpuAlgorithm<RealType, KernelClass>;
#if defined(TBF_USE_SPECX)
    using AlgorithmClass = TbfSmSpecxAlgorithm<RealType, KernelClass>;
#elif defined(TBF_USE_OPENMP)
    using AlgorithmClass = TbfOpenmpAlgorithm<RealType, KernelClass>;
#else
    using AlgorithmClass = TbfAlgorithm<RealType, KernelClass>;
#endif

// Create an algorithm where the kernel will create using the default constructor
AlgorithmClass algorithm(configuration);
// Equivalent to
AlgorithmClass algorithm(configuration, KernelClass());
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

Both SPECX and OpenMP based algorithm can have the number of threads to use given by the environment variable `OMP_NUM_THREADS` and the binding with `OMP_PROC_BIND`.

# How-to and examples

## Basic example

Most of our examples a built similarly:

- declaration of the spacial properties
- initialization of the particles
- definition of the template types
- creation of the FMM components (tree, kernel, algorithm)
- iteration on the results and/or rebuilding the tree to run the algorithm again

```cpp
// Fix the real data type and dimension
using RealType = double;
const int Dim = 3;

// Set the simulation box property
const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
const long int TreeHeight = 8;
const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};
// Create the spacial configuration object
const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);

// Generate random particles
const long int NbParticles = 1000;
TbfRandom<RealType, Dim> randomGenerator(configuration.getBoxWidths());
std::vector<std::array<RealType, Dim>> particlePositions(NbParticles);
for(long int idxPart = 0 ; idxPart < NbParticles ; ++idxPart){
    particlePositions[idxPart] = randomGenerator.getNewItem();
}

// Fix the templates
using ParticleDataType = RealType;
constexpr long int NbDataValuesPerParticle = Dim;
using ParticleRhsType = long int;
constexpr long int NbRhsValuesPerParticle = 1;
using MultipoleClass = std::array<long int,1>;
using LocalClass = std::array<long int,1>;
using TreeClass = TbfTree<RealType,
                            ParticleDataType,
                            NbDataValuesPerParticle,
                            ParticleRhsType,
                            NbRhsValuesPerParticle,
                            MultipoleClass,
                            LocalClass>;
using KernelClass = TbfTestKernel<RealType>;

// Create the tree
TreeClass tree(configuration, particlePositions);
// Or by specifiying the tree structure:
// const long int NbElementsPerBlock = 50;
// const bool OneGroupPerParent = false;
// TreeClass tree(configuration, particlePositions, NbElementsPerBlock, OneGroupPerParent);

// Create the algorithm
using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, KernelClass>;
AlgorithmClass algorithm(configuration);
// Execute the algorithm
algorithm.execute(tree);

// Execute for 10 iterations
for(long int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
    // TODO update positions
    // Then rebuild the tree
    tree.rebuild();
    // Run the algorithm again
	algorithm.execute(tree);
}
```

The file `examples/testRotationKernel.cpp` includes some comments related to these different stages/operations.

## Changing the cells

If someone follows the design with a full template specification of the classes, then changing the type of the cells simply requires to change the corresponding templates. Potentially, if the methods of the kernel are not template-based, it might be needed to update them to match the new cell type.

## Counting the number of interactions (TbfInteractionCounter)

It is often very useful to count the number of interactions, and their types. To do so, we propose a wrapper kernel that has to be plugged between the real kernel and the FMM algorithm.

This kernel is fully templatized, and this can work with any kernel. It takes as template the type of the original kernel and support the `<<` operator to print the results. However, since each thread has its own copy of the kernel, we need to use a trick to merge all the counter-kernels together.

Consider an original source code:

```cpp
using KernelClass = TbfTestKernel<RealType>;
```

This must be transformed into (ant that's all):

```cpp
using KernelClass = TbfInteractionCounter<TbfTestKernel<RealType>>;
```

However, to get the results we have to use the following piece of code

```cpp
auto counters = typename KernelClass::ReduceType();

algorithm.applyToAllKernels([&](const auto& inKernel){
    counters = KernelClass::ReduceType::Reduce(counters, inKernel.getReduceData());
});
// Print the counters (merge from all the kernels)
std::cout << counters << std::endl;
```

## Timer (TbfTimer)

TBFMM provides an easy to use timer based on standard C++. It is the `TbfTimer`, which supports start/stop/etc...

In the following example, we print the time it takes to build the tree:

```cpp
TbfTimer timerBuildTree;

TreeClass tree(configuration, NbElementsPerBlock, particlePositions, OneGroupPerParent);

timerBuildTree.stop();
std::cout << "Build the tree in " << timerBuildTree.getElapsed() << std::endl;
```



## Timing the different operations (TbfInteractionTimer)

It is often very useful to time in details the different operators. To do so, we propose a wrapper kernel that has to be plugged between the real kernel and the FMM algorithm.

This kernel is fully templatized, and this can work with any kernel. It takes as template the type of the original kernel and support the `<<` operator to print the results. However, since each thread has its own copy of the kernel, we need to use a trick to merge all the timer-kernels together.

Consider an original source code:

```cpp
using KernelClass = TbfTestKernel<RealType>;
```

This must be transformed into (ant that's all):

```cpp
using KernelClass = TbfInteractionTimer<TbfTestKernel<RealType>>;
```

However, to get the results we have to use the following piece of code

```cpp
auto timers = typename KernelClass::ReduceType();

algorithm.applyToAllKernels([&](const auto& inKernel){
     timers = KernelClass::ReduceType::Reduce(timers, inKernel.getReduceData());
});
// Print the execution time of the different operators
std::cout << timers << std::endl;
```

## Printing all the interactions (TbfInteractionPrinter)

It is often very useful to print all the interactions between cells/leaves, for instance to debug. To do so, we propose a wrapper kernel that has to be plugged between the real kernel and the FMM algorithm.

This kernel is fully templatized, and this can work with any kernel. 

Consider an original source code:

```cpp
using KernelClass = TbfTestKernel<RealType>;
```

This must be transformed into (ant that's all):

```cpp
using KernelClass = TbfInteractionPrinter<TbfTestKernel<RealType>>;
```

## Iterating on the tree

Once the tree is built, it might be useful to iterate on the cells, and it is usually needed to iterate on the leaves/particles.

Our tree does not support range for loop because we prefer to fully abstract the iteration system, avoid creating iterator, and constraint the prototype of the lambda function.

Consequently, to iterate over all the cells could be done with the following code:

```cpp
 tree.applyToAllCells([](
                       const long int inLevel,
                       auto&& cellHeader,
                       const std::optional<std::reference_wrapper<MultipoleClass>> cellMultipole,
                       const std::optional<std::reference_wrapper<LocalClass>> cellLocalr){
    // Some code
});
```

Iterating over the leaves/particles could be done with:

```cpp
tree.applyToAllLeaves([](auto& leafHeader,
                         const long int* /*particleIndexes*/,
                         const std::array<ParticleDataType*, NbDataValuesPerParticle> particleDataPtr,
                         const std::array<ParticleRhsType*, NbRhsValuesPerParticle> particleRhsPtr){
    // Some code
    // The number of particles is in leafHeader.nbParticles
});
```

The `particleIndexes` parameter gives the index of each particle (which correspond to the position of the particles when they were inserted in the tree).



## Have source and target particles (Tsm)

Most of the examples we give consider that particles are sources and targets at the same time.

But there are many applications where that it is not the case. It is very easy to switch to this mode by putting `Tsm` postfix behind the algorithm and the tree classes:

```cpp
// TbfAlgorithmSelecter => TbfAlgorithmSelecterTsm
// Same if specific algorithm class is used
using AlgorithmClass = TbfAlgorithmSelecterTsm::type<RealType, TbfTestKernel<RealType>>;
// TbfTree => TbfTreeTsm
using TreeClass = TbfTreeTsm<RealType,
                                 ParticleDataType,
                                 NbDataValuesPerParticle,
                                 ParticleRhsType,
                                 NbRhsValuesPerParticle,
                                 MultipoleClass,
                                 LocalClass>;
```

The main differences come from the creation of the tree and how to iterate over the tree's elements.

```cpp
// Usually only one array of positions is needed, but with tsm
// we need to pass one array for the sources, and one array for the targets
TreeClass tree(configuration, particlePositionsSource, particlePositionsTarget);
// Optional extra arguments: OneGroupPerParent, NbElementsPerBlock
```

When we iterate, we have to specify if we iterate over the targets or the sources:

```cpp
tree.applyToAllCellsSource([](const long int inLevel,
                              auto&& cellHeader,
                              const std::optional<std::reference_wrapper<MultipoleClass>> cellMultipole,
                                 auto&& /*local part unused*/){
    // Some code
});

tree.applyToAllCellsTarget([](const long int inLevel,
                              auto&& cellHeader,
                              auto&& /*multipole part unused*/,
                              const std::optional<std::reference_wrapper<LocalClass>> cellLocal){
    // Some code
});
```

In the previous example, the source cells do not have a local part, therefore we leave this parameter unused. Similarly, the target cells do not have a multipole part.

For the leaves/particles:

```cpp
tree.applyToAllLeavesSource([](auto&& leafHeader,
                               const long int* particleIndexes,
                               const std::array<ParticleDataType*, NbDataValuesPerParticle> particleDataPtr,
                               auto&& /*particles rhs unused*/){
    // Some code
    // The number of particles is in leafHeader.nbParticles
});
tree.applyToAllLeavesTarget([](auto&& leafHeader,
                         const long int* /*particleIndexes*/,
                         const std::array<ParticleDataType*, NbDataValuesPerParticle> particleDataPtr,
                         const std::array<ParticleRhsType*, NbRhsValuesPerParticle> particleRhsPtr){
    // Some code
    // The number of particles is in leafHeader.nbParticles
});
```

Source particles do not have rhs, this is why we commented this parameter.



## Use periodicity

In TBFMM we support a pure FMM/algorithmic-based periodic model (described in http://berenger.eu/blog/wp-content/uploads/2016/07/BerengerBramas-thesis.pdf).

The idea is to duplicate the simulation box a huge number of times in all directions. The numerical stability of the method relies on the kernel because we simply use the classical FMM operators. With this aim, we take the cell at the level 1 of the FMM tree and then perform computation as if the tree was going to a much higher level.

In order to use this model, it is necessary to select the level up to which the FMM is applied. In addition, the classical classes that are used for a non-periodic FMM must be slightly modified to apply periodic operations. The P2P and M2L should takes cells considering that the box is repeated.

By default, many classes use `TbfDefaultSpaceIndexType` as spacial ordering (which is Morton indexing/space filling curve). When we use the periodic model, we have to specify to each class that the space ordering to use is different, and for example that it is `TbfDefaultSpaceIndexTypePeriodic` (which is also Morton indexing but with periodicity turned on). Therefore, the template declaration are a little different:

```cpp
using SpacialSystemPeriodic = TbfDefaultSpaceIndexTypePeriodic<RealType>;
using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass,
                              SpacialSystemPeriodic>; // optional last template
using AlgorithmClass = TbfTestKernel<RealType,
     							     SpacialSystemPeriodic>; // optional last template
```

Then, we need to run the classical FMM algorithm but up to level 1 (instead of level 2, because in a non periodic FMM there are no M2L to do at level 1), and to run a top periodic FMM algorithm

```cpp
using AlgorithmClass = TbfAlgorithmSelecter::type<RealType, AlgorithmClass, SpacialSystemPeriodic>;
// New template
using TopPeriodicAlgorithmClass = TbfAlgorithmPeriodicTopTree<RealType, AlgorithmClass, MultipoleClass, LocalClass, SpacialSystemPeriodic>;

// Specify to the FMM the upper level
const long int LastWorkingLevel = TbfDefaultLastLevelPeriodic;

AlgorithmClass algorithm(configuration, LastWorkingLevel);
TopPeriodicAlgorithmClass topAlgorithm(configuration, idxExtraLevel);


// Bottom to top classical FMM algorithm
algorithm.execute(tree, TbfAlgorithmUtils::TbfBottomToTopStages);
// Periodic at the top (could be done in parallel with TbfTransferStages)
topAlgorithm.execute(tree);
// Transfer (could be done in parallel with topAlgorithm.execute)
// (M2L and P2P will be periodic)
algorithm.execute(tree, TbfAlgorithmUtils::TbfTransferStages);
// Top to bottom classical FMM algorithm
algorithm.execute(tree, TbfAlgorithmUtils::TbfTopToBottomStages);
```



## Vectorization of kernels

In a previous section, we have described what is Inastemp and how we use it to vectorize the P2P. We strongly advised our users to do as well. Using Inastemp is really easy and is automatically managed by our CMake configuration.

Moreover, one can leave a comment or issue on the Inastemp's website if any feature is missing for a given project.



## Using mesh element as particles

It is quite common that the FMM is mapped on an existing application with its own data structures, etc. For instance, one could have a mesh defined with spacial elements, and where each element has an index. In this context, it could appear difficult to transform these elements into "particles" or to create some kind of P2P (because, in such cases, it is common that the P2P is nothing more than working on the mesh with a classical approach).

To solve this issue, we propose the following design. A particle will have no specific data or rhs values. A particle will simply be a position and an index (which will correspond to an index in the mesh structure). Then, in the P2P, one will simply use the index to compute the interaction between the mesh elements.

In this section, we provide a simplified code to implement this mechanism.

First, transform mesh elements' positions into array of positions:

```cpp
// Here "mesh" is an example of application specific data structure
std::vector<std::array<double, Dim>> positions(mesh.nb_elements);
for(element : mesh.elements){
    positions[element.index] = element.position;
    // It could be needed to use a second loop over "Dim" to copy each value
}
```

Then, we create template with 0 data and rhs values for the particles:

```cpp
using ParticleDataType = RealType;
constexpr long int NbDataValuesPerParticle = Dim; // Nothing more
using ParticleRhsType = void_data; // TBFMM empty struct
constexpr long int NbRhsValuesPerParticle = 0;
using MultipoleClass = TODO; // Specific to the kernel that will approximate the P2P
using LocalClass = TODO; // Specific to the kernel that will approximate the P2P
using TreeClass = TbfTree<RealType,
                            ParticleDataType,
                            NbDataValuesPerParticle,
                            ParticleRhsType,
                            NbRhsValuesPerParticle,
                            MultipoleClass,
                            LocalClass>;
```

Inside the kernel, any method that implies the particles will not use the positions or rhs but only the indexes:

```cpp
template <class RealType_T, class SpaceIndexType_T = TbfDefaultSpaceIndexType<RealType_T>>
class KernelMeshExample{    
public:
    constexpr int Dim = SpaceIndexType::Dim;
    using RealType = RealType_T;
    using SpaceIndexType = SpaceIndexType_T;
    using SpacialConfiguration = TbfSpacialConfiguration<RealType, Dim>;
    
private:
    std::array<RealType,Dim> getLeafCenter(const std::array<long int, 3>& coordinate) const {
        std::array<RealType, Dim> center;
        for(int idxDim = 0 ; idxDim < Dim ; ++idxDim){
            center[idxDim] = (configuration.getBoxCorner()[idxDim] + RealType(coordinate[idxDim]) + RealType(.5)) * inConfiguration.getLeafWidths()[idxDim];
        }
        return center;
    }

    std::array<RealType, Dim> getLeafCenter(const typename SpaceIndexType::IndexType& inIndex) const{
        return getLeafCenter(spaceIndexSystem.getBoxPosFromIndex(inIndex));
    }
    
    const SpacialConfiguration configuration;
    MeshType* myMesh; // Ptr to the real mesh data structure
    
public:
    explicit KernelExample(const SpacialConfiguration& inConfiguration,
                           MeshType* inMyMesh)
    	: configuration(inConfiguration), myMesh(inMyMesh){}

    explicit KernelExample(const KernelExample& inOther) : myMesh(inOther.myMesh){}
    
    // Just the example for a P2M
    template <class CellSymbolicData, class ParticlesClass, class MultipoleClass>
    void P2M(const CellSymbolicData& inLeafIndex,
             const long int particlesIndexes[],
             const ParticlesClass& /*inParticles*/,
             const long int inNbParticles,
             MultipoleClass& inOutLeaf) const {
        // The spacial position of the cell
        const std::array<RealType,SpaceIndexType::Dim> cellPosition = getLeafCenter(LeafIndex.boxCoord);
        
        for(long int idxElement = 0 ; idxElement < inNbParticles ; ++idxElement){
            computation from myMesh[particlesIndexes[idxElement]]
                to inOutLeaf using cellPosition;
        }
    }
    
    // All the other methods
};
```

Then, the tree and kernel are created as usual:

```cpp
using KernelClass = KernelMeshExample;

KernelClass kernelWithMeshPtr(&mesh);
AlgorithmClass algorithm(configuration, kernelWithMeshPtr);

algorithm.execute(tree);
```



## Rebuilding the tree

In most applications, the particles move at each iteration (usually their forces are updated during the FMM, and the move happen just after). Therefore, some particles are no longer in the appropriate leaves. In TBFMM, it then requires to rebuild the tree.

Here is an example:

```cpp
for(long int idxLoop = 0 ; idxLoop < 10 ; ++idxLoop){
    tree.applyToAllLeaves([](auto& leafHeader,
                         const long int* particleIndexes,
                         std::array<ParticleDataType*,
                         				NbDataValuesPerParticle> particleDataPtr,
                         const std::array<ParticleRhsType*, 
                         				NbRhsValuesPerParticle> particleRhsPtr){
        // Update the position of the particles
        for(lont int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
            // Use particle id if needed
            const long int originalParticleIndex = particleIndexes[idxPart];
            for(long int idxDim = 0 ; idxDim < Dim ; ++idxDim){
                particleDataPtr[idxPart][idxDim] = ... TODO ... ;
            }
        }
    });
    // Then rebuild the tree
    tree.rebuild();
    // Run the algorithm again
	algorithm.execute(tree);
}
```

## Cell/leaf/particles header (cellHeader/leafHeader)

In the kernel invocation or in the iteration over the tree, TBFMM provdes `cellHeader` and `leafHeader`.

The content of these structures is the following:

```cpp
struct CellHeader {
    IndexType spaceIndex;
    std::array<long int, Dim> boxCoord;
};

struct LeafHeader {
    IndexType spaceIndex;
    long int nbParticles;
    long int offSet;
    std::array<long int, Dim> boxCoord;
};
```



## Computing an accuracy (TbfAccuracyChecker)

TBFMM provides `TbfAccuracyChecker`, a class to evaluate the accuracy between good and approximate values.

It can be used as follows:

```cpp
TbfAccuracyChecker<RealType> accurater;

// Add on or lots of values
accurater.addValues(a_good_value, a_value_to_test);

// Print the result
std::cout << accurater << std::endl;

// Test the accuracy
if(accurater.getRelativeL2Norm() > 1E-6){
    // Oups...
}
```

To check all the values of the particles, one can do as follows:

```cpp
std::array<TbfAccuracyChecker<RealType>, NbDataValuesPerParticle> partcilesAccuracy;
std::array<TbfAccuracyChecker<RealType>, NbRhsValuesPerParticle> partcilesRhsAccuracy;

tree.applyToAllLeaves([&particles,&partcilesAccuracy,&particlesRhs,&partcilesRhsAccuracy]
                      (auto&& leafHeader, const long int* particleIndexes,
                       const std::array<RealType*, NbDataValuesPerParticle> particleDataPtr,
                       const std::array<RealType*, NbRhsValuesPerParticle> particleRhsPtr){
    for(int idxPart = 0 ; idxPart < leafHeader.nbParticles ; ++idxPart){
       for(int idxValue = 0 ; idxValue < NbDataValuesPerParticle ; ++idxValue){
            partcilesAccuracy[idxValue].addValues(particles[idxValue][particleIndexes[idxPart]],                                                  particleDataPtr[idxValue][idxPart]);
        }                         
        for(int idxValue = 0 ; idxValue < NbRhsValuesPerParticle ; ++idxValue){
               partcilesRhsAccuracy[idxValue].addValues(particlesRhs[idxValue][particleIndexes[idxPart]],
                         particleRhsPtr[idxValue][idxPart]);
         }
     }
 });

std::cout << "Relative differences:" << std::endl;
for(int idxValue = 0 ; idxValue < NbDataValuesPerParticle ; ++idxValue){
    std::cout << " - Data " << idxValue << " = " << partcilesAccuracy[idxValue] << std::endl;
    UASSERTETRUE(partcilesAccuracy[idxValue].getRelativeL2Norm() < 1e-16);
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
```



## Select the height of the tree (treeheight)

There is no magical formula to know the perfect height of the tree (the one for which the execution time will be minimal). In fact, the tree height should be selected depending of the costs of the FMM operators (so depending on what the kernel does, and maybe on some kind of accuracy parameters such as degree of approximation polynomial, etc), but also depending on the data distribution.

What could be done is to perform several test and try attempts to find the best tree height: adding one level will add work on the far field (the work above the leaves) and decrease the work at leaf level, while removing one level will do the opposite. Therefore, one should find a good balance between both.

## Select the block size (blocksize)

We are currently trying to create a method to find a good blocksize, which is a balance between good granularity of the parallel tasks and the degree of parallelism.

## Creating a new kernel

To create a new kernel, we refer to the file `examples/exampleEmptyKernel.cpp` that provides an empty kernel and many comments.

## Find a cell or leaf in the tree (if it exists)

After the particles are inserted and the tree built, it is possible to query to find a cell or a leaf. To do so, one has to ask for a specific spacial index (and the level to find the cell).

```cpp
// Find a cell
//    findResult will be of type:
//    std::optional<std::pair<std::reference_wrapper<CellGroupClass>,long int>>
auto findResult = tree.findGroupWithCell(levelOfTheCellToFind, indexToFind);
if(findResult){
    auto multipoleData = (*findResult).first.get().getCellMultipole((*findResult).second);
    auto LocalData = (*findResult).first.get().getCellLocal((*findResult).second);
}

// Find a leaf
//     findResult will be of type:
//     std::optional<std::pair<std::reference_wrapper<LeafGroupClass>,long int>>
auto groupForLeaf = tree.findGroupWithLeaf(indexToFind);
if(findResult){
    auto nbParticles = (*findResult).first.get().getNbParticlesInLeaf((*findResult).second);
    // Other methods are getParticleIndexes, getParticleData, getParticleRhs
}
```



## Spacial ordering/indexing (Morton, Hilbert, space filling curve)

It is possible to create a new spacial ordering, or to use one of the current system: Morton (`TbfMortonSpaceIndex`) or Hilber indexing (`TbfHilbertSpaceIndex`). By default, all classes use `TbfDefaultSpaceIndexType`, which is `TbfMortonSpaceIndex` for dimension equals to 3 and without periodicity. `TbfDefaultSpaceIndexTypePeriodic` is the same but with periodicity enabled.

If someone wants to use something else, it is needed to pass it to the tree and algorithm classes with template, and potentially to the kernel.

```cpp
using SpacialSystemToUse = select a spacial system;
using TreeClass = TbfTree<RealType,
                              ParticleDataType,
                              NbDataValuesPerParticle,
                              ParticleRhsType,
                              NbRhsValuesPerParticle,
                              MultipoleClass,
                              LocalClass,
                              SpacialSystemToUse>; // optional last template
using AlgorithmClass = TbfTestKernel<RealType,
     							     SpacialSystemToUse>; // optional last template
```



## Execute only part of the FMM

The execute method of the algorithms accepts an optional parameter to specify which part of the FMM algorithm should be done. By default, the complete FMM will be done.

```cpp
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



## Macros

The cmake system will define several macro for the potential dependencies:

```cpp
TBF_USE_SPECX
TBF_USE_OPENMP
TBF_USE_INASTEMP
TBF_USE_FFTW
```

Any cpp file in the tests or unit-tests directories will not be compiled if it contains ` @TBF_USE_X` and that `X` is not supported. For example, the tests related to the uniform kernel have the following code:

```cpp
// -- DOT NOT REMOVE AS LONG AS LIBS ARE USED --
// @TBF_USE_FFTW
// -- END --
```

Therefore, cmake will not use these file if FFTW is not supported.

## Generating several versions of a template-based kernel

TBFMM is really template oriented, and it is expected that the kernels include as template their "degree" or "order". For instance, one could have a kernel such as:

```cpp
template <class RealType, int Order>
class MyKernel{
    // Do everything based on Order,
    // allocate arrays, create loops, etc...
}
```

In the `main`, the type of Multipole and Local parts are based on this order:

```cpp
constexpr int Order = 5;
constexpr int SizeOfMultipole = Order*Order; // This is an example
using MultipoleClass = std::array<long int,SizeOfMultipole>;
constexpr int SizeOfLocal = Order*Order*Order; // This is an example
using LocalClass = std::array<long int,SizeOfLocal>;
using KernelClass = MyKernel<RealType, Order>;
```

However, one could want to decide at runtime the value used for Order, for instance, by looking at application's arguments. One possibility would be to do this manually, but we offer a class to help the generation of multiple versions.

```cpp
#include "utils/tbftemplate.hpp"

int main(int argc, char** argv){
    // Some code
    
    const int dynamicOrder = atoi(argv[1]); // This is an example
    
    // Will generate all the code for order from 0 to 4 by step 1
    TbfTemplate::If<0, 5, 1>(dynamicOrder, [&](const auto index){
    	constexpr int Order = index.index;
        assert(Order == dynamicOrder); // Will be true
        
        // From that line we can use Order as a const value,
        // but it will be specified at compile time
        
        constexpr int SizeOfMultipole = Order*Order; // This is an example
        using MultipoleClass = std::array<long int,SizeOfMultipole>;
        constexpr int SizeOfLocal = Order*Order*Order; // This is an example
        using LocalClass = std::array<long int,SizeOfLocal>;
        using KernelClass = MyKernel<RealType, Order>;
        // Do the rest as usual
        ....
    });
    // Rest of the main
    ....
    return 0;
}



```





## Existing kernels

Currently, we have taken two kernels the rotation kernel and the uniform kernel. They have been taken from ScalFMM, which is an FMM library where the kernels have a very similar interface to what we use.

We do not plan to improve or work on these kernels, they are just here to illustrate how TBFMM works, how a kernel can be implemented and to perform some experiments.

We refer to the following documents to know more:

- rotation
  - Dachsel, H. (2006). Fast and accurate determination of the Wigner rotation matrices in the fast multipole method. *The Journal of chemical physics*, *124*(14), 144115.
  - White, C. A., & Head‐Gordon, M. (1994). Derivation and efficient implementation of the fast multipole method. *The Journal of Chemical Physics*, *101*(8), 6593-6605.
  - White, C. A., & Head‐Gordon, M.  (1996). Rotating around the quartic angular momentum barrier in fast  multipole method calculations. *The Journal of Chemical Physics*, *105*(12), 5061-5067.
  - Haigh, A. (2011). Implementation of rotation-based operators for Fast Multipole Method in X10.
- uniform
  - Blanchard, P., Coulaud, O., & Darve, E. (2015). Fast hierarchical algorithms for generating Gaussian random fields.
  - Blanchard, P., Coulaud, O., Darve, E., & Franc, A. (2016). FMR: Fast randomized algorithms for covariance  matrix computations.
  - Blanchard, P., Coulaud, O., Darve, E., & Bramas, B. (2015, October). Hierarchical Randomized Low-Rank Approximations.
  - Blanchard, P., Coulaud, O.,  Etcheverry, A., Dupuy, L., & Darve, E. (2016, June). An Efficient  Interpolation Based FMM for Dislocation Dynamics Simulations.

## Managing parameters (argc, argv)

We provide utility functions in `utils/tbfparams.hpp` to test and convert values from the command line. One can look at the example to see how to use them.

# Issues

## Uniform kernel cannot be used

This is likely that FFTW has not been found on the system. It is possible to verify by running `cmake ..` in the build directory.

```bash
# Example of falling build will have:
-- Could NOT find FFTW (missing: FFTW_INCLUDE_DIRS FLOAT_LIB DOUBLE_LIB) 
-- FFTW Cannot be found, try by setting -DFFTW_ROOT=... or env FFTW_ROOT
.....
-- UTests -- utest-should-not-compile cannot be compiled due to missing libs (/home/berenger/Projects/tbfmm/unit-tests/./utest-should-not-compile.cpp)
-- UTests -- utest-unifkernel-float needs FFTW
....
```

## It is slower with the FMM (compared to direct interactions)

Considering the test is performed in sequential, one has to make sure the correct height of the tree is used (and try to find the best value) and that there are enough particles.

## Make command builds nothing

Ensure that no protective keys are in the source file `@TBF_USE_...` or that they are correct.

## OpenMP algorithm + GCC

We currently have a bug when using OpenMP and GCC. We get a segmentation fault, and we currently think that it is a bug in the compiler (Versions 7 or 8, we do not know for newer versions). It usually works with Clang.

## Cmake error

If the following error happens during the cmake stage:

```bash
CMake Error in CMakeLists.txt:
  Target "exampleEmptyKernel" requires the language dialect "CXX17" , but
  CMake does not know the compile flags to use to enable it.
```

This means that the cmake is too old.

In fact, we need to use the CXX standard configuration:

```bash
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
```



Same if one sees the following error:

```bash
CMake Error at CMakeLists.txt:4 (cmake_policy):
  Policy "CMP0074" is not known to this version of CMake.
```

This would mean that the cmake used is too old.

# Contribution, issues and support

## How to contribute

Contributions are accepted either on Inria's Gitlab (but creating an account there can only be done with an invitation) or on Github.
Fork the `TBFMM` repository and open a merge request (https://gitlab.inria.fr/bramas/tbfmm/-/merge_requests) or a pull request (https://github.com/berenger-eu/tbfmm/pulls).
Any contribution should take into account that we try to minimize the external dependencies' footprint.

## Reporting issues or problems

We track issues at https://gitlab.inria.fr/bramas/tbfmm/-/issues and https://github.com/berenger-eu/tbfmm/issues.

## Seeking support

It is perfectly fine to open an issue to get support.
Otherwise, one should directly contact the author at Berenger.Bramas[[@]]inria.fr

# Pesperctive

TBFMM will be extended with MPI and GPUs.
