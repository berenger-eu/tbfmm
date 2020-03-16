[![pipeline status](https://gitlab.inria.fr/bramas/tbfmm/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/tbfmm/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)

TBFMM is a Fast Multipole Method (FMM) library parallelized with the task-based method.
It is designed to be easy to customize by creating new FMM kernels or new parallelization strategies.
It uses the block-tree (also known as the group-tree), which is well designed for the task-based parallelization, and can be easily extended to heterogeneous architectures.

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
Both are optional, but to be activated their repository must be cloned/updated before running cmake.

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

CMake will try to find if OpenMP is supported by the system.
If it is the case, all the OpenMP-based code will be enabled, otherwise it will be removed from the compilation process ensuring that the library can compile (but will run in sequential or with SPETABARU).

## Inastemp

Inatemp is a vectorization library that makes it possible to implement a single kernel with an abstract vector data type, which is then compiled for most vectorization instruction sets.
It includes SSE, AVX(2), AVX512, ARM SVE, etc.
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
Currently, speculation is not used in TBFMM because it requires tasks with a specific pattern.
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

TBFMM can be solved for arbitrary dimension.
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
const std::array<RealType, Dim> BoxWidths{{1, 1, 1}};
const long int TreeHeight = 8;
const std::array<RealType, Dim> BoxCenter{{0.5, 0.5, 0.5}};

const TbfSpacialConfiguration<RealType, Dim> configuration(TreeHeight, BoxWidths, BoxCenter);
```

## Block/group tree

TBFMM uses the block/group tree, which is an octree where several cells of the same level are allocated and managed together as a group of cells (or as a memory block).
This data structure is well described here http://berenger.eu/blog/wp-content/uploads/2016/07/BerengerBramas-thesis.pdf

The advantage of this structure is that the memory blocks can be moved/copied with a `memcpy`, which is really convenient in HPC when we want to use GPUs or MPI.
The side effect is a great data locality and forcing a low level abstraction on the data in the group, which mitigates the overhead.

On the other hand, the constraint (drawback?) is that particles and cells must be POD (or at least to be raw copiable).
For instance, an `std::vector` is not because, because it has a pointer internally such that making a raw copy of an object will lead to an undefined/invalid state (it could work in a shared memory system, but clearly not with different memory nodes).

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
This is a structure/class that represent the target of a P2M/M2M, or the source of a M2L.
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
By default, it will include the positions of the particles and it could also be any type of physical values, such as weights.
These are the sources values of P2M and P2P, but they are also used in the L2P.
The data type of these values can be changed.
We usually call it `DataParticleType` and make it equal to the `RealType`, which is `float` or `double` in most cases.
The user can only specify how many of symbolic values per particle, but the type will `RealType`.
The first values are used to store the particles' positions, and the extra values can be used by the users as they want.

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

In TBFMM, the tree will transform a set of particles into a block-tree and support all the access methods that the kernels need to perform the FMM algorithm.
Therefore, the tree class must be defined with all the template parameters related to the cells and particles.
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

The tree needs four parameters:
- spacial configuration

## Kernel

## Algorithms

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

# Issues

## Uniform kernel cannot be used

## It is slower with the FMM (compared to direct interactions)

## Make command builds nothing

