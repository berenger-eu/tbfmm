[![pipeline status](https://gitlab.inria.fr/bramas/tbfmm/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/tbfmm/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)

TBFMM is a Fast Multipole Method (FMM) library parallelized with the task-based method.
It is designed to be easy to customize by creating new FMM kernels or new parallelization strategies.
It uses the block-tree (also known as tha group-tree), which is well designed for the task-based parallelization, and can be easily extended to heterogeneous architectures.

The current document is at the same time a classic README but also the main documentation of TBFMM.
We try to answer all the questions that use could have regarding implementation and use of the library.
Of course, we invite users to post an issues if they find a bug or have a question about TBFMM.

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

The build process consists in the following steps: adding git submodules (optional), creating a build directory, running cmake, configuring cmake, running make and that's fine.
The submodules are Inastemp (for vectorization) and SPETABARU (a task-based runtime system).
Both are optional, but to be activated their repository must be cloned/updated before running cmake.

```bash
# To enable SPETABARU and Inastemp
git submodule init && git submodule update
# To enable only SPETABARU (runned from the main directory)
git submodule init deps/spetabaru && git submodule update
# To enable only Inastemp (runned from the main directory)
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

# Update an existing config by calling again cmake -D[an option]=[a value] ..
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
If it is the case, all the OpenMP-based code will be enabled, otherwise it will be removed from the compilation process ensuring that the library can compile (but in sequential or with SPETABARU).

## Inastemp

https://gitlab.inria.fr/bramas/inastemp

## SPETABARU

https://gitlab.inria.fr/bramas/spetabaru


# TBFMM design

## Cell

## Particles

## Tree

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

# Issues

## Uniform kernel cannot be used

## It is slower with the FMM (compared to direct interactions)

## Make command builds nothing

