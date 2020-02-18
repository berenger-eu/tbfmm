[![pipeline status](https://gitlab.inria.fr/bramas/tbfmm/badges/master/pipeline.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)
[![coverage report](https://gitlab.inria.fr/bramas/tbfmm/badges/master/coverage.svg)](https://gitlab.inria.fr/bramas/tbfmm/commits/master)

TBFMM is a Fast Multipole Method library parallelized with task-based method.
It is designed to be easy to customize and to create new FMM kernels.

# Compilation

Simply go in the build dir and use cmake as usual:
```
# To enable SPETABARU
git submodule init && git submodule update
# Go in the build dir
mkdir build
cd build
# To enable testing: cmake -DUSE_TESTING=ON -DUSE_SIMU_TESTING=ON ..
cmake ..
# To find FFTW
cmake -DFFTW_ROOT=path-to-fftw ..
# or (set environement variables FFTW_DIR FFTWDIR)
cmake ..

# Build
make
```

# Running the tests

Use
```
make test
```

To get the output of the tests, use:
```
CTEST_OUTPUT_ON_FAILURE=TRUE make test
```

# Coverage result

Can be found here: https://bramas.gitlabpages.inria.fr/tbfmm/

