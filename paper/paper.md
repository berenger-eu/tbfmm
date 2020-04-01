---
title: 'TBFMM: A C++ generic and parallel fast multipole method library'
tags:
  - C++
  - FMM
  - OpenMP
  - task-based
  - HPC
authors:
  - name: Berenger Bramas
    orcid: 0000-0003-0281-9709
    affiliation: "1, 2, 3"
  - name: Stephane Genaud
    affiliation: "2, 3, 4"
affiliations:
 - name: CAMUS Team, Inria Nancy
   index: 1
 - name: Strasbourg University
   index: 2
 - name: ICPS Team, ICube
   index: 3
 - name: Enseeiht
   index: 4
date: 26 march 2020
bibliography: paper.bib
---

# Summary

`TBFMM` is a high-performance package that implements the parallel fast multipole method in modern `C++17`.
`TBFMM` was designed with the aim of being easy to customize thanks to `C++` templates and a fine control of the `C++` classes inter-dependencies.
Users can implement new FMM kernels, new types of interacting elements or even new parallelization strategies.
Specifically, `TBFMM` can be used to perform research in HPC to study parallelization/optimization and scheduling, but it can also be used as a simulation toolbox for scientists in physics or applied mathematics.
It enables users to perform simulations while delegating the data structure, the algorithm and the parallelization to the library.

# Background

The FMM [@GREENGARD1987325] has been classified as one of the most important algorithms of the 20th century [@cipra2000best].
This algorithm was originally designed to compute pair-wise interactions between `N` particles, which belong to the class of n-body problems.
It reduces the complexity from quadratic (`N` elements interact with `N` elements) to a quasi-linear complexity.
The central idea is to avoid computing the interactions between all the elements by approximating the interactions between elements that are far enough.
To make it possible, the algorithm needs that the potential of the interactions decreases as the distance between interacting elements increases, and that the kernel to approximate far interaction exists.
In fact, providing an approximation kernel for a given physical equation can be challenging. 
Internally, the FMM is usually implemented with a tree that is mapped over the simulation box.
A cell, i.e. a node of the tree, represents a part of the simulation box and is used by the algorithm to factorize the interactions between elements.
The FMM was later extended for different type of physical simulations and different approximation kernels [@SABARIEGO2004403,pham2012fast,sabariego2004fast,frangi2003coupled,barba2011exafmm,malhotra2015pvfmm,darve2004fast,darve2013optimizing,blanchard2016efficient,blanchard2015fast].

The FMM algorithm is based on six operators that have names that respect the format `X2Y`, where `X` represents the source of the operator and `Y` the destination.
The operators are `P2M`, `M2M`, `M2L`, `L2L`, `L2P` and `P2P`, where `P` means particle, `M` multipole or `L` local.
The term particle is used for legacy reason but it represents the basic interaction elements that interact and for which we want to approximate the interactions.
The multipole part represent the aggregation of potential, i.e. it represents what is emitted by a sub-part of the simulation box.
Whereas, the local part represents the outside that is emitted onto a sub-part of the simulation box.
The different operators are schematized in Figure \autoref{fig:fmm}.

![Illustration of the FMM algorithm.
(a,b,c) Building of the octree.
(d,e,f,g) The FMM algorithm and its operators.
\label{fig:fmm}](FMM.png)

Because it is a fundamental building blocks for many types of simulation, the FMM parallelization has already been investigated.
Some strategies have been developed using classical HPC technologies like `MPI` [@10.5555/898758], to parallelize over multiple distributed memory nodes, potentially enhanced with a fork-join threaded library [@bramas2016optimization].
However, it has been demonstrated that fork-join schemes are less efficient than task-based parallelization on multicore CPUs~[doi:10.1137/130915662].
This is because some part of the FMM have a small degree of parallelism (for instance at the top of the tree), while other have high degree with a significant workload available from the early beginning of each iteration (for instance the `P2P` in the direct pass).
The task-based method is capable of interleaving the different operators, hence to balance the workload across the processing units and to spread the critical parts over time.
Moreover, the task-based method is well designed for handling heterogeneous architecture~[doi:10.1002/cpe.3723] and it has demonstrated promising performance on distributed memory platforms too [agullo:hal-01387482].

In a previous project called `ScalFMM`, we have provided a new hierarchical data structure called group-tree (or block-tree), which is an octree designed for the task-based method.
The two main ideas behind this container are (1) to allocated and manage several cells of the same level together to control the granularity of the tasks, and (2) to split the management of symbolic data, multipole data and local data, such that each memory block can be moved anywhere on the memory nodes and used by the task dependency system apart from each other.
A schematic view of the group-tree is given in Figure \autoref{fig:blocktree}.

![Caption for example figure.\label{fig:blocktree}](grouptree.png)

# Statement of need

The FMM is a major algorithm but it remains rare to have it included in HPC benchmarks when studying runtime systems, schedulers or optimizers.
The main reason is because it is tedious to implement and requires a significant programming effort when using the task-based method together with the group-tree.
However, it is an interesting, if not unique, algorithm to study irregular/hierarchical scientific method.
For the same reason, it is difficult for researchers in physics or applied mathematics to implement a complete FMM library and to optimize it for modern hardware, especially when their objectives is to focus on approximation kernels.
Therefore, `TBFMM` can be useful for both communities.

Among the few FMM libraries that exist, the closer existing package to `TBFMM` is  `ScalFMM`.
`ScalFMM` has around 170K lines of code, for only 50K for `TBFMM`, because it supports lots of different parallel strategies, including fork-join implementations, and contains several experimental methods.
Moreover, it needs several external dependencies and does not benefit from new standard `C++` features that could improve code readability.
In addition, it only works for 3D problems, where `TBFMM` can work for arbitrary dimension.
This has been a motivation to re-implement from scratch a lightweight FMM library that only supports task-based parallelization.

However, the interface of the kernels is very similar in both libraries, such that creating a kernel for `ScalFMM` or `TBFMM` and porting it to the other library is straightforward.

# Features

## Genericity

TBFMM is designed to be generic thanks to an heavy use of `C++` templates.
The tree and the kernel classes are independent from each other and from the algorithm.
The algorithm has to be templatized to know the type of the kernel, and the method `execute` of the algorithm has to be templatized with the type of the tree. 
The algorithm takes the elements from the tree and passes it to the kernel, such that a kernel never accesses the tree itself.
This is illustrated by the Figure \autoref{fig:design}.

![`TBFMM` design overview.
The `Types` of each class should be templatized, at the exception of the types of the kernel where it is optional.
The algorithm has to be selected among different variants (sequential, parallel OpenMP or parallel SPETABARU).
\label{fig:design}](./design.png)



## Tree

`TBFMM` uses the group-tree (also called block-tree) where cells of the same level are managed together.
Users can select the size of the groups, which impacts the size of tasks, but `TBFMM` also provides a simple heuristic to find a size automatically that will allow obtaining efficient executions.
In addition, the tree class provide different methods to iterate on the cells/leaves as any container, such that it is possible to work on the elements of the tree with an abstraction mechanism and without knowing how this is one is implemented.

## Kernel

As we have clearly stated in the objectives, `TBFMM` is a tool for scientist from physics and applied mathematics to create new kernels.
`TBFMM` offers an convenient way to customize the kernel and to benefit from the underlying parallelization engine automatically.
With this aim, a user has to create a new kernel that respect an interface, as it is described with lots of details in the documentation.


## Parallelization

`TBFMM` parallel algorithms use two task-based runtime systems: `OpenMP` version 4 [@openmp4] and `SPETABARU` [@10.7717/peerj-cs.183]. 
Both are optional, hence the library can compiled even if the compiler does not support `OpenMP` or if the `Git` sub-module for `SPETABARU` has not been activated.
While `OpenMP` is an API that evolve slowly, with backward compatibility guarantee, and which is implemented by different libraries that respect the standard, `SPETABARU` is our task-based runtime system that we use for research.
Therefore, we continuously work on this parallel engine.
The data accesses of the FMM operators in `write` are usually commutative [@7912335].
Commutative `write` access is supported by `SPETABARU`.
However, in `OpenMP` the `mutexinout` data access that has been created to express commutative `write` has been defined in `OpenMP` version 5 and is currently not supported by the compilers.
When a compiler that supports this access will be used with `TBFMM`, the `mutexinout` will be used automatically.

## Periodicity

The periodicity consists in considering that the simulation box is repeated in all direction, as shown in Figure \autoref{fig:periodicillu}.
Computing the FMM algorithm with periodicity is usually done in two steps.
In the first one, the regular algorithm and tree are used, with the only difference is to takes the cells at the opposite of the simulation box when we look for cells outside the boundaries.
In the second step, a numerical model is used to compute a potential that represents the world outside the simulation box.
Such model could be the Ewald summation [@407723].

![In the periodic FMM, the simulation box is considered to be in the middle of a infinite volume of the same kind.
\label{fig:periodicillu}](periodicillu.png)

In `TBFMM`, we have implemented a different approach, which is a pure algorithmic strategy [@bramas2016optimization].
The idea is to consider that the the real simulation box is a sub-part of a larger simulation box, i.e. that the real tree is a branch of a larger tree.
Then, instead of stopping the FMM algorithm at level 2, we continue up to the root where the multipole part of the root represent the complete simulation box.
We use it by continuing the FMM algorithm partially above the root by aggregating the cells together multiple times.
By doing so, we have several advantages.
This method needs nothing more than an FMM kernel, which is expected to be the same as the one use without periodicity.
Secondly, the accuracy of the method relies on the FMM kernel.
Therefore, the method is generic and can work with any FMM kernel. 
Figure \autoref{fig:periodicmerge} shows how the simulation box is repeated with this method.

![How the simulation box is repeated when using periodic FMM algorithm.
\label{fig:periodicmerge}](periodicmerge.png)

## Vectorization (Inastemp)

When implementing a kernel, some parts can rely on well optimized numerical libraries, such as BLAS or FFTW, but others might be implemented directly in `C/C++`.
In this case, it usually provides a significant improvement in performance to vectorize the code to use SIMD capability of modern CPUs.
With this aim, `TBFMM` can include a vectorization library called `Inastemp` [@bramas2017inastemp] by simply cloning the corresponding `Git` sub-module.
Using `Inastemp`, it is possible to write a single code with an abstract vector data type, and to select at compile time the desired instruction set depending on the CPU (`SSE`, `AVX`, `SVE`, etc.).
In the current version of `TBFMM`,  the `P2P` operator of the two kernels used for demonstration is vectorized with `Inastemp`.

# Performance

In Figure \autoref{fig:performance}, we provide parallel efficiency of `TBFMM` for a set of particles randomly distributed in a square simulation box.
The given results have been computed using the `uniform` kernel on a X CPU.

![Caption for example figure.\label{fig:design}](performance.png)

# Conclusion & Perspective

`TBFMM` is lightweight FMM library that could be use to do research in HPC and applied mathematics.
We will include it in our benchmarks to evaluate scheduling strategies, but also to validate new approaches to develop numerical applications on heterogeneous computing nodes.
Indeed, we would like to offer an elegant way for users to add GPU kernels while delegating most of the complexity to `TBFMM` and `SPETABARU`.
We also plan to provide an `MPI` version to support distributed memory parallelization in a near future.


# Acknowledgements

Acknowledgment: Experiments presented in this paper were carried out using the PlaFRIM experimental testbed, supported by Inria, CNRS (LABRI and IMB), Université de Bordeaux, Bordeaux INP and Conseil Régional d'Aquitaine.

# References
