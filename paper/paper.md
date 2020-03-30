---
title: 'TBFMM: A generic and parallel fast multipole method library in C++'
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

`TBFMM` is an high-performance package that implement the parallel fast multipole method in modern `C++`.
`TBFMM` has been designed to be highly customized.
The user can implement new kernels, new type of interacting elements or even new parallelization strategy and plug them into `TBFMM`.
Therefore, `TBFMM` can be used to perform research in HPC but it can also be used as a simulation tools by scientists in physics or applied mathematics.
Specifically, this package enables to perform simulation while delegating the data structure, the algorithm and the parallelization.



# Background

The FMM [@fmm] has been classified as one of the most important algorithms of the 20th century [@siam].
This algorithm was originally designed to solve n-body problems, such as computing pair-wise interactions between particles.
It was later used and extended for different type of simulations, such as FEM, BEM, .
The FMM makes it possible to reduce the complexity from quadratic (N elements interact with N elements) to a quasi-linear complexity.
The central idea is to avoid computing all the interactions between the elements but to approximate the interaction between elements are far enough.
To work, the interaction to be computed must decrease with the distance, and the kernels to approximate far interaction must defined. 
Internally, the FMM is usually implemented with a tree that is mapped over the simulation box, where cells (nodes of the tree) represent parts of the simulation box and are used to factorize the interactions between elements.



The FMM algorithm is described using different operators that use letter to express the type of elements they work on: `P` for particle, `M` for multipole and `L` for local.
The term particle is used for legacy reason but it represent the basic elements that interact and for which we want to approximate.
The multipole part represent the aggregation of potential, it represent what is emitted by a sub-part of the simulation box.
Wherease, the local part represent the outside that is emitted onto a sub-part of the simulation box.
The different operator are schemtized in Figure~X.

![Caption for example figure.\label{fig:fmm}](FMM.png)
and referenced from text using \autoref{fig:fmm}

Because it is a fundamental building blocks for many type of simulation, the FMM parallelization has been investigated.
Traditional pure MPI or MPI+fork-join has been used.
Later task-based parallelization have been developed for multicore~[paper], heterogeneous architecture~[paper] and heterogeneous distributed platforms~[paper].
We have participated on these investegation and we have provided a new hierarchical data structure called group-tree (or block-tree), which is an octree designed for the task-based method.
The two main idea of this container is (1) to allocated and manage several cells of the same level together (2) to split the management of symbolic data, multipole data and local data, such that each memory block can be moved anywhere on the memory and used by a task independently from the other.

![Caption for example figure.\label{fig:blocktree}](grouptree.png)
and referenced from text using \autoref{fig:blocktree}

# Statement of need

More closer existing package is ScalFMM.
But is has X lines of code, for only Y for TBFMM.
It needs several dependencies, does not relies on standard C++ and include lots of old approaches.
It only works for 3D problems, where TBFMM can work for any dimension.
But, the interace for the kernel is very similar to ours such that creating a kernel for ScalFMM or TBFMM it is easy to adapt to the other.

To be used for scientist to create new kernels.
To be used for computer scientist to study block-based FMM and task-based parallelization (as benchmark).
From our size we will apply GPU but with the idea to remain generic.
We will also use it to study scheduling of irregular applications.

# Features

## Genericity

TBFMM is designed to be generic thanks to an heavy use of C++ template.
For instance, the tree and the kernel are independent from each other and from the algorithm.
The algorithm has to be templatized to the use the tree and the kernel and bridge the gap between them.
This is illustrated by the Figure X.

![Caption for example figure.\label{fig:design}](design.png)
and referenced from text using \autoref{fig:design}

## Tree

TBFMM uses the block-tree (also called group-tree) where cells of the same level are managed together.
This has been shown that it is well designed for the task-based parallelization including
TBFMM also provides a simple heuristic to find a bloc-size automatically that will allow for obtaining efficient executions.
It is also possible to iterate on the cells/leaves of the tree.

## Kernel

One of the main objective of TBFMM is to provide a tool for scientist from physics and applied mathematics to create new kernels.
Therefore, TBFMM offer an easy way to customize the kernel.
The parallelization is then automatic and independent of the underlying parallelization engine.


## Parallelization

TBFMM currently use two task-based runtime systems.
It uses OpenMP tasks version 4.5.
It is ready to benefit from the new `mutexinout` from OpenMP 5.
It also uses Spetabaru, which is a runtime system used for research to study scheduling, task-based programming and speculative execution.
It supports commute data access.

## Periodicity

The periodicity consists in considering that the simulation box is repeated unlimited in all direction.
To compute the potential of the periodic box over the simulation, it is classic to use the X approach.
In TBFMM we have implemented a different approach, which is a pure algorithmic strategy [cite].
The idea is to consider that the the FMM is a sub-part of a more larger tree.
Then, instead of stopping the algorithm up to level 2, we continue up to the root where the multipole part of the root represent the complete simulation box.
We use it by continuing the FMM algorithm partially above the root.
By doing so, we have several advantages.
The method needs nothing more than a FMM kernel (the same as the one use without periodicity).
The accuracy relies on the FMM kernel.
The method is generic.

![Caption for example figure.\label{fig:periodicfmm}](periodicfmm.png)
and referenced from text using \autoref{fig:periodicfmm}

## Vectorization (Inastemp)



# Performance

Example of performance

![Caption for example figure.\label{fig:design}](performance.png)
and referenced from text using \autoref{fig:performance}

# Conclusion & Perspective


# Acknowledgements

.

# References
