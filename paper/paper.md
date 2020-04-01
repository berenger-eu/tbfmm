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

`TBFMM` is a high-performance package that implements the parallel fast multipole method in modern `C++`.
`TBFMM` was designed with the aim of being easy to customize thanks to `C++` templates and a fine control of the classes inter-dependencies.
Users can implement new FMM kernels, new types of interacting elements or even new parallelization strategies.
Specifically, `TBFMM` can be used to perform research in HPC to study parallelization and scheduling approaches, but it can also be used as a simulation toolbox for scientists in physics or applied mathematics.
It enables users to perform simulations while delegating the data structure, the algorithm and the parallelization to the library.

# Background

The FMM [@GREENGARD1987325] has been classified as one of the most important algorithms of the 20th century [@cipra2000best].
This algorithm was originally designed to solve n-body problems, such as computing pair-wise interactions between particles.
It reduces the complexity from quadratic (N elements interact with N elements) to a quasi-linear complexity.
The central idea is to avoid computing the interactions between all the elements by approximating the interactions between elements that are far enough.
To make it possible, the algorithm needs that the potential of the interactions decreases as the distance between interacting elements increases, and that the kernel to approximate far interaction exists.
In fact, providing an approximation kernel for a given physical equation can be quite challenging. 
Internally, the FMM is usually implemented with a tree that is mapped over the simulation box.
A cell, i.e. a node of the tree, represents a part of the simulation box and is used by the algorithm to factorize the interactions between elements.
The FMM was later extended for different type of physical simulations and different approximation kernels [@SABARIEGO2004403,pham2012fast,sabariego2004fast,frangi2003coupled,barba2011exafmm,malhotra2015pvfmm,darve2004fast,darve2013optimizing,blanchard2016efficient,blanchard2015fast].

The FMM algorithm is based on six operators that have a name respecting the format `X2Y`, where `X` represents the source of the operator and `Y` the destination.
`X` and `Y` can be either `P` for particle, `M` for multipole or `L` for local.
The term particle is used for legacy reason but it represents the basic interaction elements that interact and for which we want to approximate the interactions.
The multipole part represent the aggregation of potential, i.e. it represent what is emitted by a sub-part of the simulation box.
Whereas, the local part represent the outside that is emitted onto a sub-part of the simulation box.
The different operators are schematized in Figure \autoref{fig:fmm}.

![Illustration of the FMM algorithm.
(a,b,c) Building of the octree.
(d,e,f,g) The FMM algorithm and its operators.
\label{fig:fmm}](FMM.png)

Because it is a fundamental building blocks for many types of simulation, the FMM parallelization has been investigated.
Some strategies have been developed using classical HPC technologies such as a `MPI` [@10.5555/898758], to parallelize over multiple distributed memory nodes, potentially enhanced with a fork-join threaded library [@bramas2016optimization].
However, it has been demonstrated that fork-join strategies are less efficient than task-based parallelization on multicore CPUs~[doi:10.1137/130915662].
This is because some part of the FMM have a small degree of parallelism (for instance at the top of the tree), while other have high degree with a significant workload available from the beginning (for instance the `P2P` in the direct pass).
The task-based method is capable of interleaving the different operators, hence to balance the workload across the processing units and to spread the critical parts over time.
Moreover, the task-based method is well designed for handling heterogeneous architecture~[doi:10.1002/cpe.3723] and it has demonstrated nice performance on distributed memory platforms too [agullo:hal-01387482].
In a previous project called `ScalFMM`, we have provided a new hierarchical data structure called group-tree (or block-tree), which is an octree designed for the task-based method.
The two main ideas behind this container are (1) to allocated and manage several cells of the same level together (2) to split the management of symbolic data, multipole data and local data, such that each memory block can be moved anywhere on the memory and used by the task dependency system apart from each other, as illustrated in Figure \autoref{fig:blocktree}.

![Caption for example figure.\label{fig:blocktree}](grouptree.png)

# Statement of need

The FMM is a major algorithm but it remains rare to have it included in HPC benchmarks when studying runtime systems, schedulers or optimizers.
The main reason is because it is tedious to implement and requires a significant programming effort when using the task-based method together with the group-tree.
However, it is an interesting, if not unique, algorithm to study irregular/hierarchical scientific method.
For the same reason, it is difficult to researchers in physics or applied mathematics to implement a complete FMM library and to optimize it for modern hardware, especially when their objectives is to focus on the study of an approximation kernel.
Therefore, `TBFMM` with its generic design and absence of required dependencies can be useful for both communities.

Among the few FMM libraries that exist, the closer existing package to `TBFMM` is  `ScalFMM`.
`ScalFMM` has around 170K lines of code, for only 50K for `TBFMM`, because it supports lots of different parallel strategies, including some based on the fork-join model, and contains several experimental methods.
Moreover, it needs several external dependencies and does not benefit from the new standard `C++` features.
In addition, it only works for 3D problems, where `TBFMM` can work for arbitrary dimension.

However, the interface of the kernels is very similar in both libraries, such that creating a kernel for `ScalFMM` or `TBFMM` and porting it to the other library is convenient.

# Features

## Genericity

TBFMM is designed to be generic thanks to an heavy use of `C++` template.
The tree and the kernel classes are independent from each other and from the algorithm.
The algorithm has to be templatized to the use the tree and the kernel and bridge the gap between them.
This is illustrated by the Figure \autoref{fig:design}.

![`TBFMM` design overview.
The `Types` of each class should be templatized, at the exception of the types of the kernel where it is optional.
The algorithm has to be selected among different variants (sequential, parallel OpenMP or parallel SPETABARU).
\label{fig:design}](./design.png)



## Tree

`TBFMM` uses the group-tree (also called block-tree) where cells of the same level are managed together.
Users can select the size of the groups, but `TBFMM` also provides a simple heuristic to find a size automatically that will allow for obtaining efficient executions.
The tree class provide different methods to iterate on the cells/leaves as any container.

## Kernel

One of the main objective of `TBFMM` is to provide a tool for scientist from physics and applied mathematics to create new kernels.
Therefore, `TBFMM` offer an easy way to customize the kernel and to benefit from the underlying parallelization engine automatically.
With this aim, a user has to create a new kernel that respect an interface, as described in the documentation.


## Parallelization

`TBFMM` currently use two task-based runtime systems: `OpenMP` version 4 [@openmp4] and `SPETABARU` [@10.7717/peerj-cs.183]. 
The data access of the FMM operators in write are usually commutative [@7912335].
However, in OpenMP the `mutexinout` data access has been defined in OpenMP 5 and is currently not supported yet (when a compiler that supports it will be used, `TBFMM` will use it automatically).

## Periodicity

The periodicity consists in considering that the simulation box is repeated unlimited in all direction.
To compute the potential of the periodic box over the simulation, it is classic to use the X approach.
In `TBFMM` we have implemented a different approach, which is a pure algorithmic strategy [@bramas2016optimization].
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

It could be beneficial to vectorize some part of the code in order to benefit from the performance of modern CPUs.
With this aim, `TBFMM` is able to include `Inastemp` [@bramas2017inastemp], which is a vectorization library.
Using `Inastemp` it possible to write a single kernel using an abstract vector type and to select at a compile the target instruction set depending on the CPU.
In the current code, the two kernels used for demonstration have the `P2P` vectorized with `Inastemp`, but one could use it to vectorize any other operators.

# Performance

Example of performance

![Caption for example figure.\label{fig:design}](performance.png)
and referenced from text using \autoref{fig:performance}

# Conclusion & Perspective

`TBFMM` is lightweight FMM library that could be use to do research in HPC and applied mathematics.
We will use it in benchmarks to study scheduling strategy, but also to test new approaches to develop on heterogeneous computing.
Indeed, we would like to offer an elegant way for users to add GPU kernels while delegating most of the things to `TBFMM` and `SPETABARU`.
We also plan to use `MPI` to support distributed memory parallelization.




# Acknowledgements

Acknowledgment: Experiments presented in this paper were carried out using the PlaFRIM experimental testbed, supported by Inria, CNRS (LABRI and IMB), Université de Bordeaux, Bordeaux INP and Conseil Régional d'Aquitaine.

# References
