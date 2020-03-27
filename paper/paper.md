---
title: 'TBFMM: A C++ fast multipole method library for multicore architecture with generic design'
tags:
  - C++
  - FMM
  - OpenMP
  - task-based
  - HPC
authors:
  - name: Berenger Bramas
    orcid: 0000-0003-0281-9709
    affiliation: "1, 2, 3" # (Multiple affiliations must be quoted)
  - name: Stephane Genaud
    affiliation: "2, 3, 4"
affiliations:
 - name: CAMUS Team, Inria Nancy
   index: 1
 - name: Strasbourg University
   index: 2
 - name: ICPS Team, ICube
   index: 3
 - name: Enseeit
   index: 4
date: 26 march 2020
bibliography: paper.bib
---

# Summary

The fast multipole library [@fmm] has been classified as one of the most important algorithms of the 20th century [@siam].
This algorithm was originally designed to compute pair-wise interactions between particles (n-body problems) but has been used and extended to different other type of simulations, such as FEM, BEM, .
The main idea behind the FMM is to avoid computing all the interactions between the elements but to approximate the interaction between far elements, considering that the interaction decrease with the distance.
This makes it possible to reduce the complexity from quadratic to quasi-linear. 
To this achievement, the FMM uses a tree over the simulation box and executes a specific algorithm where cells (nodes of the tree) represent parts of the simulation box and are used to factorize the interactions.

`TBFMM` is an high-performance package that implement the FMM in a generic manner.
Its design allow to easily customize the kernels, the type of interacting elements or the parallelization scheme.

# Background

The FMM algorithm is described using different operators that use letter to express the type of elements they work on: `P` for particle, `M` for multipole and `L` for local.
The term particle is used for legacy reason but it represent the basic elements that interact and for which we want to approximate.
The multipole part represent the aggregation of potential, it represent what is emitted by a sub-part of the simulation box.
Wherease, the local part represent the outside that is emitted onto a sub-part of the simulation box.
The different operator are schemtized in Figure~X.

Because it is a fundamental building blocks for many type of simulation, the FMM parallelization has been investigated.
Traditional pure MPI or MPI+fork-join has been used.
Later task-based parallelization have been developed for multicore~[paper], heterogeneous architecture~[paper] and heterogeneous distributed platforms~[paper].
We have participated on these investegation and we have provided a new hierarchical data structure called group-tree (or block-tree), which is an octree designed for the task-based method.
The two main idea of this container is (1) to allocated and manage several cells of the same level together (2) to split the management of symbolic data, multipole data and local data, such that each memory block can be moved anywhere on the memory and used by a task independently from the other.

# Statement of need


To be used for scientist to create new kernels.
To be used for computer scientist to study block-based FMM and task-based parallelization (as benchmark).
From our size we will apply GPU but with the idea to remain generic.
We will also use it to study scheduling of irregular applications.

More closer existing package is ScalFMM.
But is has X lines of code, for only Y for TBFMM.
It needs several dependencies, does not relies on standard C++ and include lots of old approaches.
It only works for 3D problems, where TBFMM can work for any dimension.
But, the interace for the kernel is very similar to ours such that creating a kernel for ScalFMM or TBFMM it is easy to adapt to the other.

# Features

## Genericity

Full generic design.
Use of template.


## Tree
It uses the block-tree (also called group-tree) where cells of the same level are managed together.
This has been shown that it is well designed for the task-based parallelization including

Automatic bloc-size finder.

## Kernel

Easy to customize.
The parallelization is then automatic

## Parallelization

OpenMP
Spetabaru commute.

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

# Performance

Example of performance

# Conclusion & Perspective


# To be removed

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.
```python
for n in range(10):
    yield f(n)
```	

# Acknowledgements

.

# References
