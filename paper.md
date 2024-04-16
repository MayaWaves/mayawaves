---
title: 'Mayawaves: Python Library for Interacting with the Einstein Toolkit and the MAYA Catalog'
tags:
  - Python
  - Numerical Relativity
  - Gravitational Waves
  - Black Holes
authors:
  - name: Deborah Ferguson
    orcid: 0000-0002-4406-591X
    corresponding: true
    affiliation: 1, 2
  - name: Surendra Anne
    affiliation: 1
  - name: Miguel Gracia-Linares
    orcid: 0000-0002-5033-2973
    affiliation: 1
  - name: Hector Iglesias
    affiliation: 1
  - name: Aasim Jan
    affiliation: 1
  - name: Erick Martinez
    affiliation: 1
  - name: Lu Lu
    affiliation: 1
  - name: Filippo Meoni
    affiliation: 1
  - name: Ryan Nowicki
    affiliation: 1
  - name: Max L. Trostel
    affiliation: 1
    orcid: 0000-0001-9743-0303
  - name: Bing-Jyun Tsao
    affiliation: 1
    orcid: 0000-0003-4614-0378
  - name: Finny Valorz
    affiliation: 1
    
affiliations:
  - name: University of Texas at Austin, Austin TX, USA
    index: 1
  - name: University of Illinois Urbana-Champaign, Urbana IL, USA
    index: 2
date: \today
bibliography: paper.bib
---


# Summary

Einstein's Theory of General Relativity (GR) dictates how matter responds to the curvature of space-time and how space-time curves due to matter.
From GR came the prediction that orbiting massive objects would create ripples in space-time called gravitational waves (GW).
In 2015, the Laser Interferometer Gravitational-Wave Observatory (LIGO) [@LIGOScientific:2014pky] detected the first such GW signal from merging binary black holes (BBHs) [@LIGOScientific:2016aoc], and in the years since, the LIGO, Virgo, and KAGRA Collaborations (LVK) have accumulated 90 detections of merging compact objects [@VIRGO:2014yos; @KAGRA:2020cvd; @LIGOScientific:2018mvr; @LIGOScientific:2020ibl; @LIGOScientific:2021djp].
Extracting these signals from noise and using them to infer the parameters of coalescing black holes (BHs) relies upon having vast template banks that accurately predict the expected GWs [@LIGOScientific:2017vwq; @LIGOScientific:2020aai; @Shibata:2017xdx; @Hannam:2013oca; @Bohe:2016gbl; @Khan:2015jqa; @Blackman:2017pcm; @Husa:2015iqa; @LIGOScientific:2016fbo; @Lange:2017wki; @Schmidt:2017btt].

While analytic solutions exist for the simplest cases within GR, e.g. single BHs, merging BBHs have no analytic solution.
Approximate methods can be used when the BHs are far apart or have highly unequal masses, but the coalescence of BHs of comparable mass must be solved computationally.
Numerical relativity (NR) simulations accomplish this by evolving a BBH space-time on supercomputers, enabling us to study the dymanics of BBH systems as well as predict the GWs they emit.
The Einstein Toolkit (ETK) is a set of tools created to perform these NR simulations [@Loffler:2011ay], and `MAYA` is a branch of ETK used by the MAYA collaboration [@Herrmann_2007; @Jani:2016wkt; @Vaishnav:2007nm; @Healy:2009zm; @Pekowsky:2013ska].
The Einstein Toolkit is a finite-differencing code, evolved using the BSSN formulation [@Baumgarte:1998te; @PhysRevD.52.5428].
It is built upon the Cactus infrastructure [@Goodale2002a] with Carpet mesh refinement [@Schnetter:2003rb].

These tools allow us to study the coalescence of compact objects, their evolution, and the gravitational radiation they emit.
The `Mayawaves` library introduced in this paper is an analysis pipeline used to process and analyze such NR simulations, specifically for BBHs. 

# Statement of need

NR simulations are crucial for studying BHs and have been instrumental in the detection of GWs by the LVK.
However, these simulations produce vast amounts of data that must be processed in order to perform studies, create models, and use them with GW detection pipelines.
Additionally, given the complexity of these simulations, they are typically performed for many days or weeks across many processors, leading to data which is split into several output directories and files.
Sifting through all this data can be overwhelming for newcomers to the field and is cumbersome for even the most experienced numerical relativists.
While it can be important to develop an understanding of these files and their complexities, in many situations, a simpler, more streamlined workflow is beneficial.

`Mayawaves` is an open-source python library for processing, studying, and exporting NR simulations performed using ETK and `MAYA`.
While other tools exist to analyze ETK simulations including, but not limited to, Kuibit [@Bozzola_kuibit_2021], Power [@Johsnon], PyCactus [@Kastaun], and SimulationTools [@Hinder], `Mayawaves` is unique in the way it not only streamlines simulation analysis but also the production of NR catalogs.
`Mayawaves` builds upon the existing set of tools, creating a new python library designed for convenience and intuition, while still being versatile and powerful enough to perform more complex analyses.
When using the library to interact with a simulation, the  user does not need to be familiar with all the types of output files generated by the simulation, but rather, can think in terms of physical concepts such as *coalescences* and *compact objects*.
This dramatically reduces the barrier to entry for the field of NR.
The architecture of `Mayawaves` is easily extensible, designed with the purpose of being able to naturally grow to encompass more types of simulation output.
One of the ways in which `Mayawaves` has uniquely improved the NR analysis infrastructure is that it stitches together raw NR simulations and stores them in h5 files, a format that handles numerical data more efficiently than ascii. 
This significantly reduces the disc space taken by the simulation while still retaining the precision of the raw data.
It also keeps all the data organized in one place, making it easier to share and distribute simulations.
`Mayawaves` is designed with waveform catalogs in mind.
For this purpose, it interacts effortlessly with the `MAYA` waveform catalog and also easily exports ETK simulations to the LVK catalog of NR waveforms.

Some of the key functionalities of this library are as follows:

* read and stitch raw numerical relativity data
* store numerical relativity data in a more efficient format
* track properties of the black holes (trajectories, spins, masses, horizon information, etc.)
* compute information about the binary orbit (eccentricity, separation, orbital frequency, kick, etc.)
* compute quantities pertaining to the gravitational radiation ($\Psi_4$, strain, extrapolated strain, energy radiated, etc.)
* export numerical relativity data to the format required for LIGO analyses
* analyze the MAYA waveform catalog


The `Coalescence` class is the fundamental basis for `Mayawaves`.
It represents the entirety of the BBH coalescence and serves as the interface between the user and the simulation data.
The main data format used with `mayawaves` is an h5 file constructed from the raw simulation data.
With this h5 file in hand, the user need only create a `Coalescence` object and then proceed with analyzing the data.

Each `Coalescence` object contains `CompactObjects` associated with each of the merging bodies as well as any remnant object.
Through these `CompactObjects`, the user can track the objects' positions, spins, masses, etc.
All radiative information is stored within the `RadiationBundle` class.
Each `Coalescence` object contains a `RadiationBundle` and uses it to compute gravitational wave strain, energy radiated, etc.

A number of utility modules are included to create effortless workflows that can move from raw simulations to community standard formats.
A typical workflow would involve using the `PostProcessingUtils` functions to create the h5 file from the raw simulation data, using the `Coalescence` class to read that h5 file and analyze the simulation, and finally exporting the `Coalescence` object to another format such as that required by the LVK catalog [@Schmidt:2017btt].

`Mayawaves` is also the primary way to interact with the MAYA Public Catalog of NR waveforms hosted at https://cgp.ph.utexas.edu/waveform.
The simulations are stored in the `Mayawaves` h5 file structure, and can be read using the `Coalescence` class.
`Mayawaves` has a `CatalogUtils` module for interacting with the MAYA waveform catalog.
This module includes functions for accessing and plotting the metadata for the entire catalog as well as functions to download simulations from the catalog.

`Mayawaves` is open source and is designed to be easily extensible, and we look forward to additional contributions from the ETK community.

# Acknowledgements

The authors thank Deirdre Shoemaker, Pablo Laguna, and Helvi Witek for their support throughout the duration of this code development.
The work presented in this paper was possible due to grants NASA 80NSSC21K0900, NSF 2207780 and NSF 2114582.
This work was done by members of the Weinberg Institute and has an identifier of UTW1-33-2023.

# References