.. Mayawaves documentation master file, created by
   sphinx-quickstart on Thu May  6 10:28:27 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Mayawaves's documentation!
=====================================

`Mayawaves` serves as an analysis library for Einstein Toolkit and MAYA simulations. Follow the tutorials to learn how
to use this library to analyze both your own simulations and the simulations in the MAYA Catalog
(https://cgp.ph.utexas.edu/waveform).

The Coalescence object is the heart of the library. It interacts with an h5 file that contains everything about the
simulation. Each Coalescence object has CompactObjects for each of the black holes involved in the simulation.
Refer to the compact_objects tutorial for more details on how to use CompactObjects.

Each Coalescence object also has a RadiationBundle that handles all the information pertaining to gravitational
radiation. This is primarily done behind the scenes and you can access all gravitational wave information directly
through Coalescence. Refer to the gravitational_waves tutorial for more information on reading gravitational wave data.

There is also a utility to stitch a raw simulation into the h5 format that the Coalescence object reads. This is
shown in the creating_h5 tutorial. Coalescence objects can also be exported to the format used by LVK analyses, as shown in
the exporking_lvk tuttorial.

The catalog_utils tutorial walks through using the Catalog object in the catalogutils module to obtain metadata about
the MAYA catalog and to download simulations from the MAYA catalog.

.. toctree::
   :maxdepth: 2
   :caption: Documentation:

   source/mayawaves.coalescence
   source/mayawaves.compactobject
   source/mayawaves.radiation
   source/mayawaves.utils

.. toctree::
   :maxdepth: 1
   :Caption: Tutorials:

   source/notebooks/creating_h5
   source/notebooks/gravitational_waves
   source/notebooks/compact_objects
   source/notebooks/exporting_lvk
   source/notebooks/reading_raw_files
   source/notebooks/catalog_utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
