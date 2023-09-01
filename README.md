# mayawaves

`Mayawaves` serves as an analysis library for Einstein Toolkit and MAYA simulations. Follow the tutorials and documentation at https://mayawaves.github.io/mayawaves to learn how
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

## Contributing

`Mayawaves` is intended to be a library that will grow and expand with additional analysis tools. We welcome the 
input of the community to both request new features and help in implementing them.
