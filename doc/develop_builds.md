
# Places to build options: architecture, device, advanced options, cuda options

These are the files that need to be updated when a new architecture or device is
added:

  + generate_makefile.bash
      * Interface for makefile system
  + cmake/kokkos_options.cmake
      * Interface for cmake system
  + Makefile.kokkos
      * Main logic for build (make and cmake) and defines (KokkosCore_config.h)

In general, an architecture is going to be from on of these platforms:
  + AMD
  + ARM
  + IBM
  + Intel
  + Intel Xeon Phi
  + NVIDIA
Although not strictly necessary, it is helpful to keep things organized by
grouping by platform.

### generate_makefile.sh

The bash code does not do any error checking on the `--arch=`  or `--device=`
arguments thus strictly speaking you do not *need* to do anything to add a
device or architecture; however, you should add it to the help menu.  For the
archictectures, please group by one of the platforms listed above.


### cmake/kokkos_options.cmake and cmake/kokkos_settings.cmake

The options for the CMake build system are: `-DKOKKOS_HOST_ARCH:STRING=` and
`-DKOKKOS_ENABLE_<device>:BOOL=`.  Although any string can be passed into
KOKKOS_HOST_ARCH option, it is checked against an accepted list.  Likewise, the
KOKKOS_ENABLE_<device> must have the option added AND it is formed using the
list. Thus:
  + A new architecture should be added to the KOKKOS_HOST_ARCH_LIST variable.
  + A new device should be added to the KOKKOS_DEVICES_LIST variable **AND** a
    KOKKOS_ENABLE_<newdevice> option specified (see KOKKOS_ENABLE_CUDA for
    example).
  + A new device should be added to the KOKKOS_DEVICES_LIST variable **AND** a

The translation from option to the `KOKKOS_SETTINGS` is done in
`kokkos_settings.cmake`.  This translation is automated for some types if you ad
to the list, but for others, it may need to be hand coded.


### Makefile.kokkos

This is the main coding used by both the make and cmake system for defining
the sources (generated makefile and cmake snippets by `core/src/Makefile`), for
setting the defines in KokkosCore_config.h, and defining various internal
variables.  To understand how to add to this file, you should work closely with
the Kokkos development team.
