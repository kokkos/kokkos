STL Compatibility Issues
========================

Kokkos developers strive to implement the Kokkos macros in a manner compatible with the latest versions of the
C++ language and with the C++ Standard Library (STL). However deviations from this approach occur when necessary,
as is the case for the following STL classes ``(std::*)``. The STL class does not work properly on GPUs so parallel Kokkos
classes have been developed. This section documents the specific deviations and provides usage guidance for developers.
Select the links below to see the details.

.. toctree::
   :maxdepth: 1

   ./stl-compat/Array
   ./stl-compat/pair
