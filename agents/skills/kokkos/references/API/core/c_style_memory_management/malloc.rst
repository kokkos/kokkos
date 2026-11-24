``kokkos_malloc``
=================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. _MemorySpace: ../memory_spaces.rst

.. |MemorySpace| replace:: ``MemorySpace``

.. _Kokkos_kokkos_free: free.rst

.. |Kokkos_kokkos_free| replace:: ``Kokkos::kokkos_free()``

.. _Kokkos_realloc: realloc.rst

.. |Kokkos_realloc| replace:: ``Kokkos::kokkos_realloc()``

Allocate ``size`` bytes of uninitialized storage on the specified memory space |MemorySpace|_ plus some extra space for metadata such as the label.

If allocation succeeds, returns a pointer to the lowest (first) byte in the allocated memory block that is suitably aligned for any scalar type.

If allocation fails, an exception of type ``Kokkos::Experimental::RawMemoryAllocationFailure`` is thrown.

.. warning::

    Calling any function that manipulates the behavior of the memory (e.g. ``memAdvise``) on memory managed by ``Kokkos`` results in undefined behavior.

Description
-----------

.. cpp:function:: template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space> void* kokkos_malloc(const string& label, size_t size);

.. cpp:function:: template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space> void* kokkos_malloc(size_t size);

    :tparam MemorySpace: Controls the storage location. If omitted the memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``).

    :param label: A user provided string which is used in profiling and debugging tools via the KokkosP Profiling Tools.

    :param size: The number of bytes to allocate.

    :returns: On success, returns the pointer to the beginning of newly allocated memory. To avoid a memory leak, the returned pointer must be deallocated with |Kokkos_kokkos_free|_ or |Kokkos_realloc|_.

    :throws: On failure, throws ``Kokkos::Experimental::RawMemoryAllocationFailure``.
