``kokkos_realloc``
==================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Core.hpp>``

.. _Kokkos_kokkos_malloc: malloc.rst

.. |Kokkos_kokkos_malloc| replace:: ``Kokkos::kokkos_malloc()``

.. _Kokkos_kokkos_realloc: realloc.rst

.. |Kokkos_kokkos_realloc| replace:: ``Kokkos::kokkos_realloc()``

.. _MemorySpace: ../memory_spaces.rst

.. |MemorySpace| replace:: ``MemorySpace``

.. _Kokkos_kokkos_free: free.rst

.. |Kokkos_kokkos_free| replace:: ``Kokkos::kokkos_free()``

Reallocates the given area of memory. It must be previously allocated by |Kokkos_kokkos_malloc|_ or |Kokkos_kokkos_realloc|_
on the same memory space |MemorySpace|_ and not yet freed with |Kokkos_kokkos_free|_, otherwise, the results are undefined.

.. warning::

   Calling any function that manipulates the behavior of the memory (e.g. ``memAdvise``)
   on memory managed by ``Kokkos`` results in undefined behavior.

Description
-----------

.. cpp:function:: template <class MemorySpace = Kokkos::DefaultExecutionSpace::memory_space> void* kokkos_realloc(void* ptr, size_t new_size);

  :tparam MemorySpace: Controls the storage location. If omitted the memory space of the default execution space is used (i.e. ``Kokkos::DefaultExecutionSpace::memory_space``).

  :param ptr: The pointer to the memory area to be reallocated.

  :param new_size: The new size in bytes.

  :returns: On success, returns a pointer to the beginning of the newly allocated memory. To avoid a memory leak, the returned pointer must be deallocated with |Kokkos_kokkos_free|_, the original pointer ``ptr`` is invalidated and any access to it is undefined behavior (even if reallocation was in-place). On failure, returns a null pointer. The original pointer ptr remains valid and may need to be deallocated with |Kokkos_kokkos_free|_.

  :throws: On failure, throws ``Kokkos::Experimental::RawMemoryAllocationFailure``.
