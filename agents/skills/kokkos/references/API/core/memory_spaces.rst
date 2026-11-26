.. _api-memory-spaces:

Memory Spaces
=============

.. role:: cpp(code)
    :language: cpp

.. _MemorySpaceType: #kokkos-memoryspaceconcept

.. |MemorySpaceType| replace:: :cpp:func:`MemorySpace` type

.. _TheDocumentationOnTheMemorySpaceConcept: #kokkos-memoryspaceconcept

.. |TheDocumentationOnTheMemorySpaceConcept| replace:: the documentation on the :cpp:func:`MemorySpace` concept

.. _Experimental: utilities/experimental.md#experimentalnamespace

.. |Experimental| replace:: Experimental

.. _ExecutionSpaceType: ./execution_spaces.rst#kokkos-executionspaceconcept

.. |ExecutionSpaceType| replace:: :cpp:func:`ExecutionSpace` type

.. _ExecutionSpaceTypes: ./execution_spaces.rst#kokkos-executionspaceconcept

.. |ExecutionSpaceTypes| replace:: :cpp:func:`ExecutionSpace` types

``Kokkos::CudaSpace``
---------------------

``Kokkos::CudaSpace`` is a |MemorySpaceType|_ representing device memory on a Cuda-capable GPU. Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::CudaHostPinnedSpace``
-------------------------------

``Kokkos::CudaHostPinnedSpace`` is a |MemorySpaceType|_ representing host-side pinned memory accessible from a Cuda-capable GPU.  This memory is typically accessible by both host and device execution spaces.  Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::CudaUVMSpace``
------------------------

``Kokkos::CudaUVMSpace`` is a |MemorySpaceType|_ representing unified virtual memory on a Cuda-capable GPU system.  Unified virtual memory is also accessible from most host execution spaces.  Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::HIPSpace``
--------------------

``Kokkos::HIPSpace`` :sup:`promoted from` |Experimental|_ :sup:`since 4.0` is a |MemorySpaceType|_ representing device memory on a GPU in the HIP GPU programming environment.  Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::HIPHostPinnedSpace``
------------------------------

``Kokkos::HIPHostPinnedSpace`` :sup:`promoted from` |Experimental|_ :sup:`since 4.0` is a |MemorySpaceType|_ representing host-side pinned memory accessible from a GPU in the HIP GPU programming environment.  This memory is accessible by both host and device execution spaces.  Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::HIPManagedSpace``
---------------------------

``Kokkos::HIPManagedSpace`` :sup:`promoted from` |Experimental|_ :sup:`since 4.0`  is a |MemorySpaceType|_ representing page-migrating memory on a GPU in the HIP GPU programming environment.  Page-migrating memory is accessible from most host execution spaces. Even though available with all combinations of operating system and HIP-supported hardware, it requires both operating system and hardware to support and enable the ``xnack`` feature. Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::SYCLDeviceUSMSpace``
--------------------------------------------

``Kokkos::SYCLDeviceUSMSpace`` :sup:`promoted from` |Experimental|_ :sup:`since 4.5` is a |MemorySpaceType|_ representing device memory on a GPU in the SYCL GPU programming environment. This memory is only accessible from the SYCL execution space.

``Kokkos::SYCLHostUSMSpace``
------------------------------------------

``Kokkos::SYCLHostUSMSpace`` :sup:`promoted from` |Experimental|_ :sup:`since 4.5` is a |MemorySpaceType|_ representing host-side pinned memory accessible from a GPU in the SYCL GPU programming environment. This memory is accessible from both host and SYCL execution spaces.

``Kokkos::SYCLSharedUSMSpace``
--------------------------------------------

``Kokkos::SYCLSharedUSMSpace`` :sup:`promoted from` |Experimental|_ :sup:`since 4.5` is a |MemorySpaceType|_ representing page-migrating memory on a GPU in the SYCL GPU programming environment. This memory is accessible from both host and SYCL execution spaces.

``Kokkos::HostSpace``
---------------------

``Kokkos::HostSpace`` is a |MemorySpaceType|_ representing traditional random access memory accessible from the CPU.  Except in rare instances, it should not be used directly, but instead should be used generically as a memory space.  For details, see |TheDocumentationOnTheMemorySpaceConcept|_.

``Kokkos::SharedSpace``
-----------------------

``Kokkos::SharedSpace`` :sup:`since 4.0` is a |MemorySpaceType|_ alias representing memory that can be accessed by any enabled |ExecutionSpaceType|_. To achieve this, the memory can be moved to and from the local memory of the processing units represented by the ``ExecutionSpaces``. The movement is done automatically by the OS and driver at the moment of access. If not currently located in the local memory of the accessing processing unit, the memory is moved in chunks (size is backend dependent). These chunks can be moved independently (e.g. only the part that is accessed on the GPU is moved to the GPU) and are treated like local memory while residing on the processing unit. For details, see |TheDocumentationOnTheMemorySpaceConcept|_.
Availability can be checked with the preprocessor define ``KOKKOS_HAS_SHARED_SPACE`` or the ``constexpr bool Kokkos::has_shared_space``.
For the following backends ``Kokkos::SharedSpace`` is pointing to the corresponding |MemorySpaceType|_:

* Cuda -> ``CudaUVMSpace``
* HIP -> ``HIPManagedSpace``
* SYCL -> ``SYCLSharedUSMSpace``
* Only backends running on host -> ``HostSpace``

``Kokkos::SharedHostPinnedSpace``
---------------------------------

``Kokkos::SharedHostPinnedSpace`` :sup:`since 4.0` is a |MemorySpaceType|_ alias which is accessible by all enabled |ExecutionSpaceTypes|_. The memory stays pinned on the host and is available on the device via zero copy access in small chunks (cache lines, memory pages, etc. depending on the backend). Writes to the memory in one ``ExecutionSpace`` become visible in other ``ExecutionSpaces`` at synchronization events. Which events trigger a synchronization depend on the backend specifics. Nevertheless, fences are synchronization events on all backends.
Availability can be checked with the preprocessor define ``KOKKOS_HAS_SHARED_HOST_PINNED_SPACE`` or the ``constexpr bool Kokkos::has_shared_host_pinned_space``.
For the following backends ``Kokkos::SharedHostPinnedSpace`` is pointing to the corresponding |MemorySpaceType|_:

* Cuda -> ``CudaHostPinnedSpace``
* HIP -> ``HipHostPinnedSpace``
* SYCL -> ``SYCLHostUSMSpace``
* Only backends running on host -> ``HostSpace``

``Kokkos::MemorySpaceConcept``
------------------------------

The concept of a ``MemorySpace`` is the fundamental abstraction to represent the "where" and the "how" that memory allocation and access takes place in Kokkos. Most code that uses Kokkos should be written to the *generic concept* of a ``MemorySpace`` rather than any specific instance. This page talks practically about how to *use* the common features of memory spaces in Kokkos; for a more formal and theoretical treatment, see `this document <KokkosConcepts.rst>`_.

    *Disclaimer*: There is nothing new about the term "concept" in C++; anyone who has ever used templates in C++ has used concepts whether they knew it or not. Please do not be confused by the word "concept" itself, which is now more often associated with a shiny new C++20 language feature. Here, "concept" just means "what you're allowed to do with a type that is a template parameter in certain places".

Synopsis
~~~~~~~~

.. code-block:: cpp

    // This is not an actual class, it just describes the concept in shorthand
    class MemorySpaceConcept {
    public:
        typedef MemorySpaceConcept memory_space;
        typedef ... execution_space;
        typedef Device<execution_space, memory_space> device_type;

        MemorySpaceConcept();
        MemorySpaceConcept(const MemorySpaceConcept& src);
        const char* name() const;
        void * allocate(ptrdiff_t size) const;
        void deallocate(void* ptr, ptrdiff_t size) const;
    };

    template<class MS>
    struct is_memory_space {
    enum { value = false };
    };

    template<>
    struct is_memory_space<MemorySpaceConcept> {
    enum { value = true };
    };  

Typedefs
~~~~~~~~

.. _ExecutionSpace: execution_spaces.rst#executionspaceconcept

.. |ExecutionSpace| replace:: :cpp:func:`ExecutionSpace`

.. _DeepCopyDocumentation: view/deep_copy.rst

.. |DeepCopyDocumentation| replace:: :cpp:func:`deep_copy` documentation

.. _KokkosSpaceAccessibility: SpaceAccessibility.rst

.. |KokkosSpaceAccessibility| replace:: :cpp:func:`Kokkos::SpaceAccessibility`

* ``memory_space``: The self type;
* ``execution_space``: the default |ExecutionSpace|_ to use when constructing objects in memory provided by an instance of ``MemorySpace``, or (potentially) when deep copying from or to such memory (see |DeepCopyDocumentation|_ for details). Kokkos guarantees that ``Kokkos::SpaceAccessibility<execution_space, memory_space>::accessible`` will be ``true`` (see |KokkosSpaceAccessibility|_).
* ``device_type``: ``DeviceType<execution_space,memory_space>``.

Constructors
~~~~~~~~~~~~

* ``MemorySpaceConcept()``: Default constructor.
* ``MemorySpaceConcept(const MemorySpaceConcept& src)``: Copy constructor.

Functions
~~~~~~~~~

* ``const char* name() const;``: Returns the label of the memory space instance.
* ``void * allocate(ptrdiff_t size) const;``: Allocates a buffer of at least ``size`` bytes using the memory resource that ``MemorySpaceConcept`` represents.
* ``void deallocate(void* ptr, ptrdiff_t size) const;``: Frees the buffer starting at ``ptr`` (of type ``void*``) previously allocated with exactly ``allocate(size)``.

Non Member Facilities
~~~~~~~~~~~~~~~~~~~~~

* ``template<class MS> struct is_memory_space;``: typetrait to check whether a class is a memory space.
* ``template<class S1, class S2> struct SpaceAccessibility;``: typetraits to check whether two spaces are compatible (assignable, deep_copy-able, accessible). 
