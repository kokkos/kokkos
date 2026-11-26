``MemoryTraits``
================

:cpp:struct:`MemoryTraits` is the last template parameter of :cpp:class:`View`.


Usage
-----

.. code-block:: cpp

   using DefaultMT = Kokkos::MemoryTraits<>;
   using UnmanagedMT = Kokkos::MemoryTraits<Kokkos::Unmanaged>;
   using AtomicRandomAccessMT =
       Kokkos::MemoryTraits<Kokkos::Atomic | Kokkos::RandomAccess>;

Struct Interface
----------------

.. cpp:struct:: template <unsigned N> MemoryTraits

  When provided to a multidimensional View, ``MemoryTraits`` allow passing extra information about the treatment of the allocation. The template argument is expected to be a bitwise OR of enumeration values described below.

  .. versionchanged:: 4.7
    ``0`` was added as the default value for the template parameter ``N``.

    .. code-block:: cpp

      template <unsigned N = 0>
      struct MemoryTraits;

.. rubric:: Nested type

.. cpp:type::  memory_traits

  A tag type signifying the memory access trait(s) denoted by ``N``.

.. rubric:: Member Variables

.. cpp:member::  static constexpr bool is_unmanaged

  A boolean that indicates whether the Unmanaged trait is enabled.

.. cpp:member::  static constexpr bool is_random_access

  A boolean that indicates whether the RandomAccess trait is enabled.

.. cpp:member::  static constexpr bool is_atomic

  A boolean that indicates whether the Atomic trait is enabled.

.. cpp:member::  static constexpr bool is_restrict

  A boolean that indicates whether the Restrict trait is enabled.

.. cpp:member::  static constexpr bool is_aligned

  A boolean that indicates whether the Aligned trait is enabled.

.. _MemoryAccessTraits: ../../../ProgrammingGuide/View.rst#memory-access-traits

.. |MemoryAccessTraits| replace:: memory access traits

.. _UnmanagedViews: ../../../ProgrammingGuide/View.rst#unmanaged-views

.. |UnmanagedViews| replace:: unmanaged views

Non-Member Enums
^^^^^^^^^^^^^^^^

The following enumeration values are used to specify the memory access traits. Check the sub-section on |MemoryAccessTraits|_ in the Programming Guide for further information about how these traits can be used in practice.

.. cpp:enum:: MemoryTraitsFlags

  The following enumeration values are defined in this enumeration type.

.. cpp:enumerator:: Unmanaged

  This traits means that Kokkos does neither reference counting nor automatic deallocation for such Views. This trait can be associated with memory allocated in any memory space. For example, an *unmanaged view* can be created by wrapping raw pointers of allocated memory, while also specifying the execution or memory space accordingly.

.. cpp:enumerator:: RandomAccess

  Views that are going to be accessed irregularly (e.g., non-sequentially) can be declared as random access. 

.. cpp:enumerator:: Atomic

  In such a view, every access (read or write) to any element will be atomic. 

.. cpp:enumerator:: Restrict

  This trait indicates that the memory of this view doesn't alias/overlap with another data structure in the current scope. 

.. cpp:enumerator:: Aligned

  This trait provides additional information to the compiler that the memory allocation in this ``View`` has an alignment of 64. 

Non-Member Type aliases
^^^^^^^^^^^^^^^^^^^^^^^

The following type aliases are also available in the ``Kokkos`` namespace.

.. cpp:type:: MemoryManaged = MemoryTraits<0>;

  .. deprecated:: 4.7
    
    The ``MemoryManaged`` alias is deprecated.  Use ``MemoryTraits<>`` instead.
    Note that prior Kokkos versions require an explicit ``0`` template
    argument.

     
.. cpp:type:: MemoryUnmanaged = MemoryTraits<Unmanaged>;
.. cpp:type:: MemoryRandomAccess = MemoryTraits<Unmanaged | RandomAccess>;

  .. versionchanged:: 4.7
    ``MemoryRandomAccess`` was changed to ``MemoryTraits<RandomAccess>`` and does
    not imply ``Unmanaged`` any more.

Example
^^^^^^^

.. code-block:: cpp

   Kokkos::View<DayaType,
                LayoutType,
                MemorySpace,
                Kokkos::MemoryTraits<SomeFlag | SomeOtherFlag>> my_view;
