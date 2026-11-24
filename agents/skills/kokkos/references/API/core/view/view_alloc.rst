``view_alloc``
==============

.. role:: cpp(code)
   :language: cpp

Header File: ``<Kokkos_Core.hpp>``

Usage
-----

.. code-block:: cpp

    Kokkos::view_alloc(exec_space, Kokkos::WithoutInitializing, "ViewString");
    Kokkos::view_wrap(pointer_to_wrapping_memory);

Create View allocation parameter bundle from argument list. Valid argument list members are:

* label as ``C``-string or ``std::string``

* memory space instance of the ``View::memory_space`` type

* execution space instance able to access ``View::memory_space``

* ``Kokkos::WithoutInitializing`` to bypass element initialization and destruction

* ``Kokkos::SequentialHostInit`` to perform element initialization and destruction serially on host (since 4.4.01)

* ``Kokkos::AllowPadding`` to allow allocation to pad dimensions for memory alignment

* a pointer to create an unmanaged View wrapping that pointer


Description
-----------

.. cpp:function:: template <class... Args> ALLOC_PROP view_alloc(Args const&... args)

   Create View allocation parameter bundle from argument list.

   ``args`` : Cannot contain a pointer to memory.

.. cpp:function:: template <class... Args> ALLOC_PROP view_wrap(Args const&... args)

   Create View allocation parameter bundle from argument list.

   ``args`` : Can only be a pointer to memory.


.. cpp:type:: ALLOC_PROP

   :cpp:type:`ALLOC_PROP` is a special, unspellable implementation-defined type that is returned by :cpp:func:`view_alloc`
   and :cpp:func:`view_wrap`. It represents a bundle of allocator parameters, including the View label, the memory space instance,
   the execution space instance, whether to initialize the memory, whether to allow padding, and the raw pointer value (for wrapped unmanaged views).

.. cpp:type:: WithoutInitializing

   :cpp:type:`WithoutInitializing` is intended to be used in situations where default construction of `View` elements in its
   associated execution space is not needed or not viable. In particular, it may not be viable in situations such as the construction of objects with virtual functions,
   or for `Views` of elements without default constructor. In such situations, this option is often used in conjunction with manual in-place `new`
   construction of objects and manual destruction of elements.

.. cpp:type:: SequentialHostInit

   :cpp:type:`SequentialHostInit` is intended to be used to initialize elements that do not have a default constructor or destructor that
   can be called inside a Kokkos parallel region. In particular this includes constructors and destructors which:

   * allocate or deallocate memory
   * create or destroy managed `Kokkos::View` objects
   * call Kokkos parallel operations

   When using this allocation option the `View` constructor/destructor will create/destroy elements in a serial loop on the Host.

   .. warning::

     `SequentialHostInit` can only be used when creating host accessible `View`s, such as `View`s with `HostSpace`, `SharedSpace`,
     or `SharedHostPinnedSpace` as memory space.

   .. versionadded:: 4.4.01
