``resize``
==========

.. role:: cpp(code)
   :language: cpp

Header File: <Kokkos_Core.hpp>

Usage
-----

.. code-block:: cpp

    resize(view,n0,n1,n2,n3);
    resize(view,layout);

Reallocates a view to have the new dimensions. Can grow or shrink, and will preserve content of the common subextents.


Description
-----------

* .. code-block:: cpp

     template <class T, class... P>
     void resize(View<T, P...>& v,
		 const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  Resizes ``v`` to have the new dimensions while preserving the contents for the common subview of the old and new view.

  * ``v``: existing view, can be a default constructed one.
  * ``n[X]``: new length for extent X.

  Restrictions:

  * ``View<T, P...>::array_layout`` is either ``LayoutLeft`` or ``LayoutRight``.

* .. code-block:: cpp

      template <class I, class T, class... P>
      void resize(const I& arg_prop, Kokkos::View<T, P...>& v,
		  const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		  const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  Resizes ``v`` to have the new dimensions while preserving the contents for the common
  subview of the old and new view. The new ``Kokkos::View`` is constructed using the
  View constructor property ``arg_prop``, e.g., Kokkos::WithoutInitializing.

  * ``v``: existing view, can be a default constructed one.

  * ``n[X]``: new length for extent X.

  * ``arg_prop``: View constructor property, e.g., ``Kokkos::WithoutInitializing``.

  Restrictions:

  * ``View<T, P...>::array_layout`` is either ``LayoutLeft` or `LayoutRight``.

* .. code-block:: cpp

     template <class T, class... P, class ALLOC_PROP>
     void resize(const ALLOC_PROP& arg_prop,
		 Kokkos::View<T, P...>& v,
		 const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		 const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

  Resizes ``v`` to have the new dimensions while preserving the contents for the common
  subview of the old and new view. The new ``Kokkos::View`` is constructed using the View constructor
  properties ``arg_prop``, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.
  If ``arg_prop`` includes an execution space, it is used for allocating memory and for copying elements without using a final fence.

  * ``v``: existing view, can be a default constructed one.
  * ``n[X]``: new length for extent X.
  * ``arg_prop``: View constructor properties, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.

  Restrictions:

  * ``View<T, P...>::array_layout`` is either ``LayoutLeft`` or ``LayoutRight``.
  * ``arg_prop`` must not include a pointer to memory, a label, or a memory space.

* .. code-block:: cpp

     template <class T, class... P>
     void resize(Kokkos::View<T, P...>& v,
                 const typename Kokkos::View<T, P...>::array_layout& layout);

  Resizes ``v`` to have the new dimensions while preserving the contents for the common subview of the old and new view.

  * ``v``: existing view, can be a default constructed one.
  * ``layout``: a layout instance containing the new dimensions.

* .. code-block:: cpp

     template <class T, class... P>
     void resize(const I& arg_prop, Kokkos::View<T, P...>& v,
	         const typename Kokkos::View<T, P...>::array_layout& layout);

  Resizes ``v`` to have the new dimensions while preserving the contents for the common subview
  of the old and new view. The new ``Kokkos::View`` is constructed using the View constructor
  property ``arg_prop``, e.g., Kokkos::WithoutInitializing.

  * ``v``: existing view, can be a default constructed one.
  * ``layout``: a layout instance containing the new dimensions.
  * ``arg_prop``: View constructor property, e.g., ``Kokkos::WithoutInitializing``.

* .. code-block:: cpp

     template <class T, class... P, class ALLOC_PROP>
     void resize(const ALLOC_PROP& arg_prop,
	         Kokkos::View<T, P...>& v,
	         const typename Kokkos::View<T, P...>::array_layout& layout);

  Resizes ``v`` to have the new dimensions while preserving the contents for the
  common subview of the old and new view. The new ``Kokkos::View`` is constructed using
  the View constructor properties ``arg_prop``, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.
  If ``arg_prop`` includes an execution space, it is used for allocating memory and for copying elements without using a final fence.

  * ``v``: existing view, can be a default constructed one.
  * ``layout``: a layout instance containing the new dimensions.
  * ``arg_prop``: View constructor properties, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.

  Restrictions:

  * ``arg_prop`` must not include a pointer to memory, a label, or a memory space.

Example:
--------

* .. code-block:: cpp

    Kokkos::resize(v, 2, 3);

Resize a ``Kokkos::View`` with dynamic rank 2 to have dynamic extent 2 and 3 respectively preserving previous content.

* .. code-block:: cpp

    Kokkos::resize(Kokkos::WithoutInitializing, v, 2, 3);

Resize a ``Kokkos::View`` with dynamic rank 2 to have dynamic extent 2 and 3 respectively preserving previous content. After this call, the new content is uninitialized.
