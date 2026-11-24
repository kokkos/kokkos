``realloc``
===========

Header File: ``<Kokkos_Core.hpp>``

.. role:: cpp(code)
    :language: cpp

Usage
-----

.. code-block:: cpp

    realloc(view,n0,n1,n2,n3);
    realloc(view,layout);

Reallocates a view to have the new dimensions. Can grow or shrink, and will not preserve content.

Description
-----------

.. code-block:: cpp

   template <class T, class... P>
   void realloc(View<T, P...>& v,
		const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

* Resizes ``v`` to have the new dimensions without preserving its contents.

  - ``v``: existing view, can be a default constructed one.

  - ``n[X]``: new length for extent X.

* Restrictions: ``View<T, P...>::array_layout`` is either ``LayoutLeft`` or ``LayoutRight``.

.. code-block:: cpp

   template <class I, class T, class... P>
   void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
		const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

* Resizes ``v`` to have the new dimensions without preserving its contents.
  The new ``Kokkos::View`` is constructed using the View constructor property ``arg_prop``, e.g., Kokkos::WithoutInitializing.

  - ``v``: existing view, can be a default constructed one.

  - ``n[X]``: new length for extent X.

  - ``arg_prop``: View constructor property, e.g., ``Kokkos::WithoutInitializing``.


* Restrictions: ``View<T, P...>::array_layout`` is either ``LayoutLeft`` or ``LayoutRight``.

.. code-block:: cpp

   template <class ALLOC_PROP, class T, class... P>
   void realloc(const ALLOC_PROP& arg_prop,
		Kokkos::View<T, P...>& v,
		const size_t n0 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n1 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n2 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n3 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n4 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n5 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n6 = KOKKOS_IMPL_CTOR_DEFAULT_ARG,
		const size_t n7 = KOKKOS_IMPL_CTOR_DEFAULT_ARG);

* Resizes ``v`` to have the new dimensions without preserving its contents.
  The new ``Kokkos::View`` is constructed using the View constructor properties ``arg_prop``, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.

  - ``v``: existing view, can be a default constructed one.

  - ``n[X]``: new length for extent X.

  - ``arg_prop``: View constructor properties, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.


* Restrictions:

  - ``View<T, P...>::array_layout`` is either ``LayoutLeft`` or ``LayoutRight``.

  - ``arg_prop`` must not include a pointer to memory, a label, or a memory space.

.. code-block:: cpp

   template <class T, class... P>
   void realloc(Kokkos::View<T, P...>& v,
		const typename Kokkos::View<T, P...>::array_layout& layout);

* Resizes ``v`` to have the new dimensions without preserving its contents.

  - ``v``: existing view, can be a default constructed one.

  - ``layout``: a layout instance containing the new dimensions.

.. code-block:: cpp

   template <class I, class T, class... P>
   void realloc(const I& arg_prop, Kokkos::View<T, P...>& v,
                const typename Kokkos::View<T, P...>::array_layout& layout);

* Resizes ``v`` to have the new dimensions without preserving its contents.
  The new ``Kokkos::View`` is constructed using the View constructor property ``arg_prop``, e.g., Kokkos::WithoutInitializing.

  - ``v``: existing view, can be a default constructed one.

  - ``layout``: a layout instance containing the new dimensions.

  - ``arg_prop``: View constructor property, e.g., ``Kokkos::WithoutInitializing``.

.. code-block:: cpp

   template <class ALLOC_PROP, class T, class... P>
   void realloc(const ALLOC_PROP& arg_prop,
                Kokkos::View<T, P...>& v,
		const typename Kokkos::View<T, P...>::array_layout& layout);

* Resizes ``v`` to have the new dimensions without preserving its contents.
  The new ``Kokkos::View`` is constructed using the View constructor properties ``arg_prop``, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.

  - ``v``: existing view, can be a default constructed one.

  - ``layout``: a layout instance containing the new dimensions.

  - ``arg_prop``: View constructor properties, e.g., ``Kokkos::view_alloc(Kokkos::WithoutInitializing)``.

* Restrictions: ``arg_prop`` must not include a pointer to memory, a label, or a memory space.

Example
-------

.. code-block:: cpp

    Kokkos::realloc(v, 2, 3);

* Reallocate a ``Kokkos::View`` with dynamic rank 2 to have dynamic extent 2 and 3 respectively.

.. code-block:: cpp

    Kokkos::realloc(Kokkos::WithoutInitializing, v, 2, 3);

* Reallocate a ``Kokkos::View`` with dynamic rank 2 to have dynamic extent 2 and 3 respectively. After this call, the View is uninitialized.
