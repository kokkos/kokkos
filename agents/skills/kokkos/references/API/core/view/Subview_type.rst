``Subview``
===========

.. role:: cpp(code)
   :language: cpp

.. _subviewfunc: subview.rst

.. |subviewfunc| replace:: ``Kokkos::subview()``

Header File: ``Kokkos_Core.hpp``

Description
-----------

Alias template to deduce the type that is returned by a call to the |subviewfunc|_ function with given arguments.

Interface
---------

.. code-block:: cpp

   template <class ViewType, class... Args>
   using Subview = IMPL_DETAIL; // deduce subview type from source view traits

Type of the result of ``Kokkos::subview(ViewType view_arg, Args .... args)``

Requirements
------------

Requires:

- ``ViewType`` is a specialization of ``Kokkos::View``

- ``Args...`` are slice specifiers as defined in |subviewfunc|_.

- ``sizeof... (Args) == ViewType::rank()``.


Examples
--------

.. code-block:: cpp

   using view_type = Kokkos::View<double ***[5]>;
   view_type a("A",N0,N1,N2);

   struct subViewHolder {
   Kokkos::Subview<view_type,
                   std::pair<int,int>,
                   int,
                   decltype(Kokkos::ALL),
                   int> s;
   } subViewHolder;

   subViewHolder.s  = Kokkos::subview(a,
                                      std::pair<int,int>(3,15),
                                      5,
                                      Kokkos::ALL,
                                      3);
