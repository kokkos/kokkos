``Kokkos::kokkos_swap``
=======================

.. role:: cpp(code)
    :language: cpp

Defined in header ``<Kokkos_Swap.hpp>``:sup:`(since 4.3)` which is included from ``<Kokkos_Core.hpp>``

.. code-block:: cpp

    template <class T>
    KOKKOS_FUNCTION constexpr void
    kokkos_swap(T& a, T& b) noexcept(std::is_nothrow_move_constructible_v<T> &&
                                     std::is_nothrow_move_assignable_v<T>);  // (1) (since 4.3)

    template <class T2, std::size_t N>
    KOKKOS_FUNCTION constexpr void
    kokkos_swap(T2 (&a)[N], T2 (&b)[N]) noexcept(noexcept(*a, *b));  // (2) (since 4.3)


1) Swaps the values ``a`` and ``b``. This overload does not participate in overload
   resolution unless ``std::is_move_constructible_v<T> && std::is_move_assignable_v<T>``
   is ``true``.

2) Swaps the arrays ``a`` and ``b``. This overload does not participate in
   overload resolution unless ``T2`` is swappable.

Notes
-----
.. _std_swap: https://en.cppreference.com/w/cpp/algorithm/swap

.. |std_swap| replace:: ``std::swap``

``kokkos_swap`` provides the same functionality as |std_swap|_.  It just
cannot be called ``swap`` or it would yield some ambiguities in overload
resolution in some situations because of `ADL
<https://en.cppreference.com/w/cpp/language/adl>`_.
