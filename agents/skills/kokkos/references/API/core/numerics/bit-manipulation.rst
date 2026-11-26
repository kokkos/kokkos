Bit manipulation
================

.. role::cpp(code)
    :language: cpp

.. role:: strike
    :class: strike

.. _KokkosBitManipulation: https://github.com/kokkos/kokkos/blob/4.1.00/core/src/Kokkos_BitManipulation.hpp

.. |KokkosBitManipulation| replace:: ``<Kokkos_BitManipulation.hpp>``

.. _StandardLibraryHeaderBit: https://en.cppreference.com/w/cpp/header/bit

.. |StandardLibraryHeaderBit| replace:: ``<bit>``

Defined in header |KokkosBitManipulation|_ which is included from ``<Kokkos_Core.hpp>``

Provides function templates from the standard library header |StandardLibraryHeaderBit|_ (since C++20).

Bit manipulation function templates are defined in the ``Kokkos::`` namespace since Kokkos 4.1

.. _bit_cast: https://en.cppreference.com/w/cpp/numeric/bit_cast

.. |bit_cast| replace:: ``bit_cast``

.. _byteswap: https://en.cppreference.com/w/cpp/numeric/byteswap

.. |byteswap| replace:: ``byteswap``

.. _has_single_bit: https://en.cppreference.com/w/cpp/numeric/has_single_bit

.. |has_single_bit| replace:: ``has_single_bit``

.. _bit_ceil: https://en.cppreference.com/w/cpp/numeric/bit_ceil

.. |bit_ceil| replace:: ``bit_ceil``

.. _bit_floor: https://en.cppreference.com/w/cpp/numeric/bit_floor

.. |bit_floor| replace:: ``bit_floor``

.. _bit_width: https://en.cppreference.com/w/cpp/numeric/bit_width

.. |bit_width| replace:: ``bit_width``

.. _rotl: https://en.cppreference.com/w/cpp/numeric/rotl

.. |rotl| replace:: ``rotl``

.. _rotr: https://en.cppreference.com/w/cpp/numeric/rotr

.. |rotr| replace:: ``rotr``

.. _countl_zero: https://en.cppreference.com/w/cpp/numeric/countl_zero

.. |countl_zero| replace:: ``countl_zero``

.. _countl_one: https://en.cppreference.com/w/cpp/numeric/countl_one

.. |countl_one| replace:: ``countl_one``

.. _countr_zero: https://en.cppreference.com/w/cpp/numeric/countr_zero

.. |countr_zero| replace:: ``countr_zero``

.. _countr_one: https://en.cppreference.com/w/cpp/numeric/countr_one

.. |countr_one| replace:: ``countr_one``

.. _popcount: https://en.cppreference.com/w/cpp/numeric/popcount

.. |popcount| replace:: ``popcount``

================== ============================================================
|bit_cast|_        reinterpret the object representation of one type as that of another (see note below)
|byteswap|_        reverses the bytes in the given integer value 
|has_single_bit|_  checks if a number is an integral power of two 
|bit_ceil|_        finds the smallest integral power of two not less than the given value
|bit_floor|_       finds the largest integral power of two not greater than the given value
|bit_width|_       finds the smallest number of bits needed to represent the given value
|rotl|_            computes the result of bitwise left-rotation
|rotr|_            computes the result of bitwise right-rotation
|countl_zero|_     counts the number of consecutive 0 bits, starting from the most significant bit
|countl_one|_      counts the number of consecutive 1 bits, starting from the most significant bit
|countr_zero|_     counts the number of consecutive 0 bits, starting from the least significant bit
|countr_one|_      counts the number of consecutive 1 bits, starting from the least significant bit
|popcount|_        counts the number of 1 bits in an unsigned integer
================== ============================================================

----

Notes
-----

* For all the above template functions, a non-``constexpr`` counterpart ending
  with the ``*_builtin`` suffix is provided in the ``Kokkos::Experimental::``
  namespace to make up for some compiler intrinsics that cannot appear in
  constant expressions.
* In contrast to its counterpart in the C++ standard library,
  ``Kokkos::bit_cast`` is not usable in constant expressions (not a
  ``constexpr`` function) as it is not implementable as a library facility
  and requires compiler magic which is not available to us.
