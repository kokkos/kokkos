//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_BIT_MANIPULATION_HPP
#define KOKKOS_BIT_MANIPULATION_HPP

#include <Kokkos_Macros.hpp>
#include <Kokkos_NumericTraits.hpp>

namespace Kokkos::Impl {

template <class T>
KOKKOS_FUNCTION constexpr int countl_zero_fallback(T x) {
  // From Hacker's Delight (2nd edition) section 5-3
  unsigned int y = 0;
  using ::Kokkos::Experimental::digits_v;
  int n = digits_v<T>;
  int c = digits_v<T> / 2;
  do {
    y = x >> c;
    if (y != 0) {
      n -= c;
      x = y;
    }
    c >>= 1;
  } while (c != 0);
  return n - static_cast<int>(x);
}

template <class T>
KOKKOS_FUNCTION constexpr int countr_zero_fallback(T x) {
  using ::Kokkos::Experimental::digits_v;
  return digits_v<T> - countl_zero_fallback(static_cast<T>(
                           static_cast<T>(~x) & static_cast<T>(x - 1)));
}

template <class T>
KOKKOS_FUNCTION constexpr int popcount_fallback(T x) {
  int c = 0;
  for (; x != 0; x &= x - 1) {
    ++c;
  }
  return c;
}

template <class T>
inline constexpr bool is_standard_unsigned_integer_type_v =
    std::is_same_v<T, unsigned char> || std::is_same_v<T, unsigned short> ||
    std::is_same_v<T, unsigned int> || std::is_same_v<T, unsigned long> ||
    std::is_same_v<T, unsigned long long>;

}  // namespace Kokkos::Impl

namespace Kokkos {

//<editor-fold desc="[bit.count], counting">
template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countl_zero(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
  // TODO use compiler intrinsics when available
  return Impl::countl_zero_fallback(x);
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countl_one(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  using ::Kokkos::Experimental::finite_max_v;
  if (x == finite_max_v<T>) return digits_v<T>;
  return countl_zero(static_cast<T>(~x));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countr_zero(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  if (x == 0) return digits_v<T>;
  // TODO use compiler intrinsics when available
  return Impl::countr_zero_fallback(x);
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
countr_one(T x) noexcept {
  using ::Kokkos::Experimental::digits_v;
  using ::Kokkos::Experimental::finite_max_v;
  if (x == finite_max_v<T>) return digits_v<T>;
  return countr_zero(static_cast<T>(~x));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, int>
popcount(T x) noexcept {
  if (x == 0) return 0;
  // TODO use compiler intrinsics when available
  return Impl::popcount_fallback(x);
}
//</editor-fold>

//<editor-fold desc="[bit.pow.two], integral powers of 2">
template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, bool>
has_single_bit(T x) noexcept {
  return x != 0 && (((x & (x - 1)) == 0));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, T>
bit_ceil(T x) noexcept {
  if (x <= 1) return 1;
  using ::Kokkos::Experimental::digits_v;
  return T{1} << (digits_v<T> - countl_zero(static_cast<T>(x - 1)));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, T>
bit_floor(T x) noexcept {
  if (x == 0) return 0;
  using ::Kokkos::Experimental::digits_v;
  return T{1} << (digits_v<T> - 1 - countl_zero(x));
}

template <class T>
KOKKOS_FUNCTION constexpr std::enable_if_t<
    Impl::is_standard_unsigned_integer_type_v<T>, T>
bit_width(T x) noexcept {
  if (x == 0) return 0;
  using ::Kokkos::Experimental::digits_v;
  return digits_v<T> - countl_zero(x);
}
//</editor-fold>

}  // namespace Kokkos

#endif
