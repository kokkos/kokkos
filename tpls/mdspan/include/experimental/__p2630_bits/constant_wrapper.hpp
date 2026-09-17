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

#pragma once

#include "../__p0009_bits/utility.hpp"
#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

// ============================================================
// constant_wrapper, cw, increment, is_constant_wrapper
// ============================================================

#if defined(__cpp_lib_constant_wrapper)

using std::constant_wrapper;
using std::cw;

template<class T>
constexpr bool is_constant_wrapper = false;

template<class T>
constexpr bool is_constant_wrapper<const T> = is_constant_wrapper<T>;

template<auto Value>
constexpr bool is_constant_wrapper<constant_wrapper<Value>> = true;

#else // back-port: constant_wrapper = detail::integral_constant

template<auto Value, class T = decltype(Value)>
using constant_wrapper = integral_constant<T, Value>;

template<auto Value>
  constexpr auto cw = constant_wrapper<Value>{};

template<class T>
constexpr bool is_constant_wrapper = false;

template<class T>
constexpr bool is_constant_wrapper<const T> = is_constant_wrapper<T>;

// integral_constant is the underlying type of the back-port constant_wrapper
// (alias templates can't be used in partial specialization patterns)
template<class Type, Type Value>
constexpr bool is_constant_wrapper<integral_constant<Type, Value>> = true;

#endif // __cpp_lib_constant_wrapper

// ============================================================
// increment function for constant wrapper
// ============================================================

template<auto Value>
MDSPAN_INLINE_FUNCTION
constexpr auto
increment([[maybe_unused]] constant_wrapper<Value> x) {
  using value_type = typename decltype(x)::value_type;
  return cw< value_type(Value) + value_type(1) >;
}


// ============================================================
// Generic divide / multiply (scalar fall-through)
// ============================================================

template <class IndexT, class T0, class T1>
MDSPAN_INLINE_FUNCTION
constexpr auto divide(const T0 &v0, const T1 &v1) {
  return IndexT(v0) / IndexT(v1);
}

template <class IndexT, class T0, class T1>
MDSPAN_INLINE_FUNCTION
constexpr auto multiply(const T0 &v0, const T1 &v1) {
  return IndexT(v0) * IndexT(v1);
}

// ============================================================
// Compile-time-preserving overloads for std::integral_constant
// (used when strided_slice template parameters are std::integral_constant)
// ============================================================

template <class IndexT, class T0, T0 v0, class T1, T1 v1>
MDSPAN_INLINE_FUNCTION
constexpr auto divide(const std::integral_constant<T0, v0> &,
                      const std::integral_constant<T1, v1> &) {
  // Short-circuit division by zero
  // (used for strided_slice with zero extent/stride)
  return integral_constant<IndexT, v0 == 0 ? 0 : v0 / v1>();
}

template <class IndexT, class T0, T0 v0, class T1, T1 v1>
MDSPAN_INLINE_FUNCTION
constexpr auto multiply(const std::integral_constant<T0, v0> &,
                        const std::integral_constant<T1, v1> &) {
  return integral_constant<IndexT, v0 * v1>();
}

// ============================================================
// Compile-time-preserving overloads for constant_wrapper
// ============================================================

#if defined(__cpp_lib_constant_wrapper)

// std::constant_wrapper takes a single NTTP <auto V>
template <class IndexT, auto v0, auto v1>
MDSPAN_INLINE_FUNCTION
constexpr auto divide(const constant_wrapper<v0> &,
                      const constant_wrapper<v1> &) {
  constexpr IndexT result =
      IndexT(v0) == IndexT(0) ? IndexT(0) : IndexT(v0) / IndexT(v1);
  return cw<result>;
}

template <class IndexT, auto v0, auto v1>
MDSPAN_INLINE_FUNCTION
constexpr auto multiply(const constant_wrapper<v0> &,
                        const constant_wrapper<v1> &) {
  constexpr IndexT result = IndexT(v0) * IndexT(v1);
  return cw<result>;
}

#else // back-port: constant_wrapper<v, T> = integral_constant<T, v>

template <class IndexT, class T0, T0 v0, class T1, T1 v1>
MDSPAN_INLINE_FUNCTION
constexpr auto divide(const constant_wrapper<v0, T0> &,
                      const constant_wrapper<v1, T1> &) {
  return integral_constant<IndexT, v0 == 0 ? 0 : v0 / v1>();
}

template <class IndexT, class T0, T0 v0, class T1, T1 v1>
MDSPAN_INLINE_FUNCTION
constexpr auto multiply(const constant_wrapper<v0, T0> &,
                        const constant_wrapper<v1, T1> &) {
  return integral_constant<IndexT, v0 * v1>();
}

#endif // __cpp_lib_constant_wrapper

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE
