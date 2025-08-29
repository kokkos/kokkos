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

#ifndef KOKKOS_REDUCTION_IDENTITY_HPP
#define KOKKOS_REDUCTION_IDENTITY_HPP
#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_REDUCTION_IDENTITY
#endif

#include <Kokkos_Macros.hpp>
#include <concepts>
#include <limits>

namespace Kokkos {

template <class T>
struct reduction_identity; /*{
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T sum() { return T(); }  // 0
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T prod()  // 1
    { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom prod reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T max()   // minimum value
    { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom max reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T min()   // maximum value
    { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom min reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T bor()   // 0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom bor reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T band()  // !0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom band reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T lor()   // 0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom lor reduction type"); return T(); }
  KOKKOS_FORCEINLINE_FUNCTION constexpr static T land()  // !0, only for integer
type { static_assert( false, "Missing specialization of
Kokkos::reduction_identity for custom land reduction type"); return T(); }
};*/

template <typename Integral>
  requires(std::integral<Integral>)
struct reduction_identity<Integral> {
  KOKKOS_FUNCTION constexpr static Integral sum() noexcept { return 0; }
  KOKKOS_FUNCTION constexpr static Integral prod() noexcept { return 1; }
  KOKKOS_FUNCTION constexpr static Integral max() noexcept {
    return std::numeric_limits<Integral>::min();
  }
  KOKKOS_FUNCTION constexpr static Integral min() noexcept {
    return std::numeric_limits<Integral>::max();
  }
  KOKKOS_FUNCTION constexpr static Integral bor() noexcept { return 0x0; }
  KOKKOS_FUNCTION constexpr static Integral band() noexcept { return 0x0; }
  KOKKOS_FUNCTION constexpr static Integral lor() noexcept { return 0; }
  KOKKOS_FUNCTION constexpr static Integral land() noexcept { return 1; }
};

template <typename Floating>
  requires(std::floating_point<Floating> && sizeof(Floating) <= sizeof(double))
struct reduction_identity<Floating> {
  KOKKOS_FUNCTION constexpr static Floating sum() noexcept { return 0; }
  KOKKOS_FUNCTION constexpr static Floating prod() noexcept { return 1; }
  KOKKOS_FUNCTION constexpr static Floating max() noexcept {
#if __FINITE_MATH_ONLY__
    return -std::numeric_limits<Floating>::max();
#else
    return -std::numeric_limits<Floating>::infinity();
#endif
  }
  KOKKOS_FUNCTION constexpr static Floating min() noexcept {
#if __FINITE_MATH_ONLY__
    return +std::numeric_limits<Floating>::max();
#else
    return +std::numeric_limits<Floating>::infinity();
#endif
  }
};

// No __host__ __device__ annotation because long double treated as double in
// device code.  May be revisited later if that is not true any more.
template <typename Floating>
  requires(std::floating_point<Floating> && sizeof(Floating) > sizeof(double))
struct reduction_identity<Floating> {
  constexpr static Floating sum() noexcept { return 0; }
  constexpr static Floating prod() noexcept { return 1; }
  constexpr static Floating max() noexcept {
#if __FINITE_MATH_ONLY__
    return -std::numeric_limits<Floating>::max();
#else
    return -std::numeric_limits<Floating>::infinity();
#endif
  }
  constexpr static Floating min() noexcept {
#if __FINITE_MATH_ONLY__
    return +std::numeric_limits<Floating>::max();
#else
    return +std::numeric_limits<Floating>::infinity();
#endif
  }
};

}  // namespace Kokkos

#ifdef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_REDUCTION_IDENTITY
#undef KOKKOS_IMPL_PUBLIC_INCLUDE
#undef KOKKOS_IMPL_PUBLIC_INCLUDE_NOTDEFINED_REDUCTION_IDENTITY
#endif
#endif
