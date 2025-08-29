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
struct reduction_identity;

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
