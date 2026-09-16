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
#if defined(__cpp_lib_concepts)
#  include <concepts>
#endif // __cpp_lib_concepts

// ============================================================
// equality_comparable back-port (used only by integral_constant_like)
// ============================================================

#if defined(__cpp_lib_concepts)

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
  namespace detail {
    template<class T, class = void>
    struct is_equality_comparable : std::bool_constant<std::equality_comparable<T>> {};

    template<class T, class U, class = void>
    struct is_equality_comparable_with : std::bool_constant<std::equality_comparable_with<T, U>> {};
  } // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#else

#include <utility>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
namespace detail {

  template<typename T, typename = void>
  struct is_equality_comparable : std::false_type {};

  template<typename T>
  struct is_equality_comparable<
    T,
    std::void_t<
      decltype(std::declval<const T&>() == std::declval<const T&>()),
      decltype(std::declval<const T&>() != std::declval<const T&>())
    >
  > : std::bool_constant<
    std::is_convertible_v<
      decltype(std::declval<const T&>() == std::declval<const T&>()),
      bool
    > &&
    std::is_convertible_v<
      decltype(std::declval<const T&>() != std::declval<const T&>()),
      bool
    >
  > {};

  template<typename T, typename U, typename = void>
  struct is_equality_comparable_with : std::false_type {};

  template<typename T, typename U>
  struct is_equality_comparable_with<
    T, U,
    std::void_t<
      decltype(std::declval<const T&>() == std::declval<const U&>()),
      decltype(std::declval<const T&>() != std::declval<const U&>()),
      decltype(std::declval<const U&>() == std::declval<const T&>()),
      decltype(std::declval<const U&>() != std::declval<const T&>())
    >
  > : std::bool_constant<
    is_equality_comparable<T>::value &&
    is_equality_comparable<U>::value &&
    std::is_convertible_v<
      decltype(std::declval<const T&>() == std::declval<const U&>()),
      bool
    > &&
    std::is_convertible_v<
      decltype(std::declval<const T&>() != std::declval<const U&>()),
      bool
    > &&
    std::is_convertible_v<
      decltype(std::declval<const U&>() == std::declval<const T&>()),
      bool
    > &&
    std::is_convertible_v<
      decltype(std::declval<const U&>() != std::declval<const T&>()),
      bool
    >
  > {};

} // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif // defined(__cpp_lib_concepts)

// ============================================================
// integral_constant_like concept / trait
// ============================================================

#if defined(__cpp_lib_concepts)

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
  namespace detail {

    template<class T>
    concept integral_constant_like =
      std::is_integral_v<std::remove_cvref_t<decltype(T::value)>> &&
      !std::is_same_v<bool, std::remove_cvref_t<decltype(T::value)>> &&
      std::convertible_to<T, decltype(T::value)> &&
      std::equality_comparable_with<T, decltype(T::value)> &&
      std::bool_constant<T() == T::value>::value &&
      std::bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;

    template<class T>
    constexpr bool is_integral_constant_like_v = integral_constant_like<T>;

  } // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#else

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {
  namespace detail {

    template<class T, class = void>
    struct is_integral_constant_like_impl : std::false_type {};

    template<class T>
    struct is_integral_constant_like_impl<T, std::void_t<decltype(T::value), decltype(T())>> :
      std::bool_constant<
        std::is_integral_v<remove_cvref_t<decltype(T::value)>> &&
        ! std::is_same_v<bool, remove_cvref_t<decltype(T::value)>> &&
        std::is_convertible_v<T, decltype(T::value)> &&
        is_equality_comparable_with<T, decltype(T::value)>::value &&
        std::bool_constant<T() == T::value>::value &&
        std::bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value
      >
    {};

    template<class T>
    constexpr bool is_integral_constant_like_v = is_integral_constant_like_impl<T>::value;

  } // namespace detail
} // namespace MDSPAN_IMPL_STANDARD_NAMESPACE

#endif // __cpp_lib_concepts
