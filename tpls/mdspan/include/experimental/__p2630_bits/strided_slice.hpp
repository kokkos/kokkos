
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

#include "../__p0009_bits/config.hpp"
#include "constant_wrapper.hpp"
#include "integral_constant_like.hpp"

#include <type_traits>

namespace MDSPAN_IMPL_STANDARD_NAMESPACE {

namespace detail {
  template<class T, class = void>
  struct is_signed_or_unsigned_integral_constant_like : std::false_type {};

  template<class T>
  struct is_signed_or_unsigned_integral_constant_like<
    T, std::enable_if_t<is_integral_constant_like_v<T>>
  > : std::bool_constant<
      std::is_integral_v<remove_cvref_t<decltype(T::value)>> &&
      ! std::is_same_v<bool, remove_cvref_t<decltype(T::value)>>
    >
  {};

  template<class T>
  constexpr bool is_signed_or_unsigned_integral_constant_like_v =
    is_signed_or_unsigned_integral_constant_like<T>::value;

  template<class T>
  constexpr bool mdspan_is_index_like_v =
    (std::is_integral_v<T> && ! std::is_same_v<bool, T>) ||
    is_signed_or_unsigned_integral_constant_like_v<T>;
} // namespace detail

// Slice Specifier allowing for strides and compile time extent
template <class OffsetType, class ExtentType, class StrideType>
struct strided_slice {
  using offset_type = OffsetType;
  using extent_type = ExtentType;
  using stride_type = StrideType;

  MDSPAN_IMPL_NO_UNIQUE_ADDRESS OffsetType offset{};
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS ExtentType extent{};
  MDSPAN_IMPL_NO_UNIQUE_ADDRESS StrideType stride{};

  static_assert(detail::mdspan_is_index_like_v<OffsetType>);
  static_assert(detail::mdspan_is_index_like_v<ExtentType>);
  static_assert(detail::mdspan_is_index_like_v<StrideType>);
};

} // MDSPAN_IMPL_STANDARD_NAMESPACE
