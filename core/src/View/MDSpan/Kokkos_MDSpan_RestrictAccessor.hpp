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

#ifndef KOKKOS_RESTRICTACCESSOR_HPP
#define KOKKOS_RESTRICTACCESSOR_HPP

#include <Kokkos_Macros.hpp>

namespace Kokkos::Impl {

template <class ElementType>
struct RestrictAccessor {
  using offset_policy    = RestrictAccessor;
  using element_type     = ElementType;
  using reference        = ElementType& KOKKOS_RESTRICT;
  using data_handle_type = ElementType* KOKKOS_RESTRICT;

  KOKKOS_DEFAULTED_FUNCTION constexpr RestrictAccessor() noexcept = default;

  // (incl. non-const to const)
  template <
      class OtherElementType,
      typename ::std::enable_if<
          (std::is_convertible_v<OtherElementType (*)[], element_type (*)[]>),
          int>::type = 0>
  KOKKOS_INLINE_FUNCTION constexpr RestrictAccessor(
      RestrictAccessor<OtherElementType>) noexcept {}

  // Conversion from default_accessor (incl. non-const to const)
  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<
                OtherElementType (*)[], element_type (*)[]>>* = nullptr>
  KOKKOS_FUNCTION constexpr RestrictAccessor(
      Kokkos::default_accessor<OtherElementType>) noexcept {}

  // Conversion to default_accessor (incl. non-const to const)
  template <class OtherElementType,
            std::enable_if_t<std::is_convertible_v<
                element_type (*)[], OtherElementType (*)[]>>* = nullptr>
  KOKKOS_FUNCTION explicit operator default_accessor<OtherElementType>() const {
    return default_accessor<OtherElementType>{};
  }

  KOKKOS_INLINE_FUNCTION
  constexpr data_handle_type offset(data_handle_type p,
                                    size_t i) const noexcept {
    return p + i;
  }

  KOKKOS_FORCEINLINE_FUNCTION
  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p[i];
  }
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_RESTRICACCESSOR_HPP
