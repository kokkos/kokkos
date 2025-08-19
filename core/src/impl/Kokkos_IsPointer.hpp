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

#ifndef KOKKOS_IS_POINTER_HPP
#define KOKKOS_IS_POINTER_HPP

#include <type_traits>
#include <Kokkos_Macros.hpp>

namespace Kokkos::Impl {

template <typename T>
struct remove_restrict {
  using type = T;
};

template <typename T>
struct remove_restrict<T * KOKKOS_RESTRICT> {
  using type = T*;
};

template <typename T>
using remove_restrict_t = typename remove_restrict<T>::type;

/* std::is_pointer_v doesn't work with restrict pointers */
template <typename T>
inline constexpr bool is_pointer_v = std::is_pointer_v<remove_restrict_t<T>>;

}  // namespace Kokkos::Impl

#endif
