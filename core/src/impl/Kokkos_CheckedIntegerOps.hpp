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

#ifndef KOKKOS_CHECKED_INTEGER_OPS_HPP
#define KOKKOS_CHECKED_INTEGER_OPS_HPP

#include <type_traits>

namespace Kokkos {
namespace Impl {

template <typename T>
std::enable_if_t<std::is_integral_v<T>, bool> constexpr multiply_overflow(
    T a, T b, T& res) {
  static_assert(std::is_unsigned_v<T>,
                "Operation not implemented for signed integers.");
  auto product = a * b;
  if ((a == 0) or (b == 0) or (a == product / b)) {
    res = product;
    return false;
  } else {
    return true;
  }
}

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_CHECKED_INTEGER_OPS_HPP
