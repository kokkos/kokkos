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

#ifndef KOKKOS_SWAP_HPP
#define KOKKOS_SWAP_HPP

#include <Kokkos_Macros.hpp>

#include <type_traits>
#include <utility>

namespace Kokkos {
namespace Experimental {

template <class T>
KOKKOS_INLINE_FUNCTION void swap(T& a, T& b) noexcept(
    std::is_nothrow_move_constructible_v<T>&&
        std::is_nothrow_move_assignable_v<T>) {
  static_assert(
      std::is_move_assignable<T>::value && std::is_move_constructible<T>::value,
      "Kokkos::Experimental::swap arguments must be move assignable "
      "and move constructible");

  T tmp = std::move(a);
  a     = std::move(b);
  b     = std::move(tmp);
}

}  // namespace Experimental
}  // namespace Kokkos

#endif
