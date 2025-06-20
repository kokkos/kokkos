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

#ifndef KOKKOS_IMPL_KOKKOS_GRAPHNODETHENPOLICY_HPP
#define KOKKOS_IMPL_KOKKOS_GRAPHNODETHENPOLICY_HPP

#include <impl/Kokkos_GraphImpl_fwd.hpp>

#include <type_traits>

namespace Kokkos::Impl {
// Concept for a tag class.
template <typename T>
struct is_tag_class
    : std::integral_constant<
          bool, std::conjunction_v<std::is_class<T>, std::is_empty<T>,
                                   std::is_trivially_default_constructible<T>,
                                   std::is_trivially_copy_constructible<T>,
                                   std::is_trivially_move_constructible<T>,
                                   std::is_trivially_destructible<T>>> {};

template <typename T>
inline constexpr bool is_tag_class_v = is_tag_class<T>::value;
}  // namespace Kokkos::Impl

namespace Kokkos::Experimental {
template <typename WorkTag>
struct ThenPolicy {
  static_assert(Kokkos::Impl::is_tag_class_v<WorkTag> ||
                std::is_void_v<WorkTag>);

  using work_tag = WorkTag;
};
}  // namespace Kokkos::Experimental

#endif  // KOKKOS_IMPL_KOKKOS_GRAPHNODETHENPOLICY_HPP
