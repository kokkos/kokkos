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
#include "ctest_common.hpp"

#include <mdspan/mdspan.hpp>


using E1 = Kokkos::extents<int32_t, Kokkos::dynamic_extent, 3>;

MDSPAN_STATIC_TEST(
  std::is_same<typename E1::index_type, int32_t>::value &&
  std::is_same<typename E1::size_type, uint32_t>::value &&
  std::is_same<typename E1::rank_type, size_t>::value &&
  std::is_same<decltype(E1::rank()), typename E1::rank_type>::value &&
  std::is_same<decltype(E1::rank_dynamic()), typename E1::rank_type>::value &&
  std::is_same<decltype(E1::static_extent(0)), size_t>::value &&
  std::is_same<decltype(E1::static_extent(1)), size_t>::value &&
  std::is_same<decltype(std::declval<E1>().extent(0)), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<E1>().extent(1)), typename E1::index_type>::value &&
  (E1::rank()==2) &&
  (E1::rank_dynamic()==1) &&
  (E1::static_extent(0) == Kokkos::dynamic_extent) &&
  (E1::static_extent(1) == 3)
);

using E2 = Kokkos::extents<int64_t, Kokkos::dynamic_extent, 3, Kokkos::dynamic_extent>;

MDSPAN_STATIC_TEST(
  std::is_same<typename E2::index_type, int64_t>::value &&
  std::is_same<typename E2::size_type, uint64_t>::value &&
  std::is_same<typename E2::rank_type, size_t>::value &&
  std::is_same<decltype(E2::rank()), typename E2::rank_type>::value &&
  std::is_same<decltype(E2::rank_dynamic()), typename E2::rank_type>::value &&
  std::is_same<decltype(E2::static_extent(0)), size_t>::value &&
  std::is_same<decltype(E2::static_extent(1)), size_t>::value &&
  std::is_same<decltype(E2::static_extent(2)), size_t>::value &&
  std::is_same<decltype(std::declval<E2>().extent(0)), typename E2::index_type>::value &&
  std::is_same<decltype(std::declval<E2>().extent(1)), typename E2::index_type>::value &&
  std::is_same<decltype(std::declval<E2>().extent(2)), typename E2::index_type>::value &&
  (E2::rank()==3) &&
  (E2::rank_dynamic()==2) &&
  (E2::static_extent(0) == Kokkos::dynamic_extent) &&
  (E2::static_extent(1) == 3) &&
  (E2::static_extent(2) == Kokkos::dynamic_extent)
);

using E3 = Kokkos::extents<uint32_t, Kokkos::dynamic_extent, 3, Kokkos::dynamic_extent>;

MDSPAN_STATIC_TEST(
  std::is_same<typename E3::index_type, uint32_t>::value &&
  std::is_same<typename E3::size_type, uint32_t>::value &&
  std::is_same<typename E3::rank_type, size_t>::value &&
  std::is_same<decltype(E3::rank()), typename E3::rank_type>::value &&
  std::is_same<decltype(E3::rank_dynamic()), typename E3::rank_type>::value &&
  std::is_same<decltype(E3::static_extent(0)), size_t>::value &&
  std::is_same<decltype(E3::static_extent(1)), size_t>::value &&
  std::is_same<decltype(E3::static_extent(2)), size_t>::value &&
  std::is_same<decltype(std::declval<E3>().extent(0)), typename E3::index_type>::value &&
  std::is_same<decltype(std::declval<E3>().extent(1)), typename E3::index_type>::value &&
  std::is_same<decltype(std::declval<E3>().extent(2)), typename E3::index_type>::value &&
  (E3::rank()==3) &&
  (E3::rank_dynamic()==2) &&
  (E3::static_extent(0) == Kokkos::dynamic_extent) &&
  (E3::static_extent(1) == 3) &&
  (E3::static_extent(2) == Kokkos::dynamic_extent)
);
