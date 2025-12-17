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
using MLR1 = Kokkos::layout_right::mapping<E1>;
// Note we already separately tested that the extents interface is correct
MDSPAN_STATIC_TEST(
  std::is_same<typename MLR1::index_type, typename E1::index_type>::value &&
  std::is_same<typename MLR1::size_type,  typename E1::size_type>::value &&
  std::is_same<typename MLR1::rank_type,  typename E1::rank_type>::value &&
  std::is_same<typename MLR1::layout_type, Kokkos::layout_right>::value &&
  std::is_same<decltype(MLR1::is_always_unique()), bool>::value &&
  std::is_same<decltype(MLR1::is_always_exhaustive()), bool>::value &&
  std::is_same<decltype(MLR1::is_always_strided()), bool>::value &&
  std::is_same<decltype(std::declval<MLR1>().extents()), const E1&>::value &&
  std::is_same<decltype(std::declval<MLR1>().stride(0)), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<MLR1>().required_span_size()), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<MLR1>().is_unique()), bool>::value &&
  std::is_same<decltype(std::declval<MLR1>().is_exhaustive()), bool>::value &&
  std::is_same<decltype(std::declval<MLR1>().is_strided()), bool>::value &&
  std::is_same<decltype(std::declval<MLR1>()(1,1)), typename E1::index_type>::value &&
  (MLR1::is_always_unique() == true) &&
  (MLR1::is_always_exhaustive()== true) &&
  (MLR1::is_always_strided() == true)
);

using MLL1 = Kokkos::layout_left::mapping<E1>;

MDSPAN_STATIC_TEST(
  std::is_same<typename MLL1::index_type, typename E1::index_type>::value &&
  std::is_same<typename MLL1::size_type,  typename E1::size_type>::value &&
  std::is_same<typename MLL1::rank_type,  typename E1::rank_type>::value &&
  std::is_same<typename MLL1::layout_type, Kokkos::layout_left>::value &&
  std::is_same<decltype(MLL1::is_always_unique()), bool>::value &&
  std::is_same<decltype(MLL1::is_always_exhaustive()), bool>::value &&
  std::is_same<decltype(MLL1::is_always_strided()), bool>::value &&
  std::is_same<decltype(std::declval<MLL1>().extents()), const E1&>::value &&
  std::is_same<decltype(std::declval<MLL1>().stride(0)), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<MLL1>().required_span_size()), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<MLL1>().is_unique()), bool>::value &&
  std::is_same<decltype(std::declval<MLL1>().is_exhaustive()), bool>::value &&
  std::is_same<decltype(std::declval<MLL1>().is_strided()), bool>::value &&
  std::is_same<decltype(std::declval<MLL1>()(1,1)), typename E1::index_type>::value &&
  (MLL1::is_always_unique() == true) &&
  (MLL1::is_always_exhaustive()== true) &&
  (MLL1::is_always_strided() == true)
);


using MLS1 = Kokkos::layout_stride::mapping<E1>;

MDSPAN_STATIC_TEST(
  std::is_same<typename MLS1::index_type, typename E1::index_type>::value &&
  std::is_same<typename MLS1::size_type,  typename E1::size_type>::value &&
  std::is_same<typename MLS1::rank_type,  typename E1::rank_type>::value &&
  std::is_same<typename MLS1::layout_type, Kokkos::layout_stride>::value &&
  std::is_same<decltype(MLS1::is_always_unique()), bool>::value &&
  std::is_same<decltype(MLS1::is_always_exhaustive()), bool>::value &&
  std::is_same<decltype(MLS1::is_always_strided()), bool>::value &&
  std::is_same<decltype(std::declval<MLS1>().extents()), const E1&>::value &&
  std::is_same<decltype(std::declval<MLS1>().stride(0)), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<MLS1>().required_span_size()), typename E1::index_type>::value &&
  std::is_same<decltype(std::declval<MLS1>().is_unique()), bool>::value &&
  std::is_same<decltype(std::declval<MLS1>().is_exhaustive()), bool>::value &&
  std::is_same<decltype(std::declval<MLS1>().is_strided()), bool>::value &&
  std::is_same<decltype(std::declval<MLS1>()(1,1)), typename E1::index_type>::value &&
  (MLS1::is_always_unique() == true) &&
  (MLS1::is_always_exhaustive()== false) &&
  (MLS1::is_always_strided() == true)
);
