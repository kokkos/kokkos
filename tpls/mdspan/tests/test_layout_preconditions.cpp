//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2024) National Technology & Engineering
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
#include <mdspan/mdspan.hpp>

#include <gtest/gtest.h>

TEST(TestConvertingConstructionFromLayoutStride, precondition_failure) {
  using E = Kokkos::extents<size_t, 2, 2>;

  const auto stride = Kokkos::layout_stride::mapping<E>{E{}, std::array<size_t, 2>{2, 8}};

#if defined(MDSPAN_IMPL_HAS_CUDA) || defined(MDSPAN_IMPL_HAS_HIP) || defined(MDSPAN_IMPL_HAS_SYCL)
  ASSERT_DEATH(Kokkos::layout_left::mapping<E>{stride}, "");
  ASSERT_DEATH(Kokkos::layout_right::mapping<E>{stride}, "");
#else
  ASSERT_DEATH(Kokkos::layout_left::mapping<E>{stride}, "invalid strides");
  ASSERT_DEATH(Kokkos::layout_right::mapping<E>{stride}, "invalid strides");
#endif
}
