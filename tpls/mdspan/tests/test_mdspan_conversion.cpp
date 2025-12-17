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


#include <mdspan/mdspan.hpp>

#include <gtest/gtest.h>


TEST(TestMdspanConversionConst, test_mdspan_conversion_const) {
  std::array<double, 6> a{};
  Kokkos::mdspan<double, Kokkos::extents<uint32_t, 2, 3>> s(a.data());
  ASSERT_EQ(s.data_handle(), a.data());
  MDSPAN_IMPL_OP(s, 0, 1) = 3.14;
  Kokkos::mdspan<double const, Kokkos::extents<uint64_t, 2, 3>> c_s(s);
  ASSERT_EQ((MDSPAN_IMPL_OP(c_s, 0, 1)), 3.14);
}
