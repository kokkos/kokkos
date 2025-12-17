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


TEST(TestElementAccess, element_access_with_std_array) {
    std::array<double, 6> a{};
    Kokkos::mdspan<double, Kokkos::extents<size_t,2, 3>> s(a.data());
    ASSERT_EQ(MDSPAN_IMPL_OP(s, (std::array<int, 2>{1, 2})), 0);
    MDSPAN_IMPL_OP(s, (std::array<int, 2>{0, 1})) = 3.14;
    ASSERT_EQ(MDSPAN_IMPL_OP(s, (std::array<int, 2>{0, 1})), 3.14);
}
