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

#ifndef KOKKOS_TEST_SUBVIEW_C05_HPP
#define KOKKOS_TEST_SUBVIEW_C05_HPP
#include <TestViewSubview.hpp>

namespace Test {

TEST(TEST_CATEGORY, view_subview_2d_from_3d_atomic) {
  TestViewSubview::test_2d_subview_3d<TEST_EXECSPACE,
                                      Kokkos::MemoryTraits<Kokkos::Atomic> >();
}

}  // namespace Test
#endif
