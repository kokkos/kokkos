// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
