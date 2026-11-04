// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KOKKOS_TEST_SUBVIEW_C11_HPP
#define KOKKOS_TEST_SUBVIEW_C11_HPP
#include <TestViewSubview.hpp>

namespace Test {

TEST(TEST_CATEGORY, view_subview_3d_from_5d_right_atomic) {
  TestViewSubview::test_3d_subview_5d_right<
      TEST_EXECSPACE, Kokkos::MemoryTraits<Kokkos::Atomic> >();
}

}  // namespace Test
#endif
