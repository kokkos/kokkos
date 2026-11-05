// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef KOKKOS_TEST_SUBVIEW_C10_HPP
#define KOKKOS_TEST_SUBVIEW_C10_HPP
#include <TestViewSubview.hpp>

namespace Test {

TEST(TEST_CATEGORY, view_subview_3d_from_5d_right) {
  TestViewSubview::test_3d_subview_5d_right<TEST_EXECSPACE>();
}

}  // namespace Test
#endif
