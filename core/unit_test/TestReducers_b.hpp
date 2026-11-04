// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <TestReducers.hpp>

namespace Test {
TEST(TEST_CATEGORY, reducers_size_t) {
  TestReducers<size_t, TEST_EXECSPACE>::execute_integer();
}
}  // namespace Test
