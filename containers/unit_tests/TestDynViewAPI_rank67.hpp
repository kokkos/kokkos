// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

// Copyright Contributors to the Kokkos project
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <TestDynViewAPI.hpp>
namespace Test {
TEST(TEST_CATEGORY, dyn_rank_view_api_operator_rank67) {
  TestDynViewAPI<double, TEST_EXECSPACE>::run_operator_test_rank67();
}
}  // namespace Test
