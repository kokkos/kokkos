// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <gtest/gtest.h>
#include "TestCopyViewsBugs.hpp"

namespace Test {

TEST(copyviews_bugs, default_device) {
  using device_type = Kokkos::View<int*>::device_type;
  TestCopyViewsBugs::testCopyViewsBugs<device_type>();
}

}  // namespace Test
