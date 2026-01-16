// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <TestNextSilicon_Category.hpp>

#include <nsapi/memory.h>

/*! \brief Make sure some symbol from libnsapi can be linked */

namespace Test {
TEST(nextsilicon, nsapi_link) {
  // doesn't really do anything and don't care if it does
  nsapi_mem_migrate(nullptr, 0, NSAPI_PAGE_LOC_HOST, false);
}
}  // namespace Test
