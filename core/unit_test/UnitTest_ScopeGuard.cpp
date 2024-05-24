//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.3
//       Copyright (2024) National Technology & Engineering
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

#include <cstdlib>
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

namespace Test {

/**
 * Test to create a scope guard normally.
 */
TEST(scope_guard, create) {
  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());

  // run it in a different process so side effects are not kept
  EXPECT_EXIT(
      {
        {
          Kokkos::ScopeGuard guard{};

          ASSERT_TRUE(Kokkos::is_initialized());
          ASSERT_FALSE(Kokkos::is_finalized());
        }

        ASSERT_FALSE(Kokkos::is_initialized());
        ASSERT_TRUE(Kokkos::is_finalized());

        std::exit(EXIT_SUCCESS);
      },
      testing::ExitedWithCode(0), "");

  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());
}

/**
 * Test to create another scope guard when one has been created.
 */
TEST(scope_guard, create_while_initialize) {
  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());

  EXPECT_DEATH(
      {
        Kokkos::ScopeGuard guard1{};

        // create a second scope guard while there is one already existing
        Kokkos::ScopeGuard guard2{};
      },
      "Creating a ScopeGuard while Kokkos is initialized");

  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());
}

/**
 * Test to create another scope guard when one has been destroyed.
 */
TEST(scope_guard, create_after_finalize) {
  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());

  EXPECT_DEATH(
      {
        { Kokkos::ScopeGuard guard1{}; }

        // create a second scope guard while the first one has been destroyed
        // already
        Kokkos::ScopeGuard guard2{};
      },
      "Creating a ScopeGuard after Kokkos was finalized");

  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());
}

/**
 * Test to destroy a scope guard when finalization has been done manually.
 */
TEST(scope_guard, destroy_after_finalize) {
  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());

  EXPECT_DEATH(
      {
        // create a scope guard and finalize it manually
        Kokkos::ScopeGuard guard{};
        Kokkos::finalize();
      },
      "Destroying a ScopeGuard after Kokkos was finalized");

  ASSERT_FALSE(Kokkos::is_initialized());
  ASSERT_FALSE(Kokkos::is_finalized());
}

}  // namespace Test
