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

#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>

#include <regex>

namespace {

TEST(TEST_CATEGORY, range_policy_runtime_parameters) {
  using Policy     = Kokkos::RangePolicy<>;
  using Index      = Policy::index_type;
  Index work_begin = 5;
  Index work_end   = 15;
  Index chunk_size = 10;
  {
    Policy p(work_begin, work_end);
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
  }
  {
    Policy p(Kokkos::DefaultExecutionSpace(), work_begin, work_end);
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
  }
  {
    Policy p(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
    ASSERT_EQ(p.chunk_size(), chunk_size);
  }
  {
    Policy p(Kokkos::DefaultExecutionSpace(), work_begin, work_end,
             Kokkos::ChunkSize(chunk_size));
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
    ASSERT_EQ(p.chunk_size(), chunk_size);
  }
  {
    Policy p;  // default-constructed
    ASSERT_EQ(p.begin(), Index(0));
    ASSERT_EQ(p.end(), Index(0));
    ASSERT_EQ(p.chunk_size(), Index(0));

    // copy-assigned
    p = Policy(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
    ASSERT_EQ(p.begin(), work_begin);
    ASSERT_EQ(p.end(), work_end);
    ASSERT_EQ(p.chunk_size(), chunk_size);
  }
  {
    Policy p1(work_begin, work_end, Kokkos::ChunkSize(chunk_size));
    Policy p2(p1);  // copy-constructed
    ASSERT_EQ(p1.begin(), p2.begin());
    ASSERT_EQ(p1.end(), p2.end());
    ASSERT_EQ(p1.chunk_size(), p2.chunk_size());
  }
}

TEST(TEST_CATEGORY_DEATH, range_policy_invalid_bounds) {
  using Policy    = Kokkos::RangePolicy<TEST_EXECSPACE>;
  using ChunkSize = Kokkos::ChunkSize;

  std::string msg =
      "Kokkos::RangePolicy bounds error: The lower bound (100) is greater than "
      "the upper bound (90).\n";
#ifndef KOKKOS_ENABLE_DEPRECATED_CODE_4
  // escape the parentheses in the regex to match the error message
  msg = std::regex_replace(msg, std::regex("\\(|\\)"), "\\$&");
  ASSERT_DEATH({ (void)Policy(100, 90); }, msg);

  ASSERT_DEATH({ (void)Policy(TEST_EXECSPACE(), 100, 90, ChunkSize(10)); },
               msg);
#else

  if (!Kokkos::show_warnings()) {
    GTEST_SKIP() << "Kokkos warning messages are disabled";
  }

  {
    ::testing::internal::CaptureStderr();
    Policy policy(100, 90);
    ASSERT_EQ((int)policy.begin(), 0);
    ASSERT_EQ((int)policy.end(), 0);
#ifdef KOKKOS_ENABLE_DEPRECATION_WARNINGS
    ASSERT_EQ(::testing::internal::GetCapturedStderr(), msg);
#else
    ASSERT_TRUE(::testing::internal::GetCapturedStderr().empty());
    (void)msg;
#endif
  }

  {
    ::testing::internal::CaptureStderr();
    Policy policy(TEST_EXECSPACE(), 100, 90, ChunkSize(10));
    ASSERT_EQ((int)policy.begin(), 0);
    ASSERT_EQ((int)policy.end(), 0);
#ifdef KOKKOS_ENABLE_DEPRECATION_WARNINGS
    ASSERT_EQ(::testing::internal::GetCapturedStderr(), msg);
#else
    ASSERT_TRUE(::testing::internal::GetCapturedStderr().empty());
    (void)msg;
#endif
  }

#endif
}

}  // namespace
