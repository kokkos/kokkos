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

struct TestRangePolicyCTADs {
  struct SomeExecutionSpace {
    using execution_space = SomeExecutionSpace;
    using size_type       = size_t;
  };
  static_assert(Kokkos::is_execution_space_v<SomeExecutionSpace>);

  struct ImplicitlyConvertibleToDefaultExecutionSpace {
    operator Kokkos::DefaultExecutionSpace() const;
  };
  static_assert(!Kokkos::is_execution_space_v<
                ImplicitlyConvertibleToDefaultExecutionSpace>);

  static Kokkos::DefaultExecutionSpace des;
  static ImplicitlyConvertibleToDefaultExecutionSpace notEs;
  static SomeExecutionSpace ses;
  static TEST_EXECSPACE es;

  using RangePolicyTestExecSpace = std::conditional_t<
      std::is_same_v<TEST_EXECSPACE, Kokkos::DefaultExecutionSpace>,
      Kokkos::RangePolicy<>, Kokkos::RangePolicy<TEST_EXECSPACE> >;

  static int64_t i64;
  static int32_t i32;
  static Kokkos::ChunkSize cs;

  static Kokkos::RangePolicy<> rp;

  //  Copy/move

  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>, decltype(Kokkos::RangePolicy(rp))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(std::move(rp)))>);

  static Kokkos::RangePolicy<SomeExecutionSpace> rpses;
  static_assert(std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                               decltype(Kokkos::RangePolicy(rpses))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                     decltype(Kokkos::RangePolicy(std::move(rpses)))>);

  //  RangePolicy

  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>, decltype(Kokkos::RangePolicy())>);

  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(des, i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(notEs, i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                               decltype(Kokkos::RangePolicy(ses, i64, i64))>);
  static_assert(std::is_same_v<RangePolicyTestExecSpace,
                               decltype(Kokkos::RangePolicy(es, i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(des, i32, i32))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(notEs, i32, i32))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                               decltype(Kokkos::RangePolicy(ses, i32, i32))>);
  static_assert(std::is_same_v<RangePolicyTestExecSpace,
                               decltype(Kokkos::RangePolicy(es, i32, i32))>);

  // RangePolicy(index_type, index_type)

  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i64, i64))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i32, i32))>);

  // RangePolicy(execution_space, index_type, index_type, Args...)

  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(des, i64, i64, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(notEs, i64, i64, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                     decltype(Kokkos::RangePolicy(ses, i64, i64, cs))>);
  static_assert(
      std::is_same_v<RangePolicyTestExecSpace,
                     decltype(Kokkos::RangePolicy(es, i64, i64, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(des, i32, i32, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<>,
                     decltype(Kokkos::RangePolicy(notEs, i32, i32, cs))>);
  static_assert(
      std::is_same_v<Kokkos::RangePolicy<SomeExecutionSpace>,
                     decltype(Kokkos::RangePolicy(ses, i32, i32, cs))>);
  static_assert(
      std::is_same_v<RangePolicyTestExecSpace,
                     decltype(Kokkos::RangePolicy(es, i32, i32, cs))>);

  // RangePolicy(index_type, index_type, Args...)

  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i64, i64, cs))>);
  static_assert(std::is_same_v<Kokkos::RangePolicy<>,
                               decltype(Kokkos::RangePolicy(i32, i32, cs))>);
};  // TestRangePolicyCTADs struct

}  // namespace
