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

template <class ExecutionSpace>
struct CheckClassWithExecutionSpaceAsDataMemberIsCopyable {
  Kokkos::DefaultExecutionSpace device;
  Kokkos::DefaultHostExecutionSpace host;

  KOKKOS_FUNCTION void operator()(int, int& e) const {
    // not actually doing anything useful, mostly checking that
    // ExecutionSpace::in_parallel() is callable
    if (static_cast<int>(device.in_parallel()) < 0) {
      ++e;
    }
  }

  CheckClassWithExecutionSpaceAsDataMemberIsCopyable() {
    int errors;
    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, 1), *this,
                            errors);
    EXPECT_EQ(errors, 0);
  }
};

// FIXME_OPENMPTARGET nvlink error: Undefined reference to
// '_ZSt25__throw_bad_function_callv' in
// '/tmp/TestOpenMPTarget_ExecutionSpace-434d81.cubin'
#ifndef KOKKOS_ENABLE_OPENMPTARGET
TEST(TEST_CATEGORY, execution_space_as_class_data_member) {
  CheckClassWithExecutionSpaceAsDataMemberIsCopyable<TEST_EXECSPACE>();
}
#endif

template <class ExecutionSpace>
struct CheckExecutionSpaceStatus {
  KOKKOS_FUNCTION void operator()(int i, int& update) const { update += i; };

  CheckExecutionSpaceStatus() { check(); }

  void check() const {
    ExecutionSpace exec;
    ASSERT_FALSE(exec.is_running());

    Kokkos::View<int, typename ExecutionSpace::memory_space> result_view(
        "result_view");
    int result;

    Kokkos::deep_copy(exec, result, result_view);
    while (exec.is_running()) {
    }
    ASSERT_EQ(result, 0);

    Kokkos::deep_copy(result_view, 1);
    ASSERT_FALSE(exec.is_running());
    Kokkos::deep_copy(exec, result, result_view);
    exec.fence();
    ASSERT_FALSE(exec.is_running());
    ASSERT_EQ(result, 1);

// FIXME OPENACC
#ifdef KOKKOS_ENABLE_OPENACC
    if constexpr (std::is_same_v<ExecutionSpace,
                                 Kokkos::Experimental::OpenACC>) {
      GTEST_SKIP() << "skip the other half of the test because of missing "
                      "functionality in OpenACC backend";
    }
#endif

    const int N = 10000;

    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, N),
                            *this, result_view);
    Kokkos::deep_copy(exec, result, result_view);
    // This happens to work for now but we reserve the right to have a lazy
    // implementation which would make this hang.
    while (exec.is_running()) {
    }
    ASSERT_EQ(result, N / 2 * (N - 1));

    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, N),
                            *this, result_view);
    Kokkos::deep_copy(exec, result, result_view);
    exec.fence();
    ASSERT_FALSE(exec.is_running());
    ASSERT_EQ(result, N / 2 * (N - 1));
  }
};

TEST(TEST_CATEGORY, execution_space_status) {
  CheckExecutionSpaceStatus<TEST_EXECSPACE>();
}

}  // namespace
