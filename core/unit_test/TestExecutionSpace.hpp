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

	
  KOKKOS_FUNCTION void operator()(int i, int& update) const
                  {
                    update += i;
                  };

	CheckExecutionSpaceStatus()
{
	check();
}

void check() const {
  std::cout << "submitted: " << static_cast<int>(Kokkos::Experimental::ExecutionSpaceStatus::submitted) << '\n'
	    << "running: "   << static_cast<int>(Kokkos::Experimental::ExecutionSpaceStatus::running)   << '\n'
	    << "complete: "  << static_cast<int>(Kokkos::Experimental::ExecutionSpaceStatus::complete)  << std::endl;
  ExecutionSpace exec;
  std::cout << static_cast<int>(exec.get_status()) << std::endl;
  ASSERT_EQ(exec.get_status(), Kokkos::Experimental::ExecutionSpaceStatus::complete);  
  
  const int N = 10000;
  Kokkos::View<int, typename ExecutionSpace::memory_space> result_view("result_view");
    int result;

  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, N), *this, result_view);
  Kokkos::deep_copy(exec, result, result_view);
  while (exec.get_status() != Kokkos::Experimental::ExecutionSpaceStatus::complete){}
  ASSERT_EQ(result, N/2*(N-1));

  Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(exec, 0, N), *this, result_view);
  std::cout << "before deep_copy" << std::endl;
  Kokkos::deep_copy(exec, result, result_view);
    std::cout << "after deep_copy" << std::endl;
  exec.fence();
    ASSERT_EQ(exec.get_status(), Kokkos::Experimental::ExecutionSpaceStatus::complete);
  ASSERT_EQ(result, N/2*(N-1));
}
};

TEST(TEST_CATEGORY, execution_space_status)
{
  CheckExecutionSpaceStatus<TEST_EXECSPACE>();
}

}  // namespace
