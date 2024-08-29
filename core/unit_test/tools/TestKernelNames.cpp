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

#include <Kokkos_Core.hpp>

#include <gtest/gtest.h>

namespace {

std::string last_parallel_for;
std::string last_parallel_reduce;
std::string last_parallel_scan;

void get_parallel_for_kernel_name(char const* kernelName, uint32_t /*deviceID*/,
                                  uint64_t* /*kernelID*/) {
  last_parallel_for = kernelName;
}

void get_parallel_reduce_kernel_name(char const* kernelName,
                                     uint32_t /*deviceID*/,
                                     uint64_t* /*kernelID*/) {
  last_parallel_reduce = kernelName;
}

void get_parallel_scan_kernel_name(char const* kernelName,
                                   uint32_t /*deviceID*/,
                                   uint64_t* /*kernelID*/) {
  last_parallel_scan = kernelName;
}

struct WorkTag {};

void test_kernel_name_parallel_for() {
  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(
      get_parallel_for_kernel_name);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  {
    std::string my_label = "my_parallel_for_range_policy";

    auto my_lambda = KOKKOS_LAMBDA(int){};
    Kokkos::parallel_for(my_label, Kokkos::RangePolicy<ExecutionSpace>(0, 1),
                         my_lambda);
    ASSERT_EQ(last_parallel_for, my_label);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace>(0, 1), my_lambda);
    ASSERT_EQ(last_parallel_for, typeid(my_lambda).name());

    auto my_lambda_with_tag = KOKKOS_LAMBDA(WorkTag, int){};
    Kokkos::parallel_for(my_label,
                         Kokkos::RangePolicy<ExecutionSpace, WorkTag>(0, 1),
                         my_lambda_with_tag);
    ASSERT_EQ(last_parallel_for, my_label);

    Kokkos::parallel_for(Kokkos::RangePolicy<ExecutionSpace, WorkTag>(0, 1),
                         my_lambda_with_tag);
    ASSERT_EQ(last_parallel_for,
              std::string(typeid(my_lambda_with_tag).name()) + "/" +
                  typeid(WorkTag).name());
  }

  Kokkos::Tools::Experimental::set_begin_parallel_for_callback(nullptr);
}

void test_kernel_name_parallel_reduce() {
  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(
      get_parallel_reduce_kernel_name);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  {
    std::string my_label = "my_parallel_reduce_range_policy";
    float my_result;

    auto my_lambda = KOKKOS_LAMBDA(int, float&){};
    Kokkos::parallel_reduce(my_label, Kokkos::RangePolicy<ExecutionSpace>(0, 1),
                            my_lambda, my_result);
    ASSERT_EQ(last_parallel_reduce, my_label);

    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace>(0, 1),
                            my_lambda, my_result);
    ASSERT_NE(last_parallel_reduce.find(typeid(my_lambda).name()),
              std::string::npos)
        << last_parallel_reduce << " does not contain "
        << typeid(my_lambda)
               .name();  // internally using
                         // Impl::CombinedFunctorReducer but the name should
                         // still include the lambda as template parameter

    auto my_lambda_with_tag = KOKKOS_LAMBDA(WorkTag, int, float&){};
    Kokkos::parallel_reduce(my_label,
                            Kokkos::RangePolicy<ExecutionSpace, WorkTag>(0, 1),
                            my_lambda_with_tag, my_result);
    ASSERT_EQ(last_parallel_reduce, my_label);

    Kokkos::parallel_reduce(Kokkos::RangePolicy<ExecutionSpace, WorkTag>(0, 1),
                            my_lambda_with_tag, my_result);
    auto suffix = std::string("/") + typeid(WorkTag).name();
    ASSERT_EQ(last_parallel_reduce.find(suffix),
              last_parallel_reduce.length() - suffix.length());
  }

  Kokkos::Tools::Experimental::set_begin_parallel_reduce_callback(nullptr);
}

void test_kernel_name_parallel_scan() {
  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(
      get_parallel_scan_kernel_name);

  using ExecutionSpace = Kokkos::DefaultExecutionSpace;

  {
    std::string my_label = "my_parallel_scan_range_policy";

    auto my_lambda = KOKKOS_LAMBDA(int, float&, bool){};
    Kokkos::parallel_scan(my_label, Kokkos::RangePolicy<ExecutionSpace>(0, 1),
                          my_lambda);
    ASSERT_EQ(last_parallel_scan, my_label);

    Kokkos::parallel_scan(Kokkos::RangePolicy<ExecutionSpace>(0, 1), my_lambda);
    ASSERT_EQ(last_parallel_scan, typeid(my_lambda).name());

    auto my_lambda_with_tag = KOKKOS_LAMBDA(WorkTag, int, float&, bool){};
    Kokkos::parallel_scan(my_label,
                          Kokkos::RangePolicy<ExecutionSpace, WorkTag>(0, 1),
                          my_lambda_with_tag);
    ASSERT_EQ(last_parallel_scan, my_label);

    Kokkos::parallel_scan(Kokkos::RangePolicy<ExecutionSpace, WorkTag>(0, 1),
                          my_lambda_with_tag);
    ASSERT_EQ(last_parallel_scan,
              std::string(typeid(my_lambda_with_tag).name()) + "/" +
                  typeid(WorkTag).name());
  }

  Kokkos::Tools::Experimental::set_begin_parallel_scan_callback(nullptr);
}

TEST(defaultdevicetype, kernel_name_parallel_for) {
  test_kernel_name_parallel_for();
}

TEST(defaultdevicetype, kernel_name_parallel_reduce) {
  test_kernel_name_parallel_reduce();
}

TEST(defaultdevicetype, kernel_name_parallel_scan) {
  test_kernel_name_parallel_scan();
}

}  // namespace
