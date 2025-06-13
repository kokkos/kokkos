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

// Check that moving a View outside a parallel region does not increase the
// number of views managing the allocation.
template <class ViewType>
void test_moving_view_does_not_change_use_count(ViewType v) {
  auto* const ptr = v.data();
  auto const cnt  = v.use_count();

  ViewType w(std::move(v));  // move construction
  EXPECT_EQ(w.use_count(), cnt);
  EXPECT_EQ(w.data(), ptr);
  EXPECT_EQ(v.use_count(), 0);
  EXPECT_EQ(v.data(), nullptr);

  v = std::move(w);  // move assignment
  EXPECT_EQ(v.use_count(), cnt);
  EXPECT_EQ(v.data(), ptr);
  EXPECT_EQ(w.use_count(), 0);
  EXPECT_EQ(w.data(), nullptr);
}

TEST(TEST_CATEGORY, view_move_and_use_count) {
  using ExecutionSpace = TEST_EXECSPACE;

  test_moving_view_does_not_change_use_count(
      Kokkos::View<int, ExecutionSpace>("v0"));

  test_moving_view_does_not_change_use_count(
      Kokkos::View<float*, ExecutionSpace>("v1", 1));

  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  test_moving_view_does_not_change_use_count(
      Kokkos::View<double**, ExecutionSpace>(v2.data(), v2.extent(0),
                                             v2.extent(1)));
  test_moving_view_does_not_change_use_count(
      Kokkos::View<double**, ExecutionSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
          v2.data(), v2.extent(0), v2.extent(1)));
}

// Check that moving a View leaves the moved-from object in a state equivalent
// to being default constructed
// returns the number of errors encountered
template <class ViewType>
KOKKOS_FUNCTION int check_moved_from_view_state(ViewType v) {
  int err = 0;

  ViewType w(std::move(v));  // move construction
  if (v != ViewType()) {
    Kokkos::printf("failed moved-from view after calling move constructor\n");
    ++err;
  }

  v = std::move(w);  // move assignment
  if (w != ViewType()) {
    Kokkos::printf(
        "failed moved-from view after calling move assignment operator\n");
    ++err;
  }

  return err;
}

template <class ViewType>
void test_moved_from_view(ViewType v) {
  EXPECT_EQ(check_moved_from_view_state(v), 0) << "outside parallel region";

  using ExexutionSpace = typename ViewType::execution_space;
  int errors;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExexutionSpace>(0, 1),
      KOKKOS_LAMBDA(int, int& err) { err += check_moved_from_view_state(v); },
      errors);
  EXPECT_EQ(errors, 0) << "within parallel region";
}

TEST(TEST_CATEGORY, view_moved_from) {
  using ExecutionSpace = TEST_EXECSPACE;

  test_moved_from_view(Kokkos::View<int, ExecutionSpace>("v0"));
  test_moved_from_view(Kokkos::View<float*, ExecutionSpace>("v1", 1));
  Kokkos::View<double**, ExecutionSpace> v2("v2", 1, 2);
  test_moved_from_view(Kokkos::View<double**, ExecutionSpace>(
      v2.data(), v2.extent(0), v2.extent(1)));
  test_moved_from_view(Kokkos::View<double**, ExecutionSpace,
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
      v2.data(), v2.extent(0), v2.extent(1)));
}

}  // namespace
