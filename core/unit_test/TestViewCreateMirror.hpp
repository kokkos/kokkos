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

// Test that the result of create_mirror is assignable to host views
TEST(TEST_CATEGORY, create_mirror_assign_to_host) {
  Kokkos::View<float*, TEST_EXECSPACE> a("A", 10);
  Kokkos::View<float*, Kokkos::HostSpace> h_a1 = Kokkos::create_mirror(a);
  Kokkos::View<float*, Kokkos::HostSpace> h_a2 =
      Kokkos::create_mirror(Kokkos::HostSpace(), a);
}

// Test that the result of create_mirror_view is assignable to host views
TEST(TEST_CATEGORY, create_mirror_view_assign_to_host) {
  Kokkos::View<float*, TEST_EXECSPACE> a("A", 10);
  Kokkos::View<float*, Kokkos::HostSpace> h_a1 = Kokkos::create_mirror_view(a);
  Kokkos::View<float*, Kokkos::HostSpace> h_a2 =
      Kokkos::create_mirror_view(Kokkos::HostSpace(), a);
}
