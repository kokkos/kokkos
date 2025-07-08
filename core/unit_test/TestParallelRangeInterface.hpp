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

namespace {

template <class Exec, class X, class Y>
KOKKOS_INLINE_FUNCTION void axpy(const Exec& exec, const X& x, const Y& y) {
  Kokkos::parallel_for(
      Kokkos::parallel_range(exec, 0, x.extent(0)),
      KOKKOS_LAMBDA(const int& i) { x(i) += y(i); });
}

void test_parallel_range_interface() {
  int N = 10000;
  int M = 100;

  Kokkos::View<float*> VA("VA", N), VB("VB", N);
  Kokkos::View<float**> MA("MA", N / M, M), MB("MB", N / M, M);
  Kokkos::deep_copy(VA, 1);
  Kokkos::deep_copy(VB, 2);
  Kokkos::deep_copy(MA, 1);
  Kokkos::deep_copy(MB, 2);

  // Call axpy from host
  axpy(Kokkos::DefaultExecutionSpace(), VA, VB);

  // call axpy from device with team handle
  using team_t = typename Kokkos::TeamPolicy<>::member_type;
  Kokkos::parallel_for(
      "apxyFromTeam", Kokkos::TeamPolicy(N / M, Kokkos::AUTO()),
      KOKKOS_LAMBDA(const team_t& team) {
        axpy(team, Kokkos::subview(MA, team.league_rank(), Kokkos::ALL()),
             Kokkos::subview(MB, team.league_rank(), Kokkos::ALL()));
      });

  // check
  size_t result = 0;
  Kokkos::parallel_reduce(
      "Check1", VA.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) { val += VA(i); }, result);
  ASSERT_EQ(result, size_t(3) * VA.extent(0));
  Kokkos::parallel_reduce(
      "Check2", MA.extent(0),
      KOKKOS_LAMBDA(int i, size_t& val) {
        for (int j = 0; j < MA.extent(1); j++) val += MA(i, j);
      },
      result);
  ASSERT_EQ(result, size_t(3) * MA.extent(0) * MA.extent(1));
}

TEST(TEST_CATEGORY, parallel_range_interface) {
  test_parallel_range_interface();
}
}  // namespace
