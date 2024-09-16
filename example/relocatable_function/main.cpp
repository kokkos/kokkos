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

#include <iostream>

KOKKOS_RELOCATABLE_FUNCTION void count_even(const long i, long& lcount);

int main() {
  Kokkos::ScopeGuard scope_guard;

  for (int n = 10; n <= 100'000'000; n *= 10) {
    Kokkos::Timer timer;

    long count = 0;
    // Compute the number of even integers from 0 to n-1, in parallel.
    Kokkos::parallel_reduce(
        n, KOKKOS_LAMBDA(const long i, long& lcount) { count_even(i, lcount); },
        count);

    double count_time_rdc = timer.seconds();

    timer.reset();

    // Compute the number of even integers from 0 to n-1, in parallel.
    Kokkos::parallel_reduce(
        n,
        KOKKOS_LAMBDA(const long i, long& lcount) { lcount += (i % 2) == 0; },
        count);

    double count_time_no_rdc = timer.seconds();
    std::cout << std::scientific << n * 1. << ' ' << count_time_rdc
              << "s (RDC) vs. " << count_time_no_rdc << "s (inline)\n";
  }
}
