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

int main(int argc, char* argv[]) {
  Kokkos::ScopeGuard scope_guard(argc, argv);

  for (int n = 10; n <= 100'000'000; n *= 10) {
    Kokkos::Timer timer;

    long count = 0;
    // Compute the number of even integers from 0 to n-1 using a relocatable
    // functor
    Kokkos::parallel_reduce(
        n, KOKKOS_LAMBDA(const long i, long& lcount) { count_even(i, lcount); },
        count);

    double count_time_relocatable = timer.seconds();

    timer.reset();

    // Compute the number of even integers from 0 to n-1 using an inline lambda
    Kokkos::parallel_reduce(
        n,
        KOKKOS_LAMBDA(const long i, long& lcount) { lcount += (i % 2) == 0; },
        count);

    double count_time_inline = timer.seconds();
    std::cout << std::scientific << n * 1. << ' ' << count_time_relocatable
              << "s (relocatable) vs. " << count_time_inline << "s (inline)\n";
  }
}
