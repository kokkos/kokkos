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

#include <cstdio>
#include <iostream>

KOKKOS_RELOCATABLE_FUNCTION void count_even(const long i, long& lcount);

int main() {
  Kokkos::ScopeGuard scope_guard;

  const long n = 10000;

  std::cout << "Number of even integers from 0 to " << n - 1 << '\n';

  Kokkos::Timer timer;
  timer.reset();

  // Compute the number of even integers from 0 to n-1, in parallel.
  long count = 0;
  Kokkos::parallel_reduce(
      n, KOKKOS_LAMBDA(const long i, long& lcount) { count_even(i, lcount); },
      count);

  double count_time = timer.seconds();
  std::cout << "Parallel: " << count << ", " << count_time << "s\n";

  timer.reset();

  // Compare to a sequential loop.
  long seq_count = 0;
  for (long i = 0; i < n; ++i) {
    seq_count += (i % 2) == 0;
  }

  count_time = timer.seconds();
  std::cout << "Sequential: " << seq_count << ", " << count_time << "s\n";

  return (count == seq_count) ? 0 : -1;
}
