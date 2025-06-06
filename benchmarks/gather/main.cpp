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
#include <Kokkos_Timer.hpp>
#include "gather.hpp"
#include <cstdlib>

int main(int argc, char* argv[]) {  // NOLINT(bugprone-exception-escape)
  Kokkos::initialize(argc, argv);

  if (argc < 8) {
    printf("Arguments: S N K D\n");
    printf(
        "  S:   Scalar Type Size (1==float, 2==double, 4=complex<double>)\n");
    printf("  N:   Number of entities\n");
    printf("  K:   Number of things to gather per entity\n");
    printf("  D:   Max distance of gathered things of an entity\n");
    printf("  R:   how often to loop through the K dimension with each team\n");
    printf("  U:   how many independent flops to do per load\n");
    printf(
        "  F:   how many times to repeat the U unrolled operations before "
        "reading next element\n");
    printf("Example Input GPU:\n");
    printf("  Bandwidth Bound : 2 10000000 1 1 10 1 1\n");
    printf("  Cache Bound     : 2 10000000 64 1 10 1 1\n");
    printf("  Cache Gather    : 2 10000000 64 256 10 1 1\n");
    printf("  Global Gather   : 2 100000000 16 100000000 1 1 1\n");
    printf("  Typical MD      : 2 100000 32 512 1000 8 2\n");
    Kokkos::finalize();
    return 0;
  }

  int S = std::stoi(argv[1]);
  int N = std::stoi(argv[2]);
  int K = std::stoi(argv[3]);
  int D = std::stoi(argv[4]);
  int R = std::stoi(argv[5]);
  int U = std::stoi(argv[6]);
  int F = std::stoi(argv[7]);

  if ((S != 1) && (S != 2) && (S != 4)) {
    printf("S must be one of 1,2,4\n");
    return 0;
  }
  if (N < D) {
    printf("N must be larger or equal to D\n");
    return 0;
  }
  if (S == 1) {
    run_gather_test<float>(N, K, D, R, U, F);
  }
  if (S == 2) {
    run_gather_test<double>(N, K, D, R, U, F);
  }
  if (S == 4) {
    run_gather_test<Kokkos::complex<double> >(N, K, D, R, U, F);
  }
  Kokkos::finalize();
}
