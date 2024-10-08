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

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  if (argc != 4 || std::string(argv[1]) != "one" ||
      std::string(argv[2]) != "2" || std::string(argv[3]) != "THREE") {
    std::cerr << "must be called as `<exe> one 2 THREE`\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
