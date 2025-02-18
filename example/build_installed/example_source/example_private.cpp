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

#include <lib_without_kokkos_dependency.h>
#include <lib_with_private_kokkos_dependency.h>

#include <cstdio>
#include <iostream>

extern "C" void print_fortran_();
void print_plain_cxx();

int main() {
  lib_without_kokkos_dependency::print();

  lib_with_private_kokkos_dependency::initialize();
  {
  lib_with_private_kokkos_dependency::print(lib_with_private_kokkos_dependency::StructOfLibWithPrivateKokkosDependency{});
  }
  lib_with_private_kokkos_dependency::finalize();

  print_fortran_();
  print_plain_cxx();
}
