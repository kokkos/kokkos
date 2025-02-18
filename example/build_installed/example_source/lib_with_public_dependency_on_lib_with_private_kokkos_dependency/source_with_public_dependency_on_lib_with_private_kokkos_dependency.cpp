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

#include "lib_with_public_dependency_on_lib_with_private_kokkos_dependency.h"
#include <iostream>

namespace lib_with_public_dependency_on_lib_with_private_kokkos_dependency {

static bool i_initialized_lib_with_private_kokkos_dependency = false;

void initialize() {
  if (!lib_with_private_kokkos_dependency::is_initialized()) {
    lib_with_private_kokkos_dependency::initialize();
    i_initialized_lib_with_private_kokkos_dependency = true;
  }
}

void finalize() {
  if (i_initialized_lib_with_private_kokkos_dependency and
      !lib_with_private_kokkos_dependency::is_finalized())
    lib_with_private_kokkos_dependency::finalize();
}

void print(lib_with_private_kokkos_dependency::
               StructOfLibWithPrivateKokkosDependency in) {
  std::cout
      << "Hello from "
         "lib_with_public_dependency_on_lib_with_private_kokkos_dependency\n";
  std::cout << "Will call lib_with_private_kokkos_dependency now:\n";
  lib_with_private_kokkos_dependency::print(in);
  std::cout << "Done\n";
}

}  // namespace lib_with_public_dependency_on_lib_with_private_kokkos_dependency
