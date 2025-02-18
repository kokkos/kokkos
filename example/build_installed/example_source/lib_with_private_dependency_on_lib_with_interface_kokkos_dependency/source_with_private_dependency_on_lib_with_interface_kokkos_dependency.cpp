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

#include "lib_with_private_dependency_on_lib_with_interface_kokkos_dependency.h"
#include <lib_with_interface_kokkos_dependency.h>
#include <iostream>

namespace lib_with_private_dependency_on_lib_with_interface_kokkos_dependency {

static bool i_initialized_lib_with_interface_kokkos_dependency = false;

void initialize() {
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
    i_initialized_lib_with_interface_kokkos_dependency = true;
  }
}

void finalize() {
  if (i_initialized_lib_with_interface_kokkos_dependency and
      !Kokkos::is_finalized())
    Kokkos::finalize();
}

void print() {
  std::cout
      << "Hello from "
         "lib_with_private_dependency_on_lib_with_interface_kokkos_dependency\n";
  std::cout << "Will call lib_with_interface_kokkos_dependency now:\n";
  lib_with_interface_kokkos_dependency::print(Kokkos::View<int*>{"a", 10});
  std::cout << "Done\n";
}

}  // namespace lib_with_private_dependency_on_lib_with_interface_kokkos_dependency
