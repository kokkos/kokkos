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

#ifndef LIB_WITH_INTERFACE_DEPENDENCY_ON_LIB_WITH_PRIVATE_KOKKOS_DEPENDENCY
#define LIB_WITH_INTERFACE_DEPENDENCY_ON_LIB_WITH_PRIVATE_KOKKOS_DEPENDENCY

#include <lib_with_private_kokkos_dependency.h>
#include <iostream>

namespace lib_with_interface_dependency_on_lib_with_private_kokkos_dependency {

void print(
    lib_with_private_kokkos_dependency::StructOfLibWithPrivateKokkosDependency
        in) {
  std::cout << "Hello from "
               "lib_with_interface_dependency_on_lib_with_private_kokkos_"
               "dependency\n";
  std::cout << "Will call lib_with_private_kokkos_dependency now:\n";
  lib_with_private_kokkos_dependency::print(in);
  std::cout << "Done\n";
}

}  // namespace
   // lib_with_interface_dependency_on_lib_with_private_kokkos_dependency
#endif
