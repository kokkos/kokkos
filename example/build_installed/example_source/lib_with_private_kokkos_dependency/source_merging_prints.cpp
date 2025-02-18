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

#include "lib_with_private_kokkos_dependency.h"

namespace lib_with_private_kokkos_dependency {

void print([[maybe_unused]] StructOfLibWithPrivateKokkosDependency in) {

  print_non_kokkos();
  print_kokkos();
}

}  // namespace lib_with_private_kokkos_dependency
