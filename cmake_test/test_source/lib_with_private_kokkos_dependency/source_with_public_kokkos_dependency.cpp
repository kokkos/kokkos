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

#include "header_with_private_kokkos_dependency.h"
#include <Kokkos_Core.hpp>
#include <iostream>

namespace lib_with_private_kokkos_dependency {

static bool i_initialized_kokkos = false;

bool is_initialized() { return Kokkos::is_initialized(); }
bool is_finalized() { return Kokkos::is_finalized(); }

void initialize() {
  // if I have to initialize kokkos, I assume I also have to finalize after I
  // did what I needed Kokkos for
  if (!Kokkos::is_initialized()) {
    Kokkos::initialize();
    i_initialized_kokkos = true;
  }
}

void finalize() {
  if (i_initialized_kokkos and !Kokkos::is_finalized()) {
    Kokkos::finalize();
  }
}

void print_kokkos() {
  std::cout << "Hello from a kokkos function within "
               "lib_with_private_kokkos_dependency\n";
  Kokkos::print_configuration(std::cout);
}
}  // namespace lib_with_private_kokkos_dependency
