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

#ifndef HEADER_WITH_PRIVATE_KOKKOS_DEPENDENCY
#define HEADER_WITH_PRIVATE_KOKKOS_DEPENDENCY

namespace lib_with_private_kokkos_dependency {

bool is_initialized();

bool is_finalized();

void initialize();

void finalize();

void print_kokkos();

}  // namespace lib_with_private_kokkos_dependency
#endif
