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

#ifndef LIB_WITH_PUBLIC_DEPENDENCY_ON_LIB_WITH_INTERFACE_KOKKOS_DEPENDENCY
#define LIB_WITH_PUBLIC_DEPENDENCY_ON_LIB_WITH_INTERFACE_KOKKOS_DEPENDENCY

#include <lib_with_interface_kokkos_dependency.h>

namespace lib_with_public_dependency_on_lib_with_interface_kokkos_dependency {

void print(
    lib_with_interface_kokkos_dependency::StructOfLibWithInterfaceKokkosDependency<Kokkos::View<int*>>
        in);

}  // namespace lib_with_public_dependency_on_lib_with_interface_kokkos_dependency
#endif
