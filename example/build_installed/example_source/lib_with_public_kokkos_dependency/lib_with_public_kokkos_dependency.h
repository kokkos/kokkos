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

#ifndef LIB_WITH_PUBLIC_KOKKOS_DEPENDENCY
#define LIB_WITH_PUBLIC_KOKKOS_DEPENDENCY

#include <Kokkos_Core.hpp>

namespace lib_with_public_kokkos_dependency {

void print(Kokkos::View<int*> a);

struct StructOfLibWithPublicKokkosDependency {
  Kokkos::View<int*> value;
};

}  // namespace lib_with_public_kokkos_dependency

#endif
