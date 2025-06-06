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

#include <Kokkos_Core.hpp>

namespace Test {

template <typename DstTraits, typename SrcTraits>
void impl_view_memory_trait_assign() {
  Kokkos::View<double *, DstTraits> a =
      Kokkos::View<double *, SrcTraits>("a", 1);
}

void view_memory_trait_assign() {
  impl_view_memory_trait_assign<void, Kokkos::MemoryTraits<Kokkos::Restrict>>();
  impl_view_memory_trait_assign<void, Kokkos::MemoryTraits<Kokkos::Atomic>>();

  impl_view_memory_trait_assign<Kokkos::MemoryTraits<Kokkos::Restrict>,
                                Kokkos::MemoryTraits<Kokkos::Atomic>>();
  impl_view_memory_trait_assign<Kokkos::MemoryTraits<Kokkos::Atomic>,
                                Kokkos::MemoryTraits<Kokkos::Restrict>>();
  impl_view_memory_trait_assign<
      Kokkos::MemoryTraits<Kokkos::Restrict | Kokkos::Unmanaged>, void>();
}

}  // namespace Test
