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

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#define KOKKOS_IMPL_PUBLIC_INCLUDE
#endif

#include <Kokkos_Core.hpp>
#include <Cuda/Kokkos_Cuda_UniqueToken.hpp>

namespace Kokkos::Experimental {

UniqueToken<Cuda, UniqueTokenScope::Global>::UniqueToken(size_type max_size) {
  m_locks = Kokkos::View<uint32_t*, Kokkos::CudaSpace>(
      std::string("Kokkos::UniqueToken::m_locks"), max_size);
}

UniqueToken<Cuda, UniqueTokenScope::Global>::UniqueToken(size_type max_size,
                                                         Cuda const& exec) {
  m_locks = Kokkos::View<uint32_t*, Kokkos::CudaSpace>(
      Kokkos::view_alloc(exec, std::string("Kokkos::UniqueToken::m_locks")),
      max_size);
}

}  // namespace Kokkos::Experimental
