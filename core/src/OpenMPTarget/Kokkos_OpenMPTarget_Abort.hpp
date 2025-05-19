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

#ifndef KOKKOS_OPENMPTARGET_ABORT_HPP
#define KOKKOS_OPENMPTARGET_ABORT_HPP

#include <Kokkos_Macros.hpp>
#ifdef KOKKOS_ENABLE_OPENMPTARGET

namespace Kokkos {
namespace Impl {

KOKKOS_INLINE_FUNCTION void OpenMPTarget_abort(char const *msg) {
  fprintf(stderr, "%s.\n", msg);
  std::abort();
}

}  // namespace Impl
}  // namespace Kokkos

#endif
#endif
