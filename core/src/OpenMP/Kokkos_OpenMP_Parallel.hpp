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

#ifndef KOKKOS_OPENMP_PARALLEL_HPP
#define KOKKOS_OPENMP_PARALLEL_HPP

#include <Kokkos_Macros.hpp>
#if defined(KOKKOS_ENABLE_OPENMP)

#include <omp.h>
#include <OpenMP/Kokkos_OpenMP_Instance.hpp>

#include <KokkosExp_MDRangePolicy.hpp>

namespace Kokkos {
namespace Impl {

inline bool execute_in_serial(OpenMP const& space = OpenMP()) {
  return (OpenMP::in_parallel(space) &&
          !(omp_get_nested() && (omp_get_level() == 1)));
}

}  // namespace Impl
}  // namespace Kokkos

#endif /* KOKKOS_ENABLE_OPENMP */
#endif /* KOKKOS_OPENMP_PARALLEL_HPP */
