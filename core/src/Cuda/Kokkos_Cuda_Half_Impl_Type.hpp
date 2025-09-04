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

#ifndef KOKKOS_CUDA_HALF_IMPL_TYPE_HPP_
#define KOKKOS_CUDA_HALF_IMPL_TYPE_HPP_

#include <Kokkos_Macros.hpp>

#if !(defined(KOKKOS_ARCH_KEPLER) || defined(KOKKOS_ARCH_MAXWELL50) || \
      defined(KOKKOS_ARCH_MAXWELL52))
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef KOKKOS_IMPL_HALF_TYPE_DEFINED
// Make sure no one else tries to define half_t
#define KOKKOS_IMPL_HALF_TYPE_DEFINED

namespace Kokkos::Impl {

struct half_impl_t {
  using type = __half;
};
#define KOKKOS_IMPL_BHALF_TYPE_DEFINED
struct bhalf_impl_t {
  using type = __nv_bfloat16;
};

}  // namespace Kokkos::Impl

#endif  // KOKKOS_IMPL_HALF_TYPE_DEFINED
#endif  // Disables for half_t on cuda:
        // KEPLER30||KEPLER32||KEPLER37||MAXWELL50||MAXWELL52

#endif
