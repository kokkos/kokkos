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

#include <cstdio>
#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#endif

namespace cuda_hip_functions_without_kokkos_dependency {

__global__ void print_from_device() { printf("Hello, from a cuda function!\n"); }

}  // namespace cuda_hip_functions_without_kokkos_dependency